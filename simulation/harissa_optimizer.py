"""
harissa_optimizer.py
====================
Optimise HARISSA parameters (basal, inter) for a fixed GRN structure.

HARISSA deterministic limit (K0=0):
    dP_i/dt = D1_i · (sigmoid(basal_i + P @ inter[:,i]) − P_i)
    P ∈ [0,1]^{G+1}; P[0] = stimulus (clamped at 1.0 ON / 0.0 OFF).

Design philosophy
-----------------
• Resting state (no stimulus): all real genes near 0.
  Enforced by: basal_i << 0 (starts at −5) + resting_weight penalty.

• Active states (stimulus ON): stable FPs found by gradient descent.
  Three-component loss:
    1. fp_loss        – ‖dP/dt‖² at target (zero iff target is a FP)
    2. contraction    – perturb target → short rollout → returns (local stability)
    3. traj_loss      – start at resting+stim, integrate, compare to target

• Cascade is handled by a dedicated temporal-checkpoint loss, not FP/contraction.

Biological scenarios (make_scenario)
-------------------------------------
activation    : single cluster activates from rest (stimulus → C0)
switch        : bistable toggle stimulus drives C1, mutual TF inhibition C0↔C1
cascade       : feedforward relay with timescale separation (C0→C1→C2)
switch_cascade: relay + inhibitory feedback (mutually exclusive sequential states)
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from grn_generator import GRNStructure, make_sign_matrix


# ══════════════════════════════════════════════════════════════════════════════
# HARISSA ODE (differentiable)
# ══════════════════════════════════════════════════════════════════════════════

class HarissaODE(nn.Module):
    def __init__(self, G: int, d: np.ndarray):
        super().__init__()
        self.G = G
        self.register_buffer("D1", torch.tensor(d[1], dtype=torch.float32))

    def dPdt(self, P, basal, inter):
        s  = torch.sigmoid(basal + P @ inter)
        dP = self.D1 * (s - P)
        dP = dP.clone(); dP[..., 0] = 0.0
        return dP

    def integrate(self, P0, basal, inter,
                  T=100.0, n_steps=500, stimulus_level=1.0):
        dt = T / n_steps
        P  = P0.clone()
        for _ in range(n_steps):
            P = P + dt * self.dPdt(P, basal, inter)
            P = P.clone(); P[..., 0] = stimulus_level
        return P

    def integrate_trajectory(self, P0, basal, inter,
                              T=200.0, n_steps=1000, n_save=100,
                              stimulus_level=1.0):
        dt         = T / n_steps
        save_every = max(1, n_steps // n_save)
        P    = P0.clone()
        traj = [P.unsqueeze(0)]
        for step in range(1, n_steps + 1):
            P = P + dt * self.dPdt(P, basal, inter)
            P = P.clone(); P[..., 0] = stimulus_level
            if step % save_every == 0 and len(traj) <= n_save:
                traj.append(P.unsqueeze(0))
        return torch.cat(traj[:n_save + 1], dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CellState:
    active_clusters: List[int]
    name: str = ""
    stimulus_on: bool = True


@dataclass
class Transition:
    source: CellState
    target: CellState
    weight: float = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# GRN wiring helpers (used by make_scenario)
# ══════════════════════════════════════════════════════════════════════════════

def _wire_stimulus(grn, signs, target_clusters):
    """Stimulus → TFs of target_clusters (+1 sign)."""
    grn.inter_mask[0, :] = False
    signs[0, :] = 0
    for k in target_clusters:
        for g in grn.tfs_per_cluster[k]:
            grn.inter_mask[0, g + 1] = True
            signs[0, g + 1] = 1


def _wire_tfs_to_cluster_targets(grn, signs):
    """
    Ensure every TF activates all non-TF genes in its own cluster.
    This guarantees full within-cluster propagation.
    """
    for k in range(grn.n_clusters):
        for tf_g in grn.tfs_per_cluster[k]:
            for tgt_g in grn.genes_per_cluster[k]:
                if not grn.tf_mask[tgt_g]:
                    grn.inter_mask[tf_g + 1, tgt_g + 1] = True
                    signs[tf_g + 1, tgt_g + 1] = 1


def _add_relay_edges(grn, signs):
    """TFs of cluster k → TFs of cluster k+1 (positive, feedforward relay)."""
    for k in range(grn.n_clusters - 1):
        for src_g in grn.tfs_per_cluster[k]:
            for dst_g in grn.tfs_per_cluster[k + 1]:
                grn.inter_mask[src_g + 1, dst_g + 1] = True
                signs[src_g + 1, dst_g + 1] = 1


def _add_inhibitory_feedback(grn, signs):
    """TFs of cluster k+1 → TFs of cluster k (negative, toggle-switch feedback)."""
    for k in range(grn.n_clusters - 1):
        for src_g in grn.tfs_per_cluster[k + 1]:
            for dst_g in grn.tfs_per_cluster[k]:
                grn.inter_mask[src_g + 1, dst_g + 1] = True
                signs[src_g + 1, dst_g + 1] = -1


# ══════════════════════════════════════════════════════════════════════════════
# Kinetic parameter helpers
# ══════════════════════════════════════════════════════════════════════════════

def _default_kinetics(G, kp=None):
    if kp is not None:
        # Accept both dict {'a':..,'d':..} and tuple (a, d)
        if isinstance(kp, dict):
            return kp["a"], kp["d"]
        else:
            return kp[0], kp[1]
    a = np.zeros((3, G + 1), dtype=np.float32)
    a[1] = 2.0; a[2] = 0.02
    d = np.zeros((2, G + 1), dtype=np.float32)
    d[0] = 0.2; d[1] = 0.04
    return a, d


def _cascade_kinetics(G, grn, n_clusters):
    """
    Assign progressively slower protein degradation (D1) to later clusters.
    This creates timescale separation driving the sequential cascade.
    Cluster 0: fastest (D1=0.10), …, last cluster: slowest (D1=0.04/n).
    """
    a, d = _default_kinetics(G)
    rates = np.linspace(0.10, 0.10 / n_clusters, n_clusters)
    for k in range(n_clusters):
        for g in grn.genes_per_cluster[k]:
            d[1][g + 1] = rates[k]
    return a, d


# ══════════════════════════════════════════════════════════════════════════════
# Scenario factory
# ══════════════════════════════════════════════════════════════════════════════

def make_scenario(grn: GRNStructure, scenario: str, seed: int = 0):
    """
    Configure GRN wiring, sign matrix, target states, transitions, and
    kinetic parameters for the requested biological scenario.

    Returns
    -------
    grn, signs, states, transitions, kinetic_params
      • states      : active CellStates to optimise as stable FPs
      • transitions : trajectory constraints (source → target)
      • kinetic_params : {'a', 'd'} arrays, or None for defaults
    """
    K = grn.n_clusters
    G = grn.n_genes
    signs = make_sign_matrix(grn, intra_positive_fraction=1.0,
                             inter_sign='negative', seed=seed)
    # Guarantee TF→targets wiring in every cluster
    _wire_tfs_to_cluster_targets(grn, signs)

    resting = CellState([], 'resting', stimulus_on=True)

    if scenario == 'activation':
        _wire_stimulus(grn, signs, [0])
        state_ON = CellState([0], 'C0_ON', stimulus_on=True)
        return grn, signs, [state_ON], [Transition(resting, state_ON)], None

    elif scenario == 'switch':
        assert K >= 2
        _wire_stimulus(grn, signs, [1])
        _add_inhibitory_feedback(grn, signs)   # C1 TFs inhibit C0 TFs
        state_B = CellState([1], 'B_C1', stimulus_on=True)
        return grn, signs, [state_B], [Transition(resting, state_B)], None

    elif scenario == 'cascade':
        _wire_stimulus(grn, signs, [0])
        _add_relay_edges(grn, signs)
        # Checkpoint states: each cluster active exclusively
        states = [CellState([k], f'C{k}', stimulus_on=True) for k in range(K)]
        # Trajectory: resting→C0, C0→C1, …, C(K-2)→C(K-1)
        transitions = [Transition(resting, states[0])]
        for k in range(K - 1):
            transitions.append(Transition(states[k], states[k + 1]))
        kp = _cascade_kinetics(G, grn, K)
        return grn, signs, states, transitions, kp

    elif scenario == 'switch_cascade':
        assert K >= 3
        _wire_stimulus(grn, signs, [0])
        _add_relay_edges(grn, signs)
        _add_inhibitory_feedback(grn, signs)
        states = [CellState([k], f'C{k}', stimulus_on=True) for k in range(K)]
        transitions = [Transition(resting, states[0])]
        for k in range(K - 1):
            transitions.append(Transition(states[k], states[k + 1]))
        kp = _cascade_kinetics(G, grn, K)
        return grn, signs, states, transitions, kp

    else:
        raise ValueError(f"Unknown scenario '{scenario}'. "
                         "Choose: activation, switch, cascade, switch_cascade")


# ══════════════════════════════════════════════════════════════════════════════
# Parameter optimizer
# ══════════════════════════════════════════════════════════════════════════════

class HarissaParameterOptimizer:
    """
    Gradient-based optimiser for HARISSA basal + inter.

    For cascade/switch_cascade scenarios, intermediate states are NOT
    enforced as stable FPs (which is biologically wrong — they are transient).
    Instead, a temporal-checkpoint loss evaluates the ODE trajectory at
    evenly-spaced timepoints and penalises wrong cluster ordering.
    Only the FINAL cluster state is enforced as a stable FP.
    """

    def __init__(
        self,
        grn: GRNStructure,
        sign_matrix: np.ndarray,
        cell_states: List[CellState],
        transitions: List[Transition],
        kinetic_params: Optional[Dict] = None,
        device: str = "cpu",
        scenario: str = "activation",
    ):
        self.grn          = grn
        self.cell_states  = cell_states
        self.transitions  = transitions
        self.device       = device
        self.scenario     = scenario
        G = grn.n_genes

        self.a, self.d = _default_kinetics(G, kinetic_params)
        self.ode = HarissaODE(G, self.d).to(device)

        self.inter_mask  = torch.tensor(
            grn.inter_mask.astype(np.float32), device=device)
        self.sign_tensor = torch.tensor(
            sign_matrix.astype(np.float32), device=device)

        self.basal_raw:     Optional[nn.Parameter] = None
        self.inter_log_mag: Optional[nn.Parameter] = None

    # ── Protein targets ───────────────────────────────────────────────────────

    def _target_P(self, state: CellState) -> torch.Tensor:
        G  = self.grn.n_genes
        P  = torch.full((G + 1,), 0.01, dtype=torch.float32, device=self.device)
        P[0] = 1.0 if state.stimulus_on else 0.0
        for k in state.active_clusters:
            for g in self.grn.genes_per_cluster[k]:
                P[g + 1] = 1.0
        return P

    def resting_state(self) -> torch.Tensor:
        """All real genes near 0, stimulus ON (P=1)."""
        G  = self.grn.n_genes
        P  = torch.full((G + 1,), 0.01, dtype=torch.float32, device=self.device)
        P[0] = 1.0
        return P

    def _stim(self, state):
        return 1.0 if state.stimulus_on else 0.0

    # ── Parameters ───────────────────────────────────────────────────────────

    def _init_params(self, init_mag=5.0):
        G = self.grn.n_genes
        # All basals strongly negative (resting near 0)
        basal_init = torch.full((G + 1,), -5.0, dtype=torch.float32)
        basal_init[0] = 0.0
        self.basal_raw     = nn.Parameter(basal_init.to(self.device))
        self.inter_log_mag = nn.Parameter(
            torch.full((G + 1, G + 1), init_mag,
                       dtype=torch.float32, device=self.device))

    def _get_inter(self):
        mag = torch.nn.functional.softplus(self.inter_log_mag)
        return self.sign_tensor * mag * self.inter_mask

    def _basal(self):
        b = self.basal_raw.clone(); b[0] = 0.0
        return b

    # ── Loss components ───────────────────────────────────────────────────────

    def _fp_loss(self, state):
        target = self._target_P(state)
        dP = self.ode.dPdt(
            target.unsqueeze(0), self._basal(), self._get_inter()
        ).squeeze(0)
        return torch.mean(dP[1:] ** 2)

    def _contraction_loss(self, state, n_perturb, eps, T_short, n_steps):
        target = self._target_P(state)
        inter  = self._get_inter()
        basal  = self._basal()
        stim   = self._stim(state)
        losses = []
        for _ in range(n_perturb):
            noise  = torch.randn(self.grn.n_genes,
                                 dtype=torch.float32, device=self.device) * eps
            P_pert = target.clone()
            P_pert[1:] = (target[1:] + noise).clamp(0.0, 1.0)
            P_pert[0]  = stim
            P_fin = self.ode.integrate(P_pert.unsqueeze(0), basal, inter,
                                       T=T_short, n_steps=n_steps,
                                       stimulus_level=stim).squeeze(0)
            losses.append(torch.mean((P_fin[1:] - target[1:]) ** 2))
        return torch.stack(losses).mean()

    def _traj_loss(self, trans, T_transit, n_steps):
        P0    = self._target_P(trans.source)
        P_tgt = self._target_P(trans.target)
        P_fin = self.ode.integrate(
            P0.unsqueeze(0), self._basal(), self._get_inter(),
            T=T_transit, n_steps=n_steps,
            stimulus_level=self._stim(trans.target),
        ).squeeze(0)
        return torch.mean((P_fin[1:] - P_tgt[1:]) ** 2)

    def _cascade_checkpoint_loss(self, T_total, n_steps_total, n_checkpoints=None):
        """
        Temporal ordering loss for cascade scenarios.
        Evaluate ODE at K evenly-spaced checkpoints; at checkpoint k, cluster k
        should be the most active and preceding clusters should be fading.
        """
        K = len(self.cell_states)
        if n_checkpoints is None:
            n_checkpoints = K
        checkpoints = np.linspace(0.2, 1.0, n_checkpoints)   # fraction of T_total

        P0   = self.resting_state()
        traj = self.ode.integrate_trajectory(
            P0.unsqueeze(0), self._basal(), self._get_inter(),
            T=T_total, n_steps=n_steps_total,
            n_save=n_checkpoints * 10,
            stimulus_level=1.0,
        )  # (n_checkpoints*10+1, 1, G+1)

        loss = torch.zeros(1, device=self.device)
        n_saved = traj.shape[0] - 1
        for cp_i, frac in enumerate(checkpoints):
            t_idx = min(int(frac * n_saved), n_saved)
            P_t   = traj[t_idx, 0, :]   # (G+1,)
            # Expected dominant cluster at this checkpoint
            k_exp = cp_i % K
            gk    = self.grn.genes_per_cluster[k_exp]
            act_k = P_t[gk + 1].mean()

            # All other clusters should be lower
            for k_other in range(K):
                if k_other == k_exp:
                    continue
                gother   = self.grn.genes_per_cluster[k_other]
                act_other = P_t[gother + 1].mean()
                loss = loss + torch.relu(act_other - act_k + 0.1)

            # Also: active cluster should be actually active (> 0.5)
            loss = loss + torch.relu(0.5 - act_k)

        return loss / (n_checkpoints * K)

    def _resting_penalty(self):
        """
        Keep all gene basals strongly negative so resting state ≈ 0.
        Threshold −4: sigmoid(−4) ≈ 0.018, bistability still accessible via
        strong self-loops (inter_self > 4) while avoiding spurious no-stim activation.
        """
        b_real = self._basal()[1:]
        return torch.mean(torch.relu(b_real + 4.0) ** 2)

    def _reg_loss(self):
        return 5e-4 * torch.mean(self._get_inter() ** 2)

    # ── Optimisation ──────────────────────────────────────────────────────────

    def optimise(
        self,
        n_epochs: int          = 600,
        lr: float              = 0.05,
        fp_weight: float       = 2.0,
        ctr_weight: float      = 4.0,
        traj_weight: float     = 1.5,
        resting_weight: float  = 0.5,
        cascade_weight: float  = 2.0,
        n_perturb: int         = 4,
        eps_perturb: float     = 0.15,
        T_short: float         = 12.0,
        n_steps_short: int     = 100,
        T_transit: float       = 120.0,
        n_steps_traj: int      = 600,
        T_cascade: float       = 300.0,
        n_steps_cascade: int   = 1500,
        init_mag: float        = 5.0,
        grad_clip: float       = 5.0,
        verbose: bool          = True,
        log_every: int         = 60,
    ) -> Dict:
        self._init_params(init_mag)
        opt   = torch.optim.Adam([self.basal_raw, self.inter_log_mag], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs, eta_min=lr * 0.05)
        loss_history = []
        is_cascade = self.scenario in ('cascade', 'switch_cascade')

        for epoch in range(n_epochs):
            opt.zero_grad()
            total = torch.zeros(1, device=self.device)
            fp_v, ctr_v, traj_v = [], [], []

            # FP + contraction for ACTIVE (non-resting, non-intermediate) states
            # For cascade: only the FINAL state is a proper FP target
            fp_states = (self.cell_states[-1:] if is_cascade
                         else [s for s in self.cell_states if s.active_clusters])

            for state in fp_states:
                l_fp  = self._fp_loss(state)
                l_ctr = self._contraction_loss(
                    state, n_perturb, eps_perturb, T_short, n_steps_short)
                fp_v.append(l_fp.item()); ctr_v.append(l_ctr.item())
                total = total + fp_weight * l_fp + ctr_weight * l_ctr

            # Trajectory loss (resting→first active state, or chain for cascade)
            for tr in self.transitions[:1]:   # only first transition for cascade
                if not tr.target.active_clusters:
                    continue
                l_tr = self._traj_loss(tr, T_transit, n_steps_traj)
                traj_v.append(l_tr.item() * tr.weight)
                total = total + traj_weight * tr.weight * l_tr

            # Cascade checkpoint loss
            if is_cascade:
                l_casc = self._cascade_checkpoint_loss(T_cascade, n_steps_cascade)
                total  = total + cascade_weight * l_casc
                traj_v.append(l_casc.item())

            total = total + resting_weight * self._resting_penalty()
            total = total + self._reg_loss()
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [self.basal_raw, self.inter_log_mag], grad_clip)
            opt.step(); sched.step()
            loss_history.append(total.item())

            if verbose and (epoch % log_every == 0 or epoch == n_epochs - 1):
                fp_s  = " ".join(f"{v:.4f}" for v in fp_v)
                ctr_s = " ".join(f"{v:.4f}" for v in ctr_v)
                tr_s  = " ".join(f"{v:.4f}" for v in traj_v)
                print(f"Epoch {epoch:4d} | {total.item():.4f} "
                      f"fp=[{fp_s}] ctr=[{ctr_s}] traj=[{tr_s}]")

        with torch.no_grad():
            basal_opt = self._basal().detach().cpu().numpy()
            inter_opt = self._get_inter().detach().cpu().numpy()

        return {"basal": basal_opt, "inter": inter_opt,
                "a": self.a, "d": self.d, "loss_history": loss_history}

    # ── Verification & trajectories ───────────────────────────────────────────

    def verify_fixed_points(self, result, T=400.0, n_steps=2000):
        print("\n=== Verification ===")
        basal = torch.tensor(result["basal"], dtype=torch.float32, device=self.device)
        inter = torch.tensor(result["inter"], dtype=torch.float32, device=self.device)

        # Resting state
        P_r   = self.resting_state()
        P_rf  = self.ode.integrate(P_r.unsqueeze(0), basal, inter,
                                   T=T, n_steps=n_steps, stimulus_level=0.0).squeeze(0)
        rest_max = P_rf[1:].max().item()
        print(f"  Resting (no stim): max expression = {rest_max:.4f} "
              f"{'✓' if rest_max < 0.1 else '~'}")

        for state in self.cell_states:
            if not state.active_clusters:
                continue
            target = self._target_P(state)
            P_fin  = self.ode.integrate(target.unsqueeze(0), basal, inter,
                                        T=T, n_steps=n_steps,
                                        stimulus_level=self._stim(state)).squeeze(0)
            conv = torch.mean((P_fin[1:] - target[1:]) ** 2).item()
            ok   = "✓" if conv < 0.05 else ("~" if conv < 0.2 else "✗")
            print(f"  '{state.name}' clusters={state.active_clusters}: "
                  f"conv={conv:.4f}  {ok}")

    def simulate_from_resting(self, result, T=300.0, n_steps=1500, n_save=150):
        """Main trajectory: resting state + stimulus ON."""
        basal = torch.tensor(result["basal"], dtype=torch.float32, device=self.device)
        inter = torch.tensor(result["inter"], dtype=torch.float32, device=self.device)
        P0    = self.resting_state()
        traj  = self.ode.integrate_trajectory(
            P0.unsqueeze(0), basal, inter,
            T=T, n_steps=n_steps, n_save=n_save, stimulus_level=1.0)
        times  = np.linspace(0, T, n_save + 1)
        P_traj = traj[:, 0, :].detach().cpu().numpy()
        return times, P_traj

    def simulate_transition(self, result, source, T=300.0, n_steps=1500,
                            n_save=100, target_stimulus=None):
        basal = torch.tensor(result["basal"], dtype=torch.float32, device=self.device)
        inter = torch.tensor(result["inter"], dtype=torch.float32, device=self.device)
        P0    = self._target_P(source)
        stim  = (target_stimulus if target_stimulus is not None else self._stim(source))
        traj  = self.ode.integrate_trajectory(
            P0.unsqueeze(0), basal, inter,
            T=T, n_steps=n_steps, n_save=n_save, stimulus_level=stim)
        return np.linspace(0, T, n_save + 1), traj[:, 0, :].detach().cpu().numpy()
