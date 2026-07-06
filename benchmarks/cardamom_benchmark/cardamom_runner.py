"""
cardamom_runner.py
------------------
Run CardamomOT inference on a prepared project directory.

Wraps the CardamomOT pipeline (infer_mixture → infer_network_structure)
and returns the inferred interaction matrix.

The HARISSA kinetic rates (d0, d1) are embedded in the h5ad by
prepare_harissa.py, so get_kinetic_rates.py is skipped.

Usage
-----
  from cardamom_runner import run_cardamom
  inter, basal = run_cardamom(project_dir, cardamom_repo_dir)
"""

import os
import subprocess
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


CARDAMOM_REPO = Path(__file__).resolve().parents[2] / "core"


def run_cardamom(
    project_dir: str,
    split: str = "train",
    cardamom_repo: Optional[str] = None,
    timeout: int = 1800,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run the core CardamomOT inference pipeline on a prepared project directory.

    Steps executed:
      1. infer_mixture.py      — estimate burst kinetics
      2. infer_network_structure.py — infer GRN interactions

    Parameters
    ----------
    project_dir   : path to CardamomOT project (must contain Data/data_train.h5ad)
    split         : data split label (default: 'train')
    cardamom_repo : path to CardamomOT repository (default: /Users/zeev/CardamomOT)
    timeout       : max seconds per script

    Returns
    -------
    inter : (G+1, G+1) inferred interaction matrix, or None if failed
    basal : (G+1,) inferred basal expression, or None if failed
    """
    project_dir   = Path(project_dir)
    cardamom_repo = Path(cardamom_repo or CARDAMOM_REPO)

    cardamot_dir = project_dir / "cardamomOT"
    cardamot_dir.mkdir(parents=True, exist_ok=True)

    def run_script(script_name, extra_args=None):
        script = cardamom_repo / script_name
        cmd = [sys.executable, str(script), "-i", str(project_dir.resolve()), "-s", split]
        if extra_args:
            cmd.extend(extra_args)
        print(f"[cardamom_runner] Running {script_name} ...")
        import tempfile
        with tempfile.TemporaryFile() as tmp_out, tempfile.TemporaryFile() as tmp_err:
            result = subprocess.run(
                cmd, stdout=tmp_out, stderr=tmp_err,
                timeout=timeout, cwd=str(cardamom_repo)
            )
            tmp_err.seek(0)
            stderr_text = tmp_err.read().decode(errors="replace")
        if result.returncode != 0:
            print(f"[cardamom_runner] STDERR from {script_name}:")
            print(stderr_text[-2000:])
            raise RuntimeError(f"{script_name} failed (returncode={result.returncode})")
        print(f"[cardamom_runner] {script_name} OK")
        return result

    # Step 1: mixture inference
    try:
        run_script("infer_mixture.py")
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"[cardamom_runner] FAILED at infer_mixture: {e}")
        return None, None

    # Verify mixture output exists before proceeding
    mixture_files = ["modes.npy", "proba.npy", "mixture_parameters.npy"]
    missing = [f for f in mixture_files if not (cardamot_dir / f).exists()]
    if missing:
        print(f"[cardamom_runner] infer_mixture.py finished but missing: {missing}")
        return None, None
    print(f"[cardamom_runner] ✓ Mixture done: {[f for f in mixture_files]}")

    # Step 2: network structure inference
    try:
        run_script("infer_network_structure.py")
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"[cardamom_runner] FAILED at infer_network_structure: {e}")
        return None, None

    # Verify network output exists
    inter_path = cardamot_dir / "inter.npy"
    basal_path = cardamot_dir / "basal.npy"
    if not inter_path.exists():
        print(f"[cardamom_runner] infer_network_structure.py finished but inter.npy missing")
        return None, None
    print(f"[cardamom_runner] ✓ Network done: inter.npy, basal.npy")

    inter = np.load(inter_path)
    basal = np.load(basal_path) if basal_path.exists() else None
    if inter.ndim == 3:
        inter = inter[:, :, 0]
    if basal is not None and basal.ndim == 2:
        basal = basal[:, 0]
    print(f"[cardamom_runner] ✓ Loaded inter {inter.shape}, basal {basal.shape if basal is not None else None}")
    return inter, basal
