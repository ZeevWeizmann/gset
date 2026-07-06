import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_selection import mutual_info_regression

# ======================
# 1. LOAD DATA
# ======================
adata = ad.read_h5ad("/Users/zeev/CardamomOT/my_project/Data/data.h5ad")

mask = np.array(adata.X.sum(axis=1)).flatten() > 0
adata = adata[mask].copy()

X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
X = np.asarray(X, dtype=np.float32)

times = adata.obs["time"].values
unique_times = np.sort(np.unique(times))

gene_names = list(adata.var_names)
gene_to_id = {g: i for i, g in enumerate(gene_names)}
n_genes = X.shape[1]

print("Cells x genes:", X.shape)
print("Genes:", gene_names)

# ======================
# 2. BUILD TIME DATA
# ======================
X_t, X_t1 = [], []

for i in range(len(unique_times) - 1):
    t0, t1 = unique_times[i], unique_times[i + 1]

    cells_t0 = X[times == t0]
    cells_t1 = X[times == t1]

    if len(cells_t0) == 0 or len(cells_t1) == 0:
        continue

    X_t.append(cells_t0.mean(axis=0))
    X_t1.append(cells_t1.mean(axis=0))

X_t = torch.tensor(np.array(X_t), dtype=torch.float32)
X_t1 = torch.tensor(np.array(X_t1), dtype=torch.float32)

print("X_t:", X_t.shape)
print("X_t1:", X_t1.shape)

# ======================
# 3. BIOLOGICAL PRIOR
# ======================
prior_edges = []

try:
    from omnipath import interactions

    print("\nTrying DoRothEA biological prior...")
    net = interactions.Dorothea.get()
    print("DoRothEA columns:", list(net.columns))

    source_col = next(
        (c for c in ["source_genesymbol", "source", "tf", "TF"] if c in net.columns),
        None
    )
    target_col = next(
        (c for c in ["target_genesymbol", "target", "target_gene"] if c in net.columns),
        None
    )

    if source_col is None or target_col is None:
        raise ValueError("Cannot find source/target columns in DoRothEA.")

    for _, row in net.iterrows():
        src = row[source_col]
        tgt = row[target_col]

        if src in gene_to_id and tgt in gene_to_id and src != tgt:
            prior_edges.append((gene_to_id[src], gene_to_id[tgt]))

    prior_edges = list(set(prior_edges))
    print("Biological prior edges found:", len(prior_edges))
    print("Prior edges:", [(gene_names[s], gene_names[t]) for s, t in prior_edges])

except Exception as e:
    print("Biological prior failed:", e)

# ======================
# 4. PLOT PRIOR ONLY
# ======================
G_prior = nx.DiGraph()

for i, g in enumerate(gene_names):
    G_prior.add_node(i, label=g)

for src, dst in prior_edges:
    G_prior.add_edge(src, dst)

plt.figure(figsize=(7, 6))
pos = nx.spring_layout(G_prior, seed=42)

nx.draw_networkx_nodes(G_prior, pos, node_size=900)
nx.draw_networkx_labels(
    G_prior,
    pos,
    labels={i: gene_names[i] for i in range(n_genes)},
    font_size=8
)
nx.draw_networkx_edges(
    G_prior,
    pos,
    arrows=True,
    arrowsize=15,
    alpha=0.7
)

plt.title(f"Biological PRIOR graph — edges: {len(prior_edges)}")
plt.axis("off")
plt.tight_layout()
plt.show()

# ======================
# 5. FALLBACK: DATA-DRIVEN MI GRAPH
# ======================
mi_edges = []

if len(prior_edges) == 0:
    print("\nNo biological prior edges found.")
    print("Using MI fallback graph...")

    mi_matrix = np.zeros((n_genes, n_genes))

    for target in range(n_genes):
        mi = mutual_info_regression(X, X[:, target], random_state=42)
        mi_matrix[target] = mi

    np.fill_diagonal(mi_matrix, 0)

    k = 3  # top-k incoming regulators per target

    for target in range(n_genes):
        regulators = np.argsort(mi_matrix[target])[::-1]

        selected = []
        for source in regulators:
            if source != target and mi_matrix[target, source] > 0:
                selected.append(source)

            if len(selected) == k:
                break

        for source in selected:
            mi_edges.append((source, target))

    mi_edges = list(set(mi_edges))
    print("MI fallback edges:", len(mi_edges))
    print("MI edges:", [(gene_names[s], gene_names[t]) for s, t in mi_edges])

# ======================
# 6. FINAL EDGES
# ======================
if len(prior_edges) > 0:
    edges = prior_edges
    graph_type = "BIOLOGICAL PRIOR"
else:
    edges = mi_edges
    graph_type = "MI FALLBACK"

print("\nFinal graph type:", graph_type)
print("Final edges:", len(edges))

if len(edges) == 0:
    raise ValueError("No edges found. Increase genes or lower filtering.")

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# ======================
# 7. MODEL
# ======================
class GeneDynamics(nn.Module):
    def __init__(self, n_genes, edge_index):
        super().__init__()

        self.edge_index = edge_index

        self.edge_weight = nn.Parameter(torch.randn(edge_index.shape[1]) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_genes))
        self.raw_decay = nn.Parameter(torch.zeros(n_genes))

    def forward(self, x, dt=1.0):
        src, dst = self.edge_index
        preds = []

        for sample in x:
            messages = self.edge_weight * sample[src]

            agg = torch.zeros_like(sample)
            agg.index_add_(0, dst, messages)

            production = torch.sigmoid(agg + self.bias)
            decay = torch.nn.functional.softplus(self.raw_decay) * sample

            dGdt = production - decay
            next_state = sample + dt * dGdt

            preds.append(next_state)

        return torch.stack(preds)

# ======================
# 8. TRAIN
# ======================
model = GeneDynamics(n_genes, edge_index)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

losses = []

for epoch in range(1, 301):
    optimizer.zero_grad()

    pred = model(X_t)
    mse = loss_fn(pred, X_t1)

    l1 = torch.abs(model.edge_weight).mean()
    loss = mse + 0.01 * l1

    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, MSE: {mse.item():.4f}")

# ======================
# 9. LOSS PLOT
# ======================
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss")
plt.tight_layout()
plt.show()

# ======================
# 10. TRUE VS PREDICTED
# ======================
model.eval()
with torch.no_grad():
    pred = model(X_t).numpy()

plt.figure(figsize=(5, 5))
plt.scatter(X_t1.numpy().flatten(), pred.flatten(), s=25)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("True vs Predicted")
plt.tight_layout()
plt.show()

# ======================
# 11. FINAL LEARNED GRAPH
# ======================
weights = model.edge_weight.detach().numpy()

G = nx.DiGraph()

for i, g in enumerate(gene_names):
    G.add_node(i, label=g)

for e, (src, dst) in enumerate(edges):
    G.add_edge(src, dst, weight=weights[e])

pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(7, 6))
nx.draw_networkx_nodes(G, pos, node_size=900)
nx.draw_networkx_labels(
    G,
    pos,
    labels={i: gene_names[i] for i in range(n_genes)},
    font_size=8
)

edge_widths = [abs(G[u][v]["weight"]) * 15 + 0.5 for u, v in G.edges()]
nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    arrowsize=15,
    width=edge_widths,
    alpha=0.7
)

plt.title(f"Learned gene graph ({graph_type})")
plt.axis("off")
plt.tight_layout()
plt.show()

# ======================
# 12. HEATMAP
# ======================
W = np.zeros((n_genes, n_genes))

for e, (src, dst) in enumerate(edges):
    W[src, dst] = weights[e]

plt.figure(figsize=(7, 6))
plt.imshow(W, aspect="auto")
plt.colorbar(label="learned regulatory strength")
plt.xlabel("Target gene")
plt.ylabel("Source gene")
plt.title(f"Learned gene → gene weights ({graph_type})")
plt.xticks(range(n_genes), gene_names, rotation=90, fontsize=8)
plt.yticks(range(n_genes), gene_names, fontsize=8)
plt.tight_layout()
plt.show()