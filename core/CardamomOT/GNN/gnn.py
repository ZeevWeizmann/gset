import anndata as ad
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ======================
# 1. LOAD DATA
# ======================
adata = ad.read_h5ad("/Users/zeev/CardamomOT/my_project/Data/data_uncut.h5ad")

X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
X = np.asarray(X, dtype=np.float32)

print("Cells x genes:", X.shape)

# ======================
# 2. PREPROCESSING
# ======================
X = np.log1p(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================
# 3. PCA
# ======================
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_scaled)

print("PCA shape:", X_pca.shape)

# ======================
# 4. kNN GRAPH
# ======================
k = 12  # лучше 10–15

nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
nbrs.fit(X_pca)

distances, indices = nbrs.kneighbors(X_pca)

# ======================
# 5. UNDIRECTED GRAPH
# ======================
G = nx.Graph()
n_cells = X_pca.shape[0]

for i in range(n_cells):
    G.add_node(i)

for i in range(n_cells):
    for j_idx, dist in zip(indices[i][1:], distances[i][1:]):
        weight = np.exp(-dist)
        G.add_edge(i, j_idx, weight=weight)

print("Undirected graph:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# ======================
# 6. DIRECTED GRAPH (FLOW)
# ======================
times = adata.obs["time"].values

G_dir = nx.DiGraph()

for i in range(n_cells):
    for j_idx in indices[i][1:]:
        # строго вперёд по времени
        if times[j_idx] > times[i]:
            G_dir.add_edge(i, j_idx)

print("\nDirected graph:")
print("Nodes:", G_dir.number_of_nodes())
print("Edges:", G_dir.number_of_edges())

# ======================
# 7. FLOW SCORE
# ======================
flow_score = {}

for node in G_dir.nodes:
    flow_score[node] = G_dir.in_degree(node) - G_dir.out_degree(node)

scores = np.array([flow_score.get(i, 0) for i in range(n_cells)])

# ======================
# 8. ATTRACTOR NODES
# ======================
# top nodes с максимальным flow
threshold = np.percentile(scores, 95)

attractors = np.where(scores >= threshold)[0]

print("\nAttractor candidates:", len(attractors))

# ======================
# 9. VISUALIZATION: GRAPH
# ======================
plt.figure(figsize=(7,6))

pos = {i: X_pca[i, :2] for i in range(n_cells)}

nx.draw_networkx_edges(G, pos, alpha=0.05)
nx.draw_networkx_nodes(G, pos, node_size=5)

plt.title("Cell graph (kNN)")
plt.axis("off")
plt.show()

# ======================
# 10. VISUALIZATION: FLOW
# ======================
plt.figure(figsize=(7,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=scores,
    cmap='coolwarm',
    s=5
)

plt.colorbar(label="flow score")
plt.title("Flow / attractor score")
plt.show()

# ======================
# 11. VISUALIZATION: ATTRACTORS
# ======================
plt.figure(figsize=(7,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c="lightgray",
    s=5
)

plt.scatter(
    X_pca[attractors,0],
    X_pca[attractors,1],
    c="red",
    s=10,
    label="Attractors"
)

plt.legend()
plt.title("Detected attractor regions")
plt.show()

# ======================
# 12. BASIN SEGMENTATION (simple)
# ======================
# каждый узел "идёт" по направлению пока не упрётся

def find_sink(node, G_dir):
    current = node
    visited = set()

    while True:
        neighbors = list(G_dir.successors(current))

        if len(neighbors) == 0:
            return current

        # идём к самому "позднему" по времени
        next_node = max(neighbors, key=lambda x: times[x])

        if next_node in visited:
            return current

        visited.add(current)
        current = next_node


basin_map = {}

for i in range(n_cells):
    basin_map[i] = find_sink(i, G_dir)

# уникальные basin'ы
unique_basins = list(set(basin_map.values()))
print("\nNumber of basins:", len(unique_basins))

# ======================
# 13. VISUALIZE BASINS
# ======================
colors = {b: idx for idx, b in enumerate(unique_basins)}
basin_colors = [colors[basin_map[i]] for i in range(n_cells)]

plt.figure(figsize=(7,6))
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=basin_colors,
    cmap="tab20",
    s=5
)

plt.title("Basins (discrete attractor regions)")
plt.show()