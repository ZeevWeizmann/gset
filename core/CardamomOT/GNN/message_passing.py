import anndata as ad
import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE

# ======================
# 1. LOAD DATA
# ======================
adata = ad.read_h5ad("/Users/zeev/CardamomOT/my_project/Data/data.h5ad")

print(adata)

# ======================
# 2. CLEAN DATA
# ======================
# убрать клетки с нулевой экспрессией
cell_sums = np.array(adata.X.sum(axis=1)).flatten()
adata = adata[cell_sums > 0].copy()

print("After filtering:", adata.shape)

# ======================
# 3. PCA + kNN GRAPH
# ======================
# ⚠️ НЕ делаем normalize/log1p (уже есть)

sc.pp.pca(adata, n_comps=5)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=5)

A = adata.obsp["connectivities"]

# ======================
# 4. PyTorch Geometric DATA
# ======================
# features
x = torch.tensor(adata.X.toarray(), dtype=torch.float)

# edges
edge_index = torch.tensor(A.nonzero(), dtype=torch.long)

data = Data(x=x, edge_index=edge_index)

print(data)

# ======================
# 5. GNN ENCODER
# ======================
class Encoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# ======================
# 6. MODEL (Graph Autoencoder)
# ======================
model = GAE(Encoder(in_dim=x.shape[1], hidden_dim=16))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ======================
# 7. TRAINING
# ======================
model.train()

for epoch in range(1, 101):
    optimizer.zero_grad()
    
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ======================
# 8. GET EMBEDDINGS
# ======================
model.eval()
z = model.encode(data.x, data.edge_index).detach().numpy()

adata.obsm["X_gnn"] = z

print("Embeddings shape:", z.shape)

# ======================
# 9. UMAP on GNN embeddings
# ======================
sc.pp.neighbors(adata, use_rep="X_gnn")
sc.tl.umap(adata)

sc.pl.umap(adata, color=["cell_type"])
sc.pl.umap(adata, color=["time"])