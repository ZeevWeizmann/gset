import anndata as ad

adata = ad.read_h5ad("/Users/zeev/CardamomOT/my_project/Data/data.h5ad")

print(adata)



import scanpy as sc

import numpy as np

cell_sums = np.array(adata.X.sum(axis=1)).flatten()

adata = adata[cell_sums > 0].copy()

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

sc.pp.pca(adata)

sc.pp.neighbors(adata, n_neighbors=10, n_pcs=10)

sc.tl.umap(adata)

sc.pl.umap(adata, color=["cell_type"])

A = adata.obsp["connectivities"]

print(A.shape)        
print(A.nnz) 

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# берём 200 клеток
idx = np.random.choice(adata.n_obs, 200, replace=False)

A_sub = A[idx][:, idx]

G = nx.from_scipy_sparse_array(A_sub)

plt.figure(figsize=(6,6))
nx.draw(G, node_size=10)
plt.show()