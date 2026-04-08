"""Relation encoder for item-item relations."""

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp


class RelationEncoder(nn.Module):
    """Encoder for item-item relations."""

    def __init__(self, item_embeddings, relation_matrices, device="cpu"):
        """Initialize relation encoder.

        Args:
            item_embeddings: Item embeddings (n_items, embed_dim)
            relation_matrices: Dict of relation_type -> sparse matrix
            device: Device to run on
        """
        super().__init__()

        self.n_items = item_embeddings.shape[0]
        self.embed_dim = item_embeddings.shape[1]
        self.device = device

        # Store item embeddings as buffer
        self.register_buffer('item_embeddings', torch.from_numpy(item_embeddings).float())

        # Convert relation matrices to sparse tensors
        self.relation_matrices = {}
        for rel_type, mat in relation_matrices.items():
            self.relation_matrices[rel_type] = self._sparse_to_tensor(mat)

    def _sparse_to_tensor(self, sparse_mat):
        """Convert sparse matrix to torch tensor."""
        indices = torch.from_numpy(np.vstack((sparse_mat.row, sparse_mat.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mat.data.astype(np.float32))
        size = torch.Size(sparse_mat.shape)

        return torch.sparse_coo_tensor(indices, values, size).to(self.device)

    def forward(self, item_indices=None):
        """Compute relation-enhanced embeddings.

        Args:
            item_indices: Optional indices to compute embeddings for

        Returns:
            Relation-enhanced embeddings
        """
        if item_indices is not None:
            base_emb = self.item_embeddings[item_indices]
        else:
            base_emb = self.item_embeddings

        # Aggregate from relations (mean aggregation)
        rel_emb = torch.zeros_like(base_emb)

        for rel_type, adj in self.relation_matrices.items():
            # Aggregate neighbor embeddings
            neighbor_emb = torch.sparse.mm(adj, self.item_embeddings)

            # Count neighbors
            degree = torch.sparse.sum(adj, dim=1).to_dense()
            degree = torch.clamp(degree, min=1)

            # Normalize
            neighbor_emb = neighbor_emb / degree.unsqueeze(1)

            rel_emb += neighbor_emb

        # Average over relation types
        if self.relation_matrices:
            rel_emb = rel_emb / len(self.relation_matrices)

        # Combine base and relation embeddings
        combined = base_emb + rel_emb

        return combined

    def get_related_items(self, item_idx, relation_type=None):
        """Get related items for a given item.

        Args:
            item_idx: Item index
            relation_type: Optional filter by relation type

        Returns:
            List of related item indices
        """
        results = []

        if relation_type and relation_type in self.relation_matrices:
            adj = self.relation_matrices[relation_type].to_dense()
            related = torch.nonzero(adj[item_idx]).squeeze().tolist()
            if isinstance(related, int):
                results.append(related)
            else:
                results.extend(related)
        else:
            # Get all related items
            for rel_type, adj in self.relation_matrices.items():
                adj_dense = adj.to_dense()
                related = torch.nonzero(adj_dense[item_idx]).squeeze().tolist()
                if isinstance(related, int):
                    results.append(related)
                else:
                    results.extend(related)

        return list(set(results))
