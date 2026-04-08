"""Graph construction utilities."""

import numpy as np
import scipy.sparse as sp
import torch


def build_user_item_graph(interactions, user2idx, item2idx, n_users, n_items):
    """Build user-item interaction graph.

    Args:
        interactions: DataFrame with user_id, item_id
        user2idx: User ID mapping
        item2idx: Item ID mapping
        n_users: Number of users
        n_items: Number of items

    Returns:
        scipy sparse adjacency matrix
    """
    rows = []
    cols = []
    data = []

    for _, row in interactions.iterrows():
        u_idx = user2idx[row['user_id']]
        i_idx = item2idx[row['item_id']] + n_users  # Items are offset

        rows.append(u_idx)
        cols.append(i_idx)
        data.append(1)

        # Undirected graph
        rows.append(i_idx)
        cols.append(u_idx)
        data.append(1)

    # Create sparse matrix
    adj_matrix = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_users + n_items, n_users + n_items)
    )

    return adj_matrix.tocsr()


def build_item_item_graph(relations, item2idx, n_items):
    """Build item-item relation graph.

    Args:
        relations: DataFrame with item_id, related_item_id, relation_type
        item2idx: Item ID mapping
        n_items: Number of items

    Returns:
        Dictionary with relation type -> sparse matrix
    """
    relation_matrices = {}

    for rel_type in relations['relation_type'].unique():
        rel_df = relations[relations['relation_type'] == rel_type]

        rows = []
        cols = []
        data = []

        for _, row in rel_df.iterrows():
            if row['item_id'] in item2idx and row['related_item_id'] in item2idx:
                i_idx = item2idx[row['item_id']]
                j_idx = item2idx[row['related_item_id']]

                rows.append(i_idx)
                cols.append(j_idx)
                data.append(1)

                # Symmetric
                rows.append(j_idx)
                cols.append(i_idx)
                data.append(1)

        if rows:
            relation_matrices[rel_type] = sp.coo_matrix(
                (data, (rows, cols)),
                shape=(n_items, n_items)
            ).tocsr()

    return relation_matrices


def normalize_adjacency(adj):
    """Normalize adjacency matrix.

    Args:
        adj: Sparse adjacency matrix

    Returns:
        Normalized sparse adjacency matrix
    """
    # Add self-loops
    adj = adj + sp.eye(adj.shape[0])

    # Degree matrix
    d = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # Normalize: D^(-1/2) * A * D^(-1/2)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def sparse_to_tensor(sparse_mat):
    """Convert sparse matrix to torch tensor."""
    # Convert to COO format if needed
    if not hasattr(sparse_mat, 'row'):
        sparse_mat = sparse_mat.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mat.row, sparse_mat.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mat.data.astype(np.float32))
    size = torch.Size(sparse_mat.shape)

    return torch.sparse_coo_tensor(indices, values, size)
