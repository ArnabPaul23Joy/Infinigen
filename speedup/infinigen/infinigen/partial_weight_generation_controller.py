import torch
import torch.nn.functional as F


def partial_weight_index_generation(query, n_head, head_dim, partial_weight_ratio):
    """Generates the indices of partial weight query and partial key cache.

    On the prefill stage, generates the indices of partial weight query and
    partial key cache using the query matrix. By comparing the absolute sum of
    each column of the query matrix, gets the indices of top-k columns. These
    columns correspond to the columns that strongly affect the attention score.
    Thus, we use only those partial columns of query and key for speculation.

    Args:
        query: Query matrix (b, n, D)
        n_head: Number of heads which we refer to as h
        head_dim: Hidden dimension of each head which we refer to as d
        partial_weight_ratio: Ratio of the top-k columns

    Returns:
        partial_weight_index: Indices of top-k columns (b, h, d')
            where d' is d * (partial_weight_ratio).
    """

    partial_weight_index = torch.zeros(n_head, int(head_dim * partial_weight_ratio)).to(
        query.device
    )
    b = query.shape[0]

    for h_idx in range(n_head):
        start = h_idx * head_dim
        end = (h_idx + 1) * head_dim
        _, ind = torch.topk(
            torch.sum(torch.abs(query[0, :, start:end]), dim=-2),
            int(head_dim * partial_weight_ratio),
        )
        partial_weight_index[h_idx] = ind

    return partial_weight_index.unsqueeze(0).repeat(b, 1, 1).to(torch.int64)


def set_partial_cache(k_cache, partial_index, n_head, head_dim):
    """Sets the partial key cache.

    On the prefill and decoding stages, generates the partial key cache
    following the partial_index which indicates the indices of the important
    columns.

    Args:
        k_cahce: Key cache (n, bh, d)
        partial_weight_index: Indices of top-k columns (b, h, d')
        n_head: Number of heads which we refer to as h
        head_dim: Hidden dimension of each head which we refer to as d

    Returns:
        partial_cache: Partial key cache (n, bh, d')
    """

    n, bh, _ = k_cache.shape

    # torch.gather(input, dim, index) selects values from `input` along `dim`
    # using positions given by `index`. output[i,j,k,l] = input[i,j,k, index[i,j,k,l]]
    # All four tensors must have the same number of dimensions and the same
    # size on every axis *other* than the gather dimension.
    partial_cache = torch.gather(
        # input — reshape k_cache from (n, b*n_head, head_dim) to
        # (n, b, n_head, head_dim) so the per-head column axis (dim 3)
        # is explicit and addressable.
        k_cache.view(n, -1, n_head, head_dim),

        # dim=3 — gather along the head_dim axis, i.e. select a subset of
        # the d columns for each token, batch, and head independently.
        3,

        # index — partial_index has shape (b, n_head, d') where d' is the
        # number of top-k columns. unsqueeze(0) adds the token axis to give
        # (1, b, n_head, d'), then repeat(n, 1, 1, 1) broadcasts it to
        # (n, b, n_head, d') so every token position uses the same column
        # indices (the important columns are the same for all tokens).
        partial_index.unsqueeze(0).repeat(n, 1, 1, 1),
    )
    # Flatten (n, b, n_head, d') back to (n, b*n_head, d') to match the
    # layout the rest of the pipeline expects for the key cache.
    return partial_cache.view(n, bh, -1)


def set_partial_weight(w_q, partial_index, n_head, head_dim):
    """Sets the partial query weight.

    On the prefill stage, generates the partial query weight following the
    partial_index which indicates the indices of the important columns.

    Args:
        w_q: Query weight (D, D)
        partial_weight_index: Indices of top-k columns (b, h, d')
        n_head: Number of heads which we refer to as h
        head_dim: Hidden dimension of each head which we refer to as d

    Returns:
        partial_weight: Partial query weight (D', D)
    """

    partial_weight = F.embedding(
        partial_index[0]
        + torch.arange(n_head)[:, None].to(partial_index.device) * head_dim,
        w_q.view(-1, w_q.shape[-1]),
    )
    return partial_weight.view(-1, w_q.shape[-1])
