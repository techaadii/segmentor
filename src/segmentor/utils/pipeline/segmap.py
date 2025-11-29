import torch


def cosine_similarity_map(featmap: torch.Tensor, row: int, col: int):
    """
    Compute a cosine similarity map between a selected pixel's feature vector
    and all feature vectors in the feature map.

    This function extracts the feature vector at a given spatial coordinate
    `(row, col)` from a feature map of shape `(C, H, W)`. It then computes the
    cosine similarity between this reference vector and every other feature
    vector in the map. The output is a 2D similarity map of shape `(H, W)`.

    Args:
        featmap (torch.Tensor):
            A tensor of shape `(C, H, W)` representing the feature map.
        row (int):
            The row index of the pixel whose feature will be used as the
            reference vector.
        col (int):
            The column index of the pixel whose feature will be used as the
            reference vector.

    Returns:
        torch.Tensor:
            A tensor of shape `(H, W)` containing the cosine similarity values
            between the reference feature and each spatial location in the
            feature map.

    Raises:
        AssertionError:
            If `featmap` is not a 3D tensor of shape `(C, H, W)`.
    """

    assert featmap.dim() == 3, "featmap must have shape (C, H, W)"

    C, H, W = featmap.shape

    # Extract reference feature
    ref = featmap[:, row, col]  # (C,)
    ref = ref / (ref.norm() + 1e-8)  # normalize

    # Flatten spatial dims
    feats = featmap.view(C, -1)  # (C, H*W)
    feats_norm = feats / (feats.norm(dim=0, keepdim=True) + 1e-8)

    # Dot product yields cosine similarity
    sim = torch.matmul(ref, feats_norm)  # (H*W)

    return sim.view(H, W)


def probabilistic_segmentation_with_contrastive_scoring(
    featmap: torch.Tensor,
    pos_feats: torch.Tensor,
    neg_feats: torch.Tensor,
    gamma: float = 30.0,
    alpha: float = 1.0,
    eta: float = 1.0,
    bias: float = 0.0,
) -> torch.Tensor:
    """
    Compute a pixelwise probability map using positive/negative features
    with contrastive scoring.

    This function fuses dense DINOv3 pixel features with user-provided
    positive and negative exemplar features using a contrastive formulation:

    1. Each exemplar feature (positive or negative) is compared to every
       pixel feature via cosine similarity.

    2. Cosine similarities are scaled into "pseudo-logits" using
       a temperature parameter `gamma`. This expands the small numerical
       range of cosine similarity ([-1, 1]) into a space where exponential
       aggregation is more expressive.

    3. Positive similarities are aggregated using a log-sum-exp (soft-max)
       operation. This makes the strongest positive exemplar dominate while
       still considering all of them.

    4. Negative similarities are aggregated with log-sum-exp as well.
       Strong matches to negative exemplars penalize the score.

    5. The final pixel logit is the difference:
           L(x) = LSE_pos(x) - LSE_neg(x) - bias

    6. Passing this logit through a sigmoid produces a probability in [0, 1],
       where higher values indicate stronger alignment with positive clicks.

    Args:
        featmap (torch.Tensor):
            Dense pixelwise feature map of shape (C, H, W).
            Typically DINOv3 features upsampled using AnyUp.
        pos_feats (torch.Tensor):
            Positive exemplar features of shape (n_pos, C).
            Each row corresponds to a feature vector extracted from a
            user positive click.
        neg_feats (torch.Tensor):
            Negative exemplar features of shape (n_neg, C).
            Each row corresponds to features extracted at negative clicks.
        gamma (float):
            Temperature scaling applied to cosine similarities to turn them
            into expressive pseudo-logits. Higher gamma increases separation.
        alpha (float):
            Additional scaling inside the exponential for log-sum-exp.
            Usually left at 1.0.
        eta (float):
            Exponent applied to pixel feature norms. When > 0, pixels with
            stronger feature norms receive slightly higher influence.
        bias (float):
            Optional bias subtracted from the final logits. Can shift the
            operating point of the sigmoid output.

    Returns:
        torch.Tensor:
            Probability map of shape (H, W), with values in [0, 1].
            Higher values indicate regions matching positive exemplars
            and not matching negative exemplars.

    Example:
        >>> pos_feats = featmap[:, [r1, r2], [c1, c2]].T
        >>> neg_feats = featmap[:, [r3], [c3]].T
        >>> prob = clicks_to_prob_map(featmap, pos_feats, neg_feats)
    """
    C, H, W = featmap.shape
    device = featmap.device

    # ----------------------------------------------------------------------
    # 1. Reshape dense feature map for batched matrix multiplication.
    #    featmap: (C, H, W) → feats: (C, H*W)
    # ----------------------------------------------------------------------
    feats = featmap.reshape(C, -1)  # (C, N), N = H*W
    N = feats.shape[1]

    # Pixelwise feature norms, used to normalize cosine and optionally weigh pixels.
    key_norm = feats.norm(dim=0, keepdim=True) + 1e-8  # (1, N)

    # ----------------------------------------------------------------------
    # 2. Normalize positive and negative exemplar features.
    #    If no exemplars exist, return empty tensors with correct shapes.
    # ----------------------------------------------------------------------
    def normalize(v: torch.Tensor) -> torch.Tensor:
        return v / (v.norm(dim=1, keepdim=True) + 1e-8)

    if pos_feats.numel() > 0:
        pos_q = normalize(pos_feats)
    else:
        pos_q = torch.empty((0, C), device=device)

    if neg_feats.numel() > 0:
        neg_q = normalize(neg_feats)
    else:
        neg_q = torch.empty((0, C), device=device)

    # Normalize pixel features for cosine computation.
    keys_unit = feats / key_norm  # (C, N)

    # ----------------------------------------------------------------------
    # 3. Compute cosine similarities to all pixels.
    #    cos_pos: (n_pos, N)
    #    cos_neg: (n_neg, N)
    # ----------------------------------------------------------------------
    if pos_q.shape[0] > 0:
        cos_pos = pos_q @ keys_unit
    else:
        cos_pos = torch.empty((0, N), device=device)

    if neg_q.shape[0] > 0:
        cos_neg = neg_q @ keys_unit
    else:
        cos_neg = torch.empty((0, N), device=device)

    # ----------------------------------------------------------------------
    # 4. Optionally weigh similarities by pixel feature norm (quality weighting).
    # ----------------------------------------------------------------------
    if eta != 0:
        weight = key_norm.squeeze(0) ** eta  # (N,)
        if pos_q.shape[0] > 0:
            cos_pos = cos_pos * weight.unsqueeze(0)
        if neg_q.shape[0] > 0:
            cos_neg = cos_neg * weight.unsqueeze(0)

    # ----------------------------------------------------------------------
    # 5. Convert cosine similarities into pseudo-logits via temperature scaling.
    # ----------------------------------------------------------------------
    logit_pos = gamma * cos_pos  # (n_pos, N)
    logit_neg = gamma * cos_neg  # (n_neg, N)

    # ----------------------------------------------------------------------
    # 6. Aggregate positive and negative evidence using log-sum-exp.
    #    - logsumexp behaves like a smooth maximum.
    #    - If no exemplars exist, return -inf so they contribute nothing.
    # ----------------------------------------------------------------------
    if logit_pos.shape[0] > 0:
        A_pos = torch.logsumexp(alpha * logit_pos, dim=0)  # (N,)
    else:
        A_pos = torch.full((N,), float("-inf"), device=device)

    if logit_neg.shape[0] > 0:
        A_neg = torch.logsumexp(alpha * logit_neg, dim=0)  # (N,)
    else:
        A_neg = torch.full((N,), float("-inf"), device=device)

    # ----------------------------------------------------------------------
    # 7. Final logit per pixel: positive evidence minus negative evidence.
    #    Bias term allows shifting decision threshold.
    # ----------------------------------------------------------------------
    logits = A_pos - A_neg - bias  # (N,)

    # Special-case behavior when only pos or neg exemplars exist
    if pos_q.shape[0] == 0:
        logits = -A_neg - bias
    if neg_q.shape[0] == 0:
        logits = A_pos - bias

    # ----------------------------------------------------------------------
    # 8. Sigmoid converts logits → probabilities in [0, 1].
    # ----------------------------------------------------------------------
    prob = torch.sigmoid(logits).reshape(H, W)

    return prob
