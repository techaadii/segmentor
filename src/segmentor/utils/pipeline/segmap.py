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


# def knn_segmentation(
#     featmap: torch.Tensor, pos_feats: torch.Tensor, neg_feats: torch.Tensor
# ) -> torch.Tensor:


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


def mahalanobis_segmentation(
    featmap: torch.Tensor,
    pos_feats: torch.Tensor,
    neg_feats: torch.Tensor,
    reg_lambda: float = 0.01,
) -> torch.Tensor:
    """
    Compute a pixelwise probability map using Mahalanobis distance to
    positive and negative feature distributions.

    Unlike dot-product (cosine) methods which assume hyperspherical clusters,
    this function models the provided exemplars as Multivariate Gaussian
    distributions. This allows the segmentation to adapt to the specific
    variance and correlation structure of the requested object's features.

    The probability is computed via a generative approach:
        1. Fit Gaussian (Mean, Covariance) to positive examples.
        2. Fit Gaussian (Mean, Covariance) to negative examples.
        3. Compute Mahalanobis distance from every pixel to both distributions.
        4. Convert distances to logits: Logit = Dist_Neg - Dist_Pos.
        5. Sigmoid(Logit) -> Probability.

    Args:
        featmap (torch.Tensor):
            Dense pixelwise feature map of shape (C, H, W).
        pos_feats (torch.Tensor):
            Positive exemplar features of shape (n_pos, C).
        neg_feats (torch.Tensor):
            Negative exemplar features of shape (n_neg, C).
        reg_lambda (float):
            Regularization term added to the diagonal of the covariance matrix
            (Sigma + lambda*I). This prevents singular matrices when n_samples
            is small or features are collinear, ensuring invertibility.

    Returns:
        torch.Tensor:
            Probability map of shape (H, W), with values in [0, 1].

    Note:
        If n_pos < 2 or n_neg < 2, the function falls back to Euclidean
        distance (equivalent to Identity covariance) for that specific class,
        as variance cannot be reliably estimated from a single point.
    """
    C, H, W = featmap.shape
    device = featmap.device

    # ----------------------------------------------------------------------
    # 1. Reshape featmap to (N, C) for distance calculation.
    #    N = H * W
    # ----------------------------------------------------------------------
    # Permute to (H, W, C) then reshape to (N, C)
    pixels = featmap.permute(1, 2, 0).reshape(-1, C)
    N = pixels.shape[0]

    def get_mahalanobis_sq_dist(
        distribution_feats: torch.Tensor, query_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates squared Mahalanobis distance: (x-u)^T Sigma^-1 (x-u)
        """
        n_samples, n_dim = distribution_feats.shape

        # Case: No exemplars provided
        if n_samples == 0:
            return torch.full((query_feats.shape[0],), float("inf"), device=device)

        # 1. Compute Mean
        mu = distribution_feats.mean(dim=0, keepdim=True)  # (1, C)

        # 2. Compute Inverse Covariance (Precision Matrix)
        # If we have too few samples to estimate covariance, use Identity (Euclidean)
        if n_samples < 2:
            inv_cov = torch.eye(n_dim, device=device)
        else:
            # Center the features
            centered = distribution_feats - mu
            # Calculate covariance: (X^T @ X) / (n-1)
            # Shape: (C, C)
            cov = (centered.T @ centered) / (n_samples - 1)

            # Regularize to ensure invertibility
            cov = cov + torch.eye(n_dim, device=device) * reg_lambda

            # Invert
            try:
                inv_cov = torch.inverse(cov)
            except RuntimeError:
                # Fallback if Cholesky/Inverse still fails despite regularization
                inv_cov = torch.eye(n_dim, device=device)

        # 3. Compute Distance for all pixels
        # Delta: (N, C)
        delta = query_feats - mu

        # Mahalanobis distance formulation: sum((delta @ inv_cov) * delta, dim=1)
        # Optimized term: (N, C) @ (C, C) -> (N, C)
        left_term = delta @ inv_cov

        # Element-wise mult followed by sum corresponds to diag(A @ B^T)
        dists = (left_term * delta).sum(dim=1)  # (N,)

        return dists

    # ----------------------------------------------------------------------
    # 2. Calculate Distances to Positive and Negative Distributions
    # ----------------------------------------------------------------------
    # Distance to Positive Distribution (Lower is better)
    d2_pos = get_mahalanobis_sq_dist(pos_feats, pixels)

    # Distance to Negative Distribution (Lower is bad for being positive)
    d2_neg = get_mahalanobis_sq_dist(neg_feats, pixels)

    # ----------------------------------------------------------------------
    # 3. Convert Distances to Probabilities
    #
    # Logic:
    # If a pixel is closer to Positive than Negative, Logit should be > 0.
    # Logit = Distance_Negative - Distance_Positive
    #
    # Note on Bias:
    # Mahalanobis distance is effectively a negative log-likelihood (ignoring
    # constants). The diff of squared distances corresponds to the log-odds ratio
    # assuming equal priors and equal determinants of covariance.
    # ----------------------------------------------------------------------

    # Handle cases where one set of clicks is missing
    if pos_feats.shape[0] == 0 and neg_feats.shape[0] > 0:
        # Only negatives exist: strongly penalize everything close to neg
        logits = -d2_neg
    elif neg_feats.shape[0] == 0 and pos_feats.shape[0] > 0:
        # Only positives exist: value things close to pos
        # We invert the distance essentially. We need a threshold.
        # Simple heuristic: Logits = -d2_pos + Constant (centering handled by sigmoid)
        logits = -d2_pos + d2_pos.mean()
    else:
        # Both exist
        logits = d2_neg - d2_pos

    # Apply Sigmoid to get [0, 1] probability
    prob = torch.sigmoid(logits).reshape(H, W)

    return prob
