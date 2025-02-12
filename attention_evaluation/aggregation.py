import torch

def aggregate_attention(attention_scores: torch.Tensor, weights: torch.Tensor=None) -> torch.Tensor:
    """
    Aggregate attention scores from all layers and heads into a single (N, N) matrix.
    This implementation first averages over heads per layer, adds an identity matrix 
    (to account for residual connections), normalizes rows, and then multiplies 
    the matrices across layers to capture the recursive flow of attention.
    
    Args:
        attention_scores (torch.Tensor): Tensor of shape (batch, layers, heads, N, N).
                                                      or (layers, heads, N, N)
    
    Returns:
        torch.Tensor: Aggregated attention matrix of shape (batch, N, N).
    """
    

    batch_size, N_layers, N_heads, N, _ = attention_scores.shape
    # Create uniform weights if not provided.
    if weights is None:
        weights = torch.ones(N_layers, N_heads, device=attention_scores.device, dtype=attention_scores.dtype) / N_heads
    # Expand weights to shape (1, N_layers, N_heads, 1, 1) for broadcasting.
    weights_exp = weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    # Weighted average over heads: result shape (batch_size, N_layers, N, N)
    layers_weighted_avg = (attention_scores * weights_exp).sum(dim=2)
    # Create identity matrix and expand to batch: shape (batch_size, N, N)
    identity = torch.eye(N, device=attention_scores.device, dtype=attention_scores.dtype).unsqueeze(0).expand(batch_size, N, N)
    # Initialize rollout as None.
    rollout = None
    # Multiply layers recursively.
    for layer in range(N_layers):
        # For each layer, add identity and normalize rows.
        A = layers_weighted_avg[:, layer, :, :] + identity
        A = A / A.sum(dim=-1, keepdim=True)
        if rollout is None:
            rollout = A
        else:
            rollout = torch.bmm(A, rollout)
    return rollout

def evaluate_attention(attention_scores: torch.Tensor, attention_labels: torch.Tensor) -> float:
    """
    Evaluate the aggregated attention matrix against labels.
    For each row that is not entirely marked as -100 (i.e. skip rows),
    compute the fraction: (sum of scores for positions labeled as 1) / 
    (sum of scores for positions with labels in {0, 1}).
    Then average these fractions over all such rows.
    
    Args:
        attention_scores (torch.Tensor): Aggregated attention matrix (N, N) or (Batch, N, N).
        attention_labels (torch.Tensor): Labels matrix (N, N) or (Batch, N, N) with:
            - 1 for significant tokens,
            - 0 for insignificant tokens, and
            - -100 for rows to skip.
    
    Returns:
        float: Average percentage of correctly attended tokens per valid row (0 to 1).
    """

    if attention_scores.shape != attention_labels.shape:
        raise ValueError("Attention scores and attention labels must have the same shape.")

    # Add the batch axis, if missing
    if len(attention_scores.shape) == 2:
        attention_scores = attention_scores.unsqueeze(0)
        attention_labels = attention_labels.unsqueeze(0) 

    percentages = []
    for batch_idx in range(attention_scores.size(0)):
        for sample_idx in range(attention_scores.size(1)):
            # If the entire row is marked to be skipped, continue.
            if torch.all(attention_labels[batch_idx][sample_idx] == -100):
                continue

            mask = attention_labels[batch_idx][sample_idx]
            
            # Sum of attention scores for positions labeled as significant (1).
            numerator = attention_scores[batch_idx][sample_idx][attention_labels[batch_idx][sample_idx] == 1].sum().item()
            # Sum of attention scores for all valid positions.
            denominator = attention_scores[batch_idx][sample_idx][mask].sum().item()
            
            if denominator > 0:
                percentages.append(numerator / denominator)
    
    if percentages:
        return sum(percentages) / len(percentages)
    else:
        return 0.0