import torch
import math
from scipy.stats import t
from typing import List
import numpy as np
import matplotlib.pyplot as plt


def aggregate_attention(attention_scores: torch.Tensor, weights: torch.Tensor=None) -> torch.Tensor:
    """
    Aggregate attention scores from all layers and heads into a single (N, N) matrix.
    This implementation first averages over heads per layer, adds an identity matrix 
    (to account for residual connections), normalizes rows, and then multiplies 
    the matrices across layers to capture the recursive flow of attention.
    
    Args:
        attention_scores (torch.Tensor): Tensor of shape (batch, layers, heads, N, N) or (layers, heads, N, N)
        weights (torch.Tensor): Optionally, you can specify weights that will be used instead of 
                                uniform averaging of heads in each layer.
                                Tensor of shape (batch, layers, heads, N, N) or (layers, heads, N, N)
    
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


def compute_percentages(attention: torch.Tensor, attention_labels: torch.Tensor) -> list:
    """
    Compute attention percentages for each row of an aggregated attention matrix.
    For each row (ignoring rows with all labels as -100), the percentage is defined as:
        (sum of scores for positions labeled as 1) / (sum of scores for positions with labels in {0, 1})
        Take the absolute value of scores in case of negative scores
    
    Args:
        attention (torch.Tensor): Aggregated attention matrix (Batch, N, N) or (N, N).
        attention_labels (torch.Tensor): Labels matrix (Batch, N, N) or (N, N).
    
    Returns:
        list: List[List[float]] of computed percentages (per specific sample and per every valid row).
    """
    # Add batch axis if missing.
    if len(attention.shape) == 2:
        attention = attention.unsqueeze(0)
        attention_labels = attention_labels.unsqueeze(0)

    percentages = []
    for sample_idx in range(attention.size(0)):
        sample_percentages = []
        for row_idx in range(attention.size(1)):
            # Skip rows marked entirely as -100.
            if torch.all(attention_labels[sample_idx][row_idx] == -100):
                continue

            row_labels = attention_labels[sample_idx][row_idx]
            score_row = torch.abs(attention[sample_idx][row_idx])
            positions_with_1 = score_row[row_labels == 1].sum().item()
            positions_with_0 = score_row[row_labels == 0].sum().item()
            denom = positions_with_1 + positions_with_0
            
            if denom > 0:
                sample_percentages.append(positions_with_1 / denom)
            else:
                sample_percentages.append(0)
        percentages.append(sample_percentages)
    return percentages


def analyze_sample_consistency(percentages: List[List[float]]) -> dict:
    """
    Compute per-sample mean of percentages and return summary statistics
    to verify consistency across samples.
    
    Args:
        percentages (List[List[float]]): Each inner list contains percentages for a sample.
    
    Returns:
        dict: Summary statistics including overall mean, standard deviation,
              min, and max of the per-sample means.
    """

    sample_means = []
    for sample in percentages:
        if sample:  # if there are values for the sample
            sample_mean = sum(sample) / len(sample)
            sample_means.append(sample_mean)
    
    n = len(sample_means)
    overall_mean = sum(sample_means) / n if n > 0 else None
    variance = sum((m - overall_mean) ** 2 for m in sample_means) / (n - 1) if n > 1 else 0.0
    std_dev = math.sqrt(variance)
    
    return {
        "sample_means": sample_means,
        "overall_mean": overall_mean,
        "std_dev": std_dev,
        "min": min(sample_means) if sample_means else None,
        "max": max(sample_means) if sample_means else None
    }


def compare_attention_runs(percentages1: List[List[float]], percentages2: List[List[float]]) -> dict:
    """
    Compare two sequences of attention percentages using Welch's t-test.
    The inputs are lists of lists, where each inner list corresponds to the percentages
    computed for each valid row in a single sample.
    
    This function flattens the lists before performing the test.
    
    Args:
        percentages1 (List[List[float]]): Attention percentages from run 1.
        percentages2 (List[List[float]]): Attention percentages from run 2.
        
    Returns:
        dict: A dictionary with:
            - 't_stat': Computed t statistic,
            - 'df': Degrees of freedom,
            - 'p_value': Two-tailed p-value.
            
    Raises:
        ValueError: If either flattened list has fewer than 2 samples.
    """
    # Flatten the list of lists.
    flat1 = [value for sublist in percentages1 for value in sublist]
    flat2 = [value for sublist in percentages2 for value in sublist]
    
    n1, n2 = len(flat1), len(flat2)
    
    if n1 < 2 or n2 < 2:
        raise ValueError("Each sequence must have at least two samples to perform a t-test.")
    
    mean1 = sum(flat1) / n1
    mean2 = sum(flat2) / n2
    variance1 = sum((x - mean1) ** 2 for x in flat1) / (n1 - 1)
    variance2 = sum((x - mean2) ** 2 for x in flat2) / (n2 - 1)
    
    if n1 < 2 or n2 < 2 or (variance1 == 0 and variance2 == 0):
        return {
        't_stat': None,
        'df': None,
        'p_value': None,
        }

    se = math.sqrt(variance1 / n1 + variance2 / n2)
    t_stat = (mean1 - mean2) / se
    df_num = (variance1 / n1 + variance2 / n2) ** 2
    df_den = ((variance1 / n1) ** 2) / (n1 - 1) + ((variance2 / n2) ** 2) / (n2 - 1)
    df = df_num / df_den
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    
    return {
        't_stat': t_stat,
        'df': df,
        'p_value': p_value,
        'statistics': {
            "percentages1": {
                "mean": mean1,
                "variance": variance1,
                "sample_consistency": analyze_sample_consistency(percentages1)
            },
            "percentages2": {
                "mean": mean2,
                "variance": variance2,
                "sample_consistency": analyze_sample_consistency(percentages2)
            }
        }
    }


def evaluate_attention(attention_scores: torch.Tensor, attention_labels: torch.Tensor, compare_with: torch.Tensor = None) -> dict:
    """
    Evaluate the aggregated attention matrix against labels.
    For each row that is not entirely marked as -100 (i.e. skip rows),
    compute the fraction: (sum of scores for positions labeled as 1) / 
    (sum of scores for positions with labels in {0, 1}).
    Then compute the mean, variance, and 95% confidence interval over these fractions.
    
    Optionally, if compare_with is provided, compute the percentages for that matrix as well and
    compare them using Welch's t-test.
    
    For 95% CI, we use the normal approximation: mean ± 1.96 * (std / sqrt(n))
    
    Args:
        attention_scores (torch.Tensor): Aggregated attention matrix (N, N) or (Batch, N, N).
        attention_labels (torch.Tensor): Labels matrix (N, N) or (Batch, N, N) with:
            - 1 for significant tokens,
            - 0 for insignificant tokens, and
            - -100 for rows to skip.
        compare_with (torch.Tensor, optional): Second aggregated attention matrix to compare with.
    
    Returns:
        dict: A dictionary with keys:
            - 'mean': average percentage (float)
            - 'variance': sample variance (float)
            - 'ci_lower': lower bound of the 95% confidence interval (float)
            - 'ci_upper': upper bound of the 95% confidence interval (float)
            - 'comparison': (optional) result from comparing both runs if compare_with is provided.
              Contains 't_stat', 'df', and 'p_value'.
    """
    percentages_orig = compute_percentages(attention_scores, attention_labels)

    # Flatten
    percentages = [prob for sample in percentages_orig for prob in sample]
    n = len(percentages)
    mean = sum(percentages) / n
    variance = sum((x - mean) ** 2 for x in percentages) / (n - 1) if n > 1 else 0.0
    std_error = math.sqrt(variance) / math.sqrt(n) if n > 0 else 0.0
    ci_lower = mean - 1.96 * std_error
    ci_upper = mean + 1.96 * std_error

    result = {
        'mean': mean,
        'variance': variance,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'percentages': percentages,
        'sample_consistency': analyze_sample_consistency(percentages_orig)
    }

    # If compare_with is provided, compute its percentages and compare.
    if compare_with is not None:
        if compare_with.shape != attention_labels.shape:
            raise ValueError("compare_with tensor must have the same shape as attention_labels.")
        percentages_compare = compute_percentages(compare_with, attention_labels)
                    
        # Use Welch's t-test to compare the two runs.
        comparison = compare_attention_runs(percentages_orig, percentages_compare)
        result['comparison'] = comparison

    return result

def evaluate_attention_prediction_correlation(attention_scores: torch.Tensor,
                                              attention_labels: torch.Tensor,
                                              target_ids: torch.Tensor,
                                              predicted_ids: torch.Tensor,
                                              significance_level: float = 0.05) -> dict:
    """
    Evaluate whether the attention scores correlate with prediction correctness.
    For each token (row in the attention matrix) that is not entirely marked as -100 
    in attention_labels and where target_ids != -100, compute the percentage:
         (sum of scores for positions labeled as 1) / (sum of scores for positions with labels in {0, 1})
         Take the absolute value of scores in case of negative scores
    Then, group these percentages by whether the prediction matches the target.
    
    For the two groups (correct and incorrect), compute the mean, sample variance,
    and 95% confidence intervals. In addition, perform Welch's t-test to evaluate
    if the means are statistically significantly different.
    
    Args:
        attention_scores (torch.Tensor): Aggregated attention matrix with shape 
            (batch, seq_length, seq_length).
        attention_labels (torch.Tensor): Labels matrix with shape 
            (batch, seq_length, seq_length). Should contain values 1, 0, or -100.
        target_ids (torch.Tensor): Ground truth token ids, shape (batch, seq_length).
            Positions with -100 will be removed from evaluation.
        predicted_ids (torch.Tensor): Predicted token ids, shape (batch, seq_length).
        significance_level (float): Significance level for confidence interval (default: 0.05).
        
    Returns:
        dict: Dictionary containing:
            - 'correct': { 'mean', 'variance', 'ci_lower', 'ci_upper', 'n' }
            - 'error': { 'mean', 'variance', 'ci_lower', 'ci_upper', 'n' }
            - 't_stat': Welch's t statistic (or None if insufficient samples)
            - 'df': Degrees of freedom (or None)
            - 'p_value': Two-tailed p-value (or None)
    """
    B, S, N, _ = attention_scores.shape

    correct_percentages = []
    error_percentages = []

    for batch_idx  in range(B):
        for sample_idx in range(S):
            for token_idx in range(N):
                # Skip if this token should not be evaluated.
                if target_ids[batch_idx, sample_idx, token_idx].item() == -100:
                    continue

                # Skip if the entire row in attention_labels is -100.
                if torch.all(attention_labels[batch_idx, sample_idx, token_idx, :] == -100):
                    continue

                row_labels = attention_labels[batch_idx, sample_idx, token_idx, :]
                score_row = torch.abs(attention_scores[batch_idx, sample_idx, token_idx, :])

                pos1 = score_row[row_labels == 1].sum().item()
                pos0 = score_row[row_labels == 0].sum().item()
                denom = pos1 + pos0
                if denom == 0:
                    percentage = 0
                else:
                    percentage = pos1 / denom

                # Group based on prediction correctness.
                if predicted_ids[batch_idx, sample_idx, token_idx].item() == target_ids[batch_idx, sample_idx, token_idx].item():
                    correct_percentages.append(percentage)
                else:
                    error_percentages.append(percentage)

    # Inner function to compute statistics.
    def compute_stats(arr: list) -> dict:
        n = len(arr)
        if n == 0:
            return {'mean': None, 'variance': None, 'ci_lower': None, 'ci_upper': None, 'n': 0}
        mean_val = sum(arr) / n
        variance_val = sum((x - mean_val) ** 2 for x in arr) / (n - 1) if n > 1 else 0.0
        std_error = math.sqrt(variance_val) / math.sqrt(n) if n > 0 else 0.0
        ci_lower = mean_val - 1.96 * std_error
        ci_upper = mean_val + 1.96 * std_error
        return {'mean': mean_val,
                'variance': variance_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n': n}

    stats_correct = compute_stats(correct_percentages)
    stats_error = compute_stats(error_percentages)

    result = {'correct': stats_correct, 'error': stats_error}

    # Welch's t-test if both groups have at least 2 samples.
    if stats_correct['n'] >= 2 and stats_error['n'] >= 2 and (stats_correct['variance'] != 0.0 or stats_error['variance'] != 0.0):
        n1, n2 = stats_correct['n'], stats_error['n']
        mean1, mean2 = stats_correct['mean'], stats_error['mean']
        var1, var2 = stats_correct['variance'], stats_error['variance']
        se = math.sqrt(var1 / n1 + var2 / n2)
        t_stat = (mean1 - mean2) / se

        df_num = (var1 / n1 + var2 / n2) ** 2
        df_den = ((var1 / n1) ** 2) / (n1 - 1) + ((var2 / n2) ** 2) / (n2 - 1)
        df_val = df_num / df_den
        p_value = 2 * (1 - t.cdf(abs(t_stat), df_val))
        result.update({
            't_stat': t_stat,
            'df': df_val,
            'p_value': p_value
        })
    else:
        result.update({
            't_stat': None,
            'df': None,
            'p_value': None
        })

    return result


def analyze_error_positions(piecewise_matrix, visualize=True):
    """
    Analyzes a non-homogeneous list of piecewise accuracy lists. For each sequence (list of 0s and 1s),
    it calculates:
      - Overall piecewise accuracy: overall percentage of correctly predicted tokens.
      - Mean and variance of the error positions (all errors across sequences, 0-indexed).
      - A histogram of error positions.
      
    Parameters:
        piecewise_matrix (list of lists of int): Each sublist represents a sequence
                                 with 1 for correct prediction and 0 for error.
                                 Sequences can have different lengths.
    Returns:
        dict: Dictionary with overall_piecewise_accuracy, mean_error_position, and variance.
    """
    all_error_positions = []
    total_correct = 0
    total_tokens = 0

    for seq in piecewise_matrix:
        total_tokens += len(seq)
        total_correct += sum(seq)
        # Record error positions from this sequence.
        errors = [i for i, val in enumerate(seq) if val == 0]
        all_error_positions.extend(errors)

    overall_accuracy = (total_correct / total_tokens) * 100 if total_tokens > 0 else 0.0

    if all_error_positions:
        mean_error_position = np.mean(all_error_positions)
        var_error_position = np.var(all_error_positions)
    else:
        mean_error_position = None
        var_error_position = None

    print(f"Overall piecewise accuracy: {overall_accuracy:.2f}%")
    if mean_error_position is not None:
        print(f"Mean error position (0-indexed): {mean_error_position:.2f}")
        print(f"Variance of error positions: {var_error_position:.2f}")
    else:
        print("No errors present in any sequence.")
    if visualize:
        # Determine the maximum sequence length for histogram binning.
        max_length = max(len(seq) for seq in piecewise_matrix) if piecewise_matrix else 0
        plt.figure(figsize=(8, 4))
        plt.hist(all_error_positions, bins=range(0, max_length + 1), edgecolor='black', align='left')
        plt.xlabel("Error Position (0-indexed)")
        plt.ylabel("Number of Occurrences")
        plt.title("Distribution of Error Positions")
        plt.tight_layout()
        plt.show()

    return {
        "overall_piecewise_accuracy": overall_accuracy,
        "mean_error_position": mean_error_position,
        "variance": var_error_position
    }