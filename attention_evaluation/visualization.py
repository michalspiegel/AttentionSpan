import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def visualize_attention(attn_scores: torch.Tensor, attn_labels: torch.Tensor, text_sample: str, cmap='viridis', name="Unnamed"):
    attn_scores_np = attn_scores.cpu().numpy()
    attn_labels_np = attn_labels.cpu().numpy()

    seq_len = attn_scores_np.shape[0]
    assert attn_scores_np.shape == attn_labels_np.shape

    # If text_sample is shorter than seq_len, pad with spaces.
    if len(text_sample) < seq_len:
        text_sample = text_sample + " " * (seq_len - len(text_sample))
    # If text_sample is longer than seq_len, truncate.
    elif len(text_sample) > seq_len:
        text_sample = text_sample[:seq_len]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"Attention Scores on {name}")
    cax = ax.imshow(attn_scores_np, cmap=cmap, interpolation="nearest")
    fig.colorbar(cax, ax=ax, label="Attention Score")

    # Set major ticks to token positions and use characters as labels.
    ax.set_xticks(np.arange(seq_len))
    ax.set_yticks(np.arange(seq_len))
    ax.set_xticklabels(list(text_sample), fontsize=10, rotation=90)
    ax.set_yticklabels(list(text_sample), fontsize=10)

    # Set up grid lines manually via minor ticks for clarity.
    ax.set_xticks(np.arange(-.5, seq_len, 1), minor=True)
    ax.set_yticks(np.arange(-.5, seq_len, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlabel("Token")
    ax.set_ylabel("Token")
    
    # Draw manual border segments only for the external borders of regions where attn_labels_np == 1.
    for i in range(seq_len):
        for j in range(seq_len):
            if attn_labels_np[i, j] == 1:
                # Left border
                if j == 0 or attn_labels_np[i, j-1] != 1:
                    ax.plot([j - 0.5, j - 0.5], [i - 0.5, i + 0.5], color='red', linewidth=2)
                # Right border
                if j == seq_len - 1 or attn_labels_np[i, j+1] != 1:
                    ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color='red', linewidth=2)
                # Top border
                if i == 0 or attn_labels_np[i-1, j] != 1:
                    ax.plot([j - 0.5, j + 0.5], [i - 0.5, i - 0.5], color='red', linewidth=2)
                # Bottom border
                if i == seq_len - 1 or attn_labels_np[i+1, j] != 1:
                    ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color='red', linewidth=2)


    plt.tight_layout()
    plt.savefig("visualizations/" + name + ".png")
    plt.show()