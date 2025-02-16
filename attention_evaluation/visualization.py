import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import os
import pickle

def visualize_attention(attn_scores: torch.Tensor, attn_labels: torch.Tensor, tokens, name="Unnamed", remove_ignore_values=True):
    attn_scores_np = attn_scores.cpu().numpy().copy()
    attn_labels_np = attn_labels.cpu().numpy().copy()
    
    row_tokens = tokens
    col_tokens = tokens
    if remove_ignore_values:
        # Identify rows and columns to keep (those not containing only -100)
        rows_to_keep = ~np.all(attn_labels_np == -100, axis=1)
        cols_to_keep = ~np.all(attn_labels_np == -100, axis=0)
        # Keep only the rows and columns that do not contain only -100
        attn_scores_np = attn_scores_np[rows_to_keep, :][:, cols_to_keep]
        attn_labels_np = attn_labels_np[rows_to_keep, :][:, cols_to_keep]
        row_tokens = ["<>" if len(token) > 1 else token for i, token in enumerate(tokens) if rows_to_keep[i]]
        col_tokens = ["<>" if len(token) > 1 else token for i, token in enumerate(tokens) if cols_to_keep[i]]
        
    rows, columns = attn_scores_np.shape
    assert attn_scores_np.shape == attn_labels_np.shape

    # Normalize attention scores to the range [0, 1]
    attn_scores_np = (attn_scores_np - attn_scores_np.min()) / (attn_scores_np.max() - attn_scores_np.min())
    
    assert len(row_tokens) == rows and len(col_tokens) == columns

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"Attention Scores on {name}")
    cax = ax.imshow(attn_scores_np, cmap="gray", interpolation=None)
    fig.colorbar(cax, ax=ax, label="Attention Score")

    ax.set_xticks(np.arange(columns))
    ax.set_yticks(np.arange(rows))
    if 20 > columns:
        # Set major ticks to token positions and use token indices as labels if seq_len is large.
        ax.set_xticklabels(col_tokens, fontsize=200//columns, rotation=45)
        ax.set_yticklabels(row_tokens, fontsize=200//columns, rotation=45)
        linewidth = 50/columns
    elif 50 > columns:
        # Set major ticks to token positions and use token indices as labels if seq_len is large.
        ax.set_xticklabels(col_tokens, fontsize=400//columns, rotation=45)
        ax.set_yticklabels(row_tokens, fontsize=400//columns, rotation=45)
        linewidth = 100/columns
    elif 100 > columns:
        # Set major ticks to token positions and use token indices as labels if seq_len is large.
        ax.set_xticklabels(col_tokens, fontsize=800//columns, rotation=45)
        ax.set_yticklabels(row_tokens, fontsize=800//columns, rotation=45)
        linewidth = 200/columns
    else:
        # Set major ticks to token positions and use token indices as labels if seq_len is large.
        ax.set_xticklabels(col_tokens, fontsize=1600//columns, rotation=45)
        ax.set_yticklabels(row_tokens, fontsize=1600//columns, rotation=45)
        linewidth = 400/columns
    


    
    # Set up grid lines manually via minor ticks for clarity.
    ax.set_xticks(np.arange(-.5, columns, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=linewidth)
    ax.tick_params(which="minor", bottom=False, left=False)



    ax.set_xlabel("Tokens")
    ax.set_ylabel("Tokens")
    
    # Draw manual border segments only for the external borders of regions where attn_labels_np == 1.
    for i in range(rows):
        for j in range(columns):
            if attn_labels_np[i, j] == 1:
                # Left border
                if j == 0 or attn_labels_np[i, j-1] != 1:
                    ax.plot([j - 0.5, j - 0.5], [i - 0.5, i + 0.5], color='red', linewidth=linewidth, alpha=1)
                # Right border
                if j == columns - 1 or attn_labels_np[i, j+1] != 1:
                    ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color='red', linewidth=linewidth, alpha=1)
                # Top border
                if i == 0 or attn_labels_np[i-1, j] != 1:
                    ax.plot([j - 0.5, j + 0.5], [i - 0.5, i - 0.5], color='red', linewidth=linewidth, alpha=1)
                # Bottom border
                if i == rows - 1 or attn_labels_np[i+1, j] != 1:
                    ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color='red', linewidth=linewidth, alpha=1)

    plt.tight_layout()
    if not os.path.exists("visualizations"):
        os.mkdir("visualizations")
    plt.savefig("visualizations/" + name + ".png")
    plt.show()

def visualize_scores_on_reference_map(attn_scores, attn_labels, tokens, name, linewidth=2, remove_ignore_values=True):
    """
    Visualize significant diagonal tokens based on attention labels.
    Only tokens where attn_labels_np[index, index] == 1 are plotted.
    
    Parameters:
        attn_scores (np.ndarray): 2D array of attention scores.
        attn_labels_np (np.ndarray): 2D array of attention labels (significance indicator).
        name (str): Name for saving the figure.
        linewidth (float): Line width for visualization.
    """
    attn_scores_np = attn_scores.cpu().numpy().copy()
    attn_labels_np = attn_labels.cpu().numpy().copy()
    
    row_tokens = tokens
    col_tokens = tokens
    if remove_ignore_values:
        # Identify rows and columns to keep (those not containing only -100)
        rows_to_keep = ~np.all(attn_labels_np == -100, axis=1)
        cols_to_keep = ~np.all(attn_labels_np == -100, axis=0)
        # Keep only the rows and columns that do not contain only -100
        attn_scores_np = attn_scores_np[rows_to_keep, :][:, cols_to_keep]
        attn_labels_np = attn_labels_np[rows_to_keep, :][:, cols_to_keep]
        row_tokens = ["<>" if len(token) > 1 else token for i, token in enumerate(tokens) if rows_to_keep[i]]
        col_tokens = ["<>" if len(token) > 1 else token for i, token in enumerate(tokens) if cols_to_keep[i]]
    
    pkl_obj = (attn_scores_np, attn_labels_np, row_tokens, col_tokens)
    with open(f"attentions/{name}.pkl", "wb") as f:
        pickle.dump(pkl_obj, f)

    rows, cols = attn_scores_np.shape
    assert attn_scores_np.shape == attn_labels_np.shape

    for row in range(rows):
        for col in range(cols):
            if attn_labels_np[row, col] == 0 or attn_labels_np[row, col] == -100:
                attn_scores_np[row, col] = 0
    
    # Normalize attention scores to the range [0, 1]
    attn_scores_np = (attn_scores_np - attn_scores_np.min()) / (attn_scores_np.max() - attn_scores_np.min())
    
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"Attention Scores on {name}")
    cax = ax.imshow(attn_scores_np, cmap="viridis", interpolation=None)
    fig.colorbar(cax, ax=ax, label="Attention Score")

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))

    # Set major ticks to token positions and use token indices as labels if seq_len is large.
    #ax.set_xticklabels(col_tokens, fontsize=200//cols, rotation=45)
    #ax.set_yticklabels(row_tokens, fontsize=200//cols, rotation=45)
    linewidth = 0.1

    # Set up grid lines manually via minor ticks for clarity.
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=linewidth)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlabel("Tokens")
    ax.set_ylabel("Tokens")

    plt.tight_layout()
    if not os.path.exists("visualizations"):
        os.mkdir("visualizations")
    plt.savefig("visualizations/" + name + ".png")
    plt.show()