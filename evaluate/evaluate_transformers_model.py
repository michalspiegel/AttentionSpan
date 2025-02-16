import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../generators")
sys.path.insert(0, "../attention_evaluation")

from generators import *
from attention_evaluation import aggregate_attention, evaluate_attention, visualize_attention, compare_attention_runs, evaluate_attention_prediction_correlation, visualize_scores_on_reference_map, analyze_error_positions

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader, Dataset
import torch
from torch.optim import AdamW
from tqdm import tqdm
import itertools
from typing import Tuple
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import random
import numpy as np

OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))

@hydra.main(config_path="configs", config_name="transformers_config", version_base="1.2")
def main(cfg: DictConfig):
    wandb.login()
    wandb.init(**cfg.wandb, config=dict(cfg))
    model, tokenizer = load_model(cfg.model)
    
    if cfg.training is not None:
        train_dataset = load_dataset(tokenizer, cfg.training.train_dataset)
        val_dataset = load_dataset(tokenizer, cfg.training.val_dataset)
        model = train(model, train_dataset, val_dataset, cfg.training)
    
    # Load two evaluation datasets from the config.
    eval_dataset1 = load_dataset(tokenizer, cfg.evaluation.dataset1)
    eval_dataset2 = load_dataset(tokenizer, cfg.evaluation.dataset2)
    
    accuracy = evaluate_two_distributions(model, tokenizer, eval_dataset1, eval_dataset2, cfg.evaluation)
    wandb.finish()
    return accuracy



def load_model(cfg: DictConfig):
    # The model should use eager attention implementation to be able to return attentions
    if cfg.is_pretrained:
        model = AutoModelForCausalLM.from_pretrained(cfg.name, attn_implementation="eager")
    else:
        config = AutoConfig.from_pretrained(cfg.name, attn_implementation="eager")
        model = AutoModelForCausalLM.from_config(config)
    
    if cfg.load_from_checkpoint:
        model.load_state_dict(torch.load(cfg.checkpoint_filepath, weights_only=True))
    tokenizer = AutoTokenizer.from_pretrained(cfg.name)
    return model, tokenizer


def load_dataset(tokenizer, config: DictConfig) -> Dataset:
    data_generator = globals()[config.generator](tokenizer=tokenizer, **config.generator_parameters)
    samples = itertools.islice(data_generator.generate_samples(), config.num_samples)
    dataset = data_generator.generate_dataset(samples)
    return dataset


def train(model: torch.nn.Module, train_dataset: Dataset, val_dataset: Dataset, config: DictConfig):
    if config.batch_size >= 64 and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_data_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    device = torch.device(config.device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=config.ignore_index)
    model.train()
    step = 0
    for epoch in range(config.epochs):
        total_loss = 0.0
        train_iterator = tqdm(train_data_loader)
        for batch in train_iterator:
            input_ids = batch["input_ids"].to(device)
            labels = batch["target_ids"].to(device)
            #attn_labels = batch["attn_labels"]
            #texts = batch["texts"]
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            step += 1
            total_loss += loss.item()
            train_iterator.set_postfix(loss=loss.item())
            if step % 10 == 0:
                wandb.log({"train_loss": loss.item()})
                grad_norm = np.sqrt(sum([torch.norm(p.grad.detach().cpu())**2 for p in model.parameters()]))
                wandb.log({"grad_norm": grad_norm})
            if step % 50 == 0:
                val_acc = run_inference(model, val_data_loader, config)
                wandb.log({"val/acc": val_acc})

        avg_loss = total_loss / len(train_data_loader)
        print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {avg_loss:.4f}")
        wandb.log({"avg_train_loss": avg_loss})
    
    if config.save_checkpoint:
        torch.save(model.state_dict(), config.checkpoint_filepath)

    return model

# Function to register hooks to capture gradients for each head.
def capture_head_gradients(attn_tensor, storage):
    # attn_tensor has shape (batch, layers, heads, N, N)
    def hook(grad):
        storage.append(grad)
    attn_tensor.register_hook(hook)

@torch.no_grad()
def run_inference(model: torch.nn.Module, data_loader: DataLoader, cfg: DictConfig):
    accurate = 0
    total = 0
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        input_ids = batch["input_ids"].to(cfg.device)
        labels = batch["target_ids"].to(cfg.device)
        attn_labels = batch["attn_labels"].to(cfg.device)
        outputs = model.forward(input_ids, output_attentions=True)
        preds = torch.argmax(outputs.logits, dim=-1)
    
        # Process predictions.
        for pred_ids, inpt_ids, label_ids in zip(preds, input_ids, labels):
            pred_list = pred_ids.tolist()
            label_list = label_ids.tolist()
            # Remove positions with ignore index (-100).
            valid_idx = [i for i, lbl in enumerate(label_list) if lbl != -100]
            filtered_preds = [pred_list[i] for i in valid_idx]
            filtered_labels = [label_list[i] for i in valid_idx]
            if filtered_preds == filtered_labels:
                accurate += 1
            total += 1
        
    return accurate / total


@torch.no_grad()
def evaluate_two_distributions(model: torch.nn.Module, tokenizer, dataset1: Dataset, dataset2: Dataset, config: DictConfig):
    """
    Evaluate the model on two different test distributions. For each distribution, we:
      - Compute prediction accuracy.
      - Aggregate attention scores.
      - Evaluate the aggregated attention statistics.
      - Validate whether the attention scores for individual tokens correlate with prediction correctness.
    Finally, we compare the attention distributions across the two datasets.
    """
    device = torch.device(config.device)
    model.eval()
    model.to(device)
    
    # Prepare accumulators for dataset-level evaluation.
    eval_results = {"dataset1": {"accurate": 0, "total": 0, "percentages": []},
                    "dataset2": {"accurate": 0, "total": 0, "percentages": []}}
    
    # Define a helper function to process one dataset.
    def process_dataset(dataset, key: str):
        data_loader = DataLoader(dataset, batch_size=config.batch_size, drop_last=True)
        # Accumulators for attention prediction correlation.
        all_target_ids = []
        all_predicted_ids = []
        all_attentions = []
        all_attn_labels = []
        all_aggr_att = []
        all_piecewice_acc = []
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input_ids = batch["input_ids"].to(device)
            labels = batch["target_ids"].to(device)
            attn_labels = batch["attn_labels"].to(device)
            outputs = model.forward(input_ids, output_attentions=True)
            preds = torch.argmax(outputs.logits, dim=-1)
            # Aggregate attention.
            # outputs.attentions is a tuple of tensors at each layer.
            attentions = torch.stack(outputs.attentions, dim=1)  # shape: (batch, layers, heads, N, N)
            attentions = attentions.detach()
            agg_attn = aggregate_attention(attentions)
            # Process predictions.
            id = 0
            for pred_ids, inpt_ids, label_ids in zip(preds, input_ids, labels):
                pred_list = pred_ids.tolist()
                label_list = label_ids.tolist()
                # Remove positions with ignore index (-100).
                valid_idx = [i for i, lbl in enumerate(label_list) if lbl != -100]
                filtered_preds = [pred_list[i] for i in valid_idx]
                filtered_labels = [label_list[i] for i in valid_idx]
                if filtered_preds == filtered_labels:
                    eval_results[key]["accurate"] += 1
                piecewice_acc = []
                for pred, label in zip(filtered_preds, filtered_labels):
                    if pred == label:
                        piecewice_acc.append(1)
                    else:
                        piecewice_acc.append(0)
                all_piecewice_acc.append(piecewice_acc)

                eval_results[key]["total"] += 1
                id += 1
            
            # Optionally visualize the attention.
            if config.visualize and input_ids.size(0) == 1:
                tokens = [tokenizer.decode(t, skip_special_tokens=False) for t in input_ids[0]]
                visualize_attention(agg_attn.squeeze(0), attn_labels.squeeze(0), tokens,
                                      name=config.name + "_" + str(random.getrandbits(32)))
                visualize_scores_on_reference_map(agg_attn.squeeze(0), attn_labels.squeeze(0), tokens,
                                      name=config.name + "_" + str(random.getrandbits(32)))
            # Compute percentages from aggregated attention.
            percentages = []  # Use the provided compute_percentages logic inside evaluate_attention.
            attn_eval = evaluate_attention(agg_attn, attn_labels)
            percentages = attn_eval.get("percentages", [])
            eval_results[key]["percentages"] += [percentages]
            
            # For attention prediction correlation evaluation.
            all_target_ids.append(labels)
            all_predicted_ids.append(preds)
            all_attentions.append(agg_attn)
            all_attn_labels.append(attn_labels)
            all_aggr_att.append(agg_attn)
            
        # Compute and log prediction accuracy.
        accuracy = (eval_results[key]["accurate"] / eval_results[key]["total"]) * 100 if eval_results[key]["total"] > 0 else 0.0
        wandb.log({f"{key}_accuracy": accuracy})
        print(f"{key} Accuracy: {accuracy:.2f}%")

        piecewise_acc_analysis= analyze_error_positions(all_piecewice_acc)
        print(f"{key}_piecewise_acc: {piecewise_acc_analysis}")
        wandb.log({f"{key}_piecewise_acc": piecewise_acc_analysis})


        attn_eval = evaluate_attention(torch.stack(all_aggr_att,dim=0), torch.stack(all_attn_labels, dim=0))
        wandb.log({f"{key}_attention_eval": attn_eval})
        print(f"{key} Attention Evaluation: {attn_eval}")

        corr_result = evaluate_attention_prediction_correlation(
            torch.stack(all_attentions, dim=0),
            torch.stack(all_attn_labels, dim=0),
            torch.stack(all_target_ids, dim=0),
            torch.stack(all_predicted_ids, dim=0)
        )
        wandb.log({f"{key}_attn_prediction_corr": corr_result})
        print(f"{key} Attention-Prediction Correlation: {corr_result}")
    
    # Process both datasets.
    process_dataset(dataset1, "dataset1")
    process_dataset(dataset2, "dataset2")
    
    # Now, compare the attention percentages across the two datasets using the provided function.
    try:
        comparison = compare_attention_runs(eval_results["dataset1"]["percentages"],
                                              eval_results["dataset2"]["percentages"])
        wandb.log({"attn_distribution_comparison": comparison})
        print("Attention Distribution Comparison:", comparison)
    except ValueError as e:
        print("Insufficient samples for attention comparison:", e)
        wandb.log({"attn_distribution_comparison": str(e)})
    
    # Return accuracy on test data
    return  (eval_results["dataset2"]["accurate"] / eval_results["dataset2"]["total"])

if __name__ == "__main__":
    main()