from typing import Tuple, List
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class GeneratedDataset(Dataset):
    def __init__(self, texts: List[str], input_ids: torch.Tensor, target_ids: torch.Tensor, attn_labels: torch.Tensor):
        self.texts = texts
        self.input_ids = input_ids
        self.target_ids = target_ids
        self.attn_labels = attn_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "texts": self.texts[idx],
            "input_ids": self.input_ids[idx],
            "target_ids": self.target_ids[idx],
            "attn_labels": self.attn_labels[idx]
        }


class DataGenerator:
    def __init__(self):
        return NotImplementedError()

    def generate_sample(self):
        return NotImplementedError()

    def generate_samples(self):
        while True:
            yield self.generate_sample()
    
    def generate_dataset(self, samples: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]], pad_token=0, ignore_index=-100) -> Dataset:

        texts, input_ids_list, target_ids_list, attn_labels_list = zip(*samples)

        # Pad 1D sequences for input_ids and target_ids
        input_ids_batch = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token)
        target_ids_batch = pad_sequence(target_ids_list, batch_first=True, padding_value=ignore_index)

        if attn_labels_list[0] is None:
            print(f"Generated dataset with {len(texts)} samples; batched input shape: {input_ids_batch.shape}")
            return GeneratedDataset(list(texts), input_ids_batch, target_ids_batch, torch.full((input_ids_batch.size(0), 1), float('nan')))

        
        # Pad attention label matrices so all samples have the same dimensions
        max_seq_len = max(attn.shape[0] for attn in attn_labels_list)
        padded_attn_labels_list = []
        for attn in attn_labels_list:
            current_seq_len = attn.shape[0]
            if current_seq_len < max_seq_len:
                # Pad rows
                pad_rows = torch.full((max_seq_len - current_seq_len, attn.shape[1]), ignore_index, dtype=attn.dtype)
                attn = torch.cat([attn, pad_rows], dim=0)
                # Pad columns
                pad_cols = torch.full((max_seq_len, max_seq_len - attn.shape[1]), ignore_index, dtype=attn.dtype)
                attn = torch.cat([attn, pad_cols], dim=1)
            padded_attn_labels_list.append(attn)
        attn_labels_batch = torch.stack(padded_attn_labels_list)

        print(f"Generated dataset with {len(texts)} samples; batched input shape: {input_ids_batch.shape}")
            
        return GeneratedDataset(list(texts), input_ids_batch, target_ids_batch, attn_labels_batch)
