from generators.string_reversal import StringReversal
from generators.successor import Successor
from attention_evaluation import aggregate_attention, evaluate_attention, visualize_attention

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from models import DiffTransformer, SingleCharTokenizer

import itertools
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm import tqdm

#model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager")
#tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = DiffTransformer(
    vocab_size=32,
    d_model=128,
    num_heads=4,
    num_layers=4,
    max_seq_length=512
)
vocabulary = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", " ", "<PAD>"]
tokenizer = SingleCharTokenizer(vocabulary)

data_generator = Successor(seed=42, tokenizer=tokenizer, length=(10, 20), start_number_range=(1, 25), generate_attn_labels=False)

samples = itertools.islice(data_generator.generate_samples(), 100000)

train_dataset = data_generator.generate_dataset(samples, pad_token=tokenizer.char2token["<PAD>"])
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=False)

# Setup device, optimizer and move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Define number of training epochs
num_epochs = 1

# Define a loss criterion that ignores indices set to -100 (commonly used to mask out tokens)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)



# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    train_iterator = tqdm(train_dataloader)
    for batch in train_iterator:
        # Get the tensors and move them to the device
        input_ids = batch["input_ids"].to(device)
        labels = batch["target_ids"].to(device)
        attn_labels = batch["attn_labels"]
        texts = batch["texts"]

        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass: GPT2LMHeadModel accepts labels and returns loss automatically.
        logits, attn_scores = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        train_iterator.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


data_generator = Successor(seed=42, tokenizer=tokenizer, length=(10, 10), start_number_range=(25, 50))

samples = itertools.islice(data_generator.generate_samples(
), 10)

test_dataset = data_generator.generate_dataset(samples, pad_token=tokenizer.char2token["<PAD>"])
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()
num_correct = 0
num_total = 0

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["target_ids"].to(device)
        attn_labels = batch["attn_labels"]
        texts = batch["texts"]
        
        # Generate predictions.
        # max_length is set to ensure we cover the full target length.
        outputs, attn_scores = model.forward(input_ids)
        pred = torch.argmax(outputs, dim=-1)

        for gen_ids, label_ids in zip(pred, labels):
             # Convert tensors to lists.
            gen_ids_list = gen_ids.tolist()
            label_ids_list = label_ids.tolist()
            
            # Get indices where label_ids are not -100.
            valid_indices = [i for i, lbl in enumerate(label_ids_list) if lbl != -100]
            
            # Filter both generated ids and label ids.
            filtered_gen_ids = [gen_ids_list[i] for i in valid_indices]
            filtered_label_ids = [label_ids_list[i] for i in valid_indices]
            
            pred_str = tokenizer.decode(filtered_gen_ids).strip()
            target_str = tokenizer.decode(filtered_label_ids).strip()
            
            # Compare prediction to target.
            if pred_str == target_str:
                num_correct += 1
            num_total += 1
        #attentions = torch.stack(outputs.attentions, dim=1)
        visualize_attention(aggregate_attention(attn_scores).squeeze(0), attn_labels.squeeze(0), texts[0])

accuracy = (num_correct / num_total) * 100
print(f"Test Accuracy: {accuracy:.2f}%")


data_generator = Successor(seed=42, tokenizer=tokenizer, length=(10, 10), start_number_range=(1, 25))

samples = itertools.islice(data_generator.generate_samples(
), 10)

test_dataset = data_generator.generate_dataset(samples, pad_token=tokenizer.char2token["<PAD>"])
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()
num_correct = 0
num_total = 0

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["target_ids"].to(device)
        attn_labels = batch["attn_labels"]
        texts = batch["texts"]
        
        # Generate predictions.
        # max_length is set to ensure we cover the full target length.
        outputs, attn_scores = model.forward(input_ids)
        pred = torch.argmax(outputs, dim=-1)

        for gen_ids, label_ids in zip(pred, labels):
             # Convert tensors to lists.
            gen_ids_list = gen_ids.tolist()
            label_ids_list = label_ids.tolist()
            
            # Get indices where label_ids are not -100.
            valid_indices = [i for i, lbl in enumerate(label_ids_list) if lbl != -100]
            
            # Filter both generated ids and label ids.
            filtered_gen_ids = [gen_ids_list[i] for i in valid_indices]
            filtered_label_ids = [label_ids_list[i] for i in valid_indices]
            
            pred_str = tokenizer.decode(filtered_gen_ids).strip()
            target_str = tokenizer.decode(filtered_label_ids).strip()
            
            # Compare prediction to target.
            if pred_str == target_str:
                num_correct += 1
            num_total += 1
        #attentions = torch.stack(outputs.attentions, dim=1)
        visualize_attention(aggregate_attention(attn_scores).squeeze(0), attn_labels.squeeze(0), texts[0])

accuracy = (num_correct / num_total) * 100
print(f"Test Accuracy: {accuracy:.2f}%")