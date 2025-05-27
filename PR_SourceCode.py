import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from datasets import load_dataset
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
import re

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def setup_device():
    """Setup and verify GPU availability"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")
    return device

# Positional Encoding
class PositionalEncoding(nn.Module):
    """Positional encoding using sinusoidal functions to capture word order"""
    def __init__(self, d_model, max_seq_length=256):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

# Multi-Head Self-Attention
class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism with scaled dot-product attention"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(0.1)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        device = x.device
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        mask = causal_mask.unsqueeze(0).unsqueeze(0)
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(attention_output)
        return output

# Feed-Forward Neural Network
class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network for each transformer layer"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

# Transformer Decoder Layer
class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with layer normalization and residual connections"""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_output = self.self_attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# GPT-2 Model
class GPT2Model(nn.Module):
    """GPT-2 transformer model with a stack of decoder layers"""
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=256):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(0.1)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)
        x = self.final_norm(x)
        logits = self.output_projection(x)
        return logits

# Dataset class
class TinyStoriesDataset(Dataset):
    """Dataset class for TinyStories with on-the-fly preprocessing"""
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            text = self.texts[idx]
            if not text or not text.strip() or len(text.strip()) < 2:
                return self._get_dummy_sample()
            text = re.sub(r'\s+', ' ', text.strip())
            text = re.sub(r'[^\w\s.,!?]', '', text)
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)[:-1]
            target_ids = encoding['input_ids'].squeeze(0)[1:]
            if input_ids.size(0) != self.max_length-1 or target_ids.size(0) != self.max_length-1:
                return self._get_dummy_sample()
            return input_ids, target_ids
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return self._get_dummy_sample()

    def _get_dummy_sample(self):
        """Return a padded dummy sample with correct shape"""
        dummy = torch.full((self.max_length-1,), self.tokenizer.pad_token_id, dtype=torch.long)
        return dummy, dummy

# Custom collate function
def collate_fn(batch):
    """Custom collation to ensure consistent tensor shapes"""
    input_ids, target_ids = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    target_ids = torch.stack(target_ids, dim=0)
    return input_ids, target_ids

# Load and preprocess data
def load_and_preprocess_data(tokenizer, max_samples=270000, train_split=0.7, val_split=0.2):
    """Load and preprocess TinyStories dataset with train/validation/test splits"""
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    texts = [item['text'] for item in dataset.select(range(min(max_samples, len(dataset))))
             if item['text'] and item['text'].strip() and len(item['text'].strip()) >= 2]
    if len(texts) < max_samples:
        print(f"Warning: Only {len(texts)} valid samples found after filtering")
    train_end = int(len(texts) * train_split)
    val_end = train_end + int(len(texts) * val_split)
    train_texts = texts[:train_end]
    val_texts = texts[train_end:val_end]
    test_texts = texts[val_end:]
    print(f"Loaded {len(train_texts)} training samples, {len(val_texts)} validation samples, and {len(test_texts)} test samples")
    train_dataset = TinyStoriesDataset(train_texts, tokenizer)
    val_dataset = TinyStoriesDataset(val_texts, tokenizer)
    test_dataset = TinyStoriesDataset(test_texts, tokenizer)
    return train_dataset, val_dataset, test_dataset

# Training function with early stopping
def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=2e-4, warmup_steps=500, patience=3, config=None):
    """Training loop with cross-entropy loss, AdamW optimizer, and early stopping"""
    model.to(device)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.95))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup_steps, 1.0) if step < warmup_steps else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))))
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    accum_steps = 8
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_tokens = 0
        train_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (input_ids, target_ids) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids)
                    loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            else:
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

            predictions = torch.argmax(logits, dim=-1)
            non_pad_mask = (target_ids != 0)
            correct = (predictions == target_ids) & non_pad_mask
            total_train_correct += correct.sum().item()
            total_train_tokens += non_pad_mask.sum().item()

            if scaler is not None:
                scaler.scale(loss / accum_steps).backward()
            else:
                (loss / accum_steps).backward()

            if (batch_idx + 1) % accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            total_train_loss += loss.item()
            train_steps += 1

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        avg_train_loss = total_train_loss / train_steps
        train_losses.append(avg_train_loss)
        train_accuracy = (total_train_correct / total_train_tokens) * 100 if total_train_tokens > 0 else 0
        train_accuracies.append(train_accuracy)

        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        total_val_tokens = 0
        val_steps = 0

        with torch.no_grad():
            for input_ids, target_ids in tqdm(val_loader, desc="Validation"):
                input_ids = input_ids.to(device, non_blocking=True)
                target_ids = target_ids.to(device, non_blocking=True)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits = model(input_ids)
                        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                else:
                    logits = model(input_ids)
                    loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

                predictions = torch.argmax(logits, dim=-1)
                non_pad_mask = (target_ids != 0)
                correct = (predictions == target_ids) & non_pad_mask
                total_val_correct += correct.sum().item()
                total_val_tokens += non_pad_mask.sum().item()

                total_val_loss += loss.item()
                val_steps += 1

                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        avg_val_loss = total_val_loss / val_steps
        val_losses.append(avg_val_loss)
        val_accuracy = (total_val_correct / total_val_tokens) * 100 if total_val_tokens > 0 else 0
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Train Acc = {train_accuracy:.2f}%, Val Acc = {val_accuracy:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save({'model_state_dict': model.state_dict(), 'config': config}, 'gpt2_best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return train_losses, val_losses, train_accuracies, val_accuracies

# Perplexity calculation
def calculate_perplexity(model, dataloader, device):
    """Calculate perplexity on a held-out test set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    with torch.no_grad():
        for input_ids, target_ids in tqdm(dataloader, desc="Calculating perplexity"):
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids)
                    loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            else:
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            non_pad_tokens = (target_ids != 0).sum().item()
            total_loss += loss.item()
            total_tokens += non_pad_tokens
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

# Text generation
def generate_text(model, tokenizer, prompt="Once upon a time", max_length=100, temperature=1.0, device='cpu', top_k=40, top_p=0.95):
    """Generate text samples using the trained model"""
    model.eval()
    model.to(device)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    generated_ids = input_ids.copy()
    with torch.no_grad():
        for _ in range(max_length - len(input_ids)):
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = model(input_tensor)
            else:
                logits = model(input_tensor)
            next_token_logits = logits[0, -1, :] / temperature
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated_ids.append(next_token)
            max_context = min(256, len(generated_ids))
            input_tensor = torch.tensor([generated_ids[-max_context:]], dtype=torch.long).to(device)
            if next_token == tokenizer.eos_token_id:
                break
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text

# Main function
def main():
    """Main function to run the complete pipeline"""
    device = setup_device()
    config = {
        'vocab_size': 50257,
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 2048,
        'max_seq_length': 256,
        'batch_size': 4 if device.type == 'cuda' else 2,
        'learning_rate': 2e-4,
        'num_epochs': 10,
        'max_samples': 270000
    }
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, val_dataset, test_dataset = load_and_preprocess_data(tokenizer, max_samples=config['max_samples'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )
    model = GPT2Model(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_length=config['max_seq_length']
    )
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, device, 
        num_epochs=config['num_epochs'], 
        learning_rate=config['learning_rate'],
        config=config
    )
    perplexity = calculate_perplexity(model, test_loader, device)
    print(f"Final perplexity on test set: {perplexity:.2f}")
    prompts = ["Once upon a time", "The little girl", "In a magical forest", "The brave knight"]
    generated_texts = []
    for prompt in prompts:
        generated_text = generate_text(
            model, tokenizer, prompt=prompt, max_length=100,
            temperature=1.0, device=device, top_k=40, top_p=0.95
        )
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated_text}")
        print("-" * 50)
        generated_texts.append({'prompt': prompt, 'generated': generated_text})
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot([math.exp(loss) for loss in val_losses], label='Validation Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_curves_with_accuracy.png')
    with open('training_history.json', 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'final_perplexity': perplexity,
            'generated_texts': generated_texts
        }, f, indent=2)

if __name__ == "__main__":
    main()