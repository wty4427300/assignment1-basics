import os
import time
import json
import pickle
import torch
import numpy as np
from model import TransformerLM
from optimizer import AdamW
from nn_utils import cross_entropy, clip_gradients, get_lr_cosine_schedule
from tokenizer import Tokenizer
from bpe_training import train_bpe

# --- Configuration ---
# Data files
dataset_path = "tinystories_train.txt"
if not os.path.exists(dataset_path):
    dataset_path = os.path.join("tests", "fixtures", "tinystories_sample.txt")
    print(f"Warning: tinystories_train.txt not found. Using small sample: {dataset_path}")

vocab_file = "tokenizer_vocab.pkl"
bin_data_file = "dataset_tokens.npy"

# Model Hyperparameters
tokenizer_vocab_size = 5000
special_tokens = ["<|endoftext|>"]
d_model = 256
n_layers = 4
n_heads = 4
d_ff = 1024
context_length = 256
rope_theta = 10000.0

# Training Hyperparameters
batch_size = 32
max_iters = 2000
max_lr = 5e-4
min_lr = 5e-5
warmup_iters = 200
weight_decay = 0.1
grad_clip = 1.0
save_interval = 500
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(data, batch_size, context_length, device):
    # Select random starting indices
    max_idx = len(data) - context_length - 1
    if max_idx <= 0:
        raise ValueError("Dataset is too small for the given context length.")
    ix = torch.randint(0, max_idx, (batch_size,))
    
    # x: input tokens, y: target tokens (shifted by 1)
    x = torch.stack([torch.from_numpy(data[i : i + context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + context_length + 1].astype(np.int64)) for i in ix])
    
    return x.to(device), y.to(device)

def main():
    print(f"Using device: {device}")

    # 1. Tokenizer Setup
    if not os.path.exists(vocab_file):
        print(f"Training BPE tokenizer on {dataset_path}...")
        vocab, merges = train_bpe(dataset_path, tokenizer_vocab_size, special_tokens)
        with open(vocab_file, "wb") as f:
            pickle.dump({"vocab": vocab, "merges": merges}, f)
        tokenizer = Tokenizer(vocab, merges, special_tokens)
    else:
        print(f"Loading tokenizer from {vocab_file}...")
        with open(vocab_file, "rb") as f:
            data = pickle.load(f)
            tokenizer = Tokenizer(data["vocab"], data["merges"], special_tokens)

    # 2. Data Preprocessing
    if not os.path.exists(bin_data_file):
        print(f"Tokenizing dataset {dataset_path}...")
        with open(dataset_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        data_np = np.array(tokens, dtype=np.uint16)
        np.save(bin_data_file, data_np)
        print(f"Tokenized data saved to {bin_data_file}")
    else:
        data_np = np.load(bin_data_file)
    
    print(f"Dataset total tokens: {len(data_np)}")

    # 3. Model & Optimizer Initialization
    model = TransformerLM(
        vocab_size=tokenizer_vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )
    model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params:,} parameters.")

    optimizer = AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=weight_decay
    )

    # 4. Training Loop
    model.train()
    start_time = time.time()
    
    print("Starting training loop...")
    for it in range(max_iters):
        # 4.1 Learning Rate Schedule
        lr = get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 4.2 Forward Pass
        try:
            x, y = get_batch(data_np, batch_size, context_length, device)
        except ValueError as e:
            print(f"Error: {e}")
            break

        logits = model(x)
        # Reshape to (N, C) for cross_entropy
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # 4.3 Backward Pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 4.4 Gradient Clipping
        clip_gradients(model.parameters(), grad_clip)

        # 4.5 Optimizer Step
        optimizer.step()

        # 4.6 Logging
        if it % 10 == 0:
            dt = time.time() - start_time
            print(f"iter {it:4d}: loss {loss.item():.4f} | lr {lr:.2e} | {dt*1000:.2f}ms/it")
            start_time = time.time()

        # 4.7 Periodic Checkpoint
        if it > 0 and it % save_interval == 0:
            ckpt_path = f"checkpoint_it_{it}.pt"
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": it,
                "config": {
                    "tokenizer_vocab_size": tokenizer_vocab_size,
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "d_ff": d_ff,
                    "context_length": context_length,
                }
            }
            torch.save(checkpoint, ckpt_path)
            print(f"--- Saved checkpoint: {ckpt_path} ---")

    print("Training process completed.")

if __name__ == "__main__":
    main()
