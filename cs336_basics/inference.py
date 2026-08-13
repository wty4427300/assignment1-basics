import torch
import pickle
import os
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

def load_model(checkpoint_path, vocab_file, special_tokens):
    # 1. Load tokenizer
    with open(vocab_file, "rb") as f:
        data = pickle.load(f)
        tokenizer = Tokenizer(data["vocab"], data["merges"], special_tokens)
    
    # 2. Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    
    # 3. Initialize model
    model = TransformerLM(
        vocab_size=config["tokenizer_vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["n_layers"],
        num_heads=config["n_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0 # Default
    )
    
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer

def main():
    checkpoint_path = "checkpoint_it_500.pt" # Adjust based on your training
    vocab_file = "tokenizer_vocab.pkl"
    special_tokens = ["<|endoftext|>"]
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found. Please train the model first.")
        return

    model, tokenizer = load_model(checkpoint_path, vocab_file, special_tokens)
    
    prompt = "Once upon a time"
    in_indices = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    
    print(f"Prompt: {prompt}")
    print("Generating...")
    
    # Generate 50 tokens with top-p sampling
    generated_indices = model.generate(
        in_indices, 
        max_new_tokens=50, 
        temperature=0.8, 
        top_p=0.9
    )
    
    output_text = tokenizer.decode(generated_indices[0].tolist())
    print(f"\nGenerated text:\n{output_text}")

if __name__ == "__main__":
    main()
