import torch
import torch.nn.functional as F
import sys

# Adjust these imports according to your actual file structure
from inference.model import Transformer, ModelArgs
from train import SimpleTokenizer  # <-- Replace 'train_script' with the actual filename that defines SimpleTokenizer

###############################################################################
# 1) Load Model + Tokenizer
###############################################################################
def load_model_and_tokenizer(checkpoint_path, vocab_size, device, dtype_str="bf16"):
    """
    Loads a Transformer model from a saved checkpoint, 
    and creates a tokenizer with the same vocabulary size.
    """
    # You likely used the same ModelArgs in training; replicate them here
    args = ModelArgs(
        max_batch_size=1,
        max_seq_len=512,
        vocab_size=vocab_size,
        dim=256,
        inter_dim=512,
        moe_inter_dim=128,
        n_layers=4,
        n_heads=4,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=64,
        v_head_dim=80,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        dtype=dtype_str,  # "bf16" or "fp32" depending on your checkpoint
    )

    # Instantiate the model
    model = Transformer(args)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # Move to device
    if dtype_str == "bf16" and torch.cuda.is_available():
        model = model.to(device).to(torch.bfloat16)
    else:
        model = model.to(device).to(torch.float32)
    
    model.eval()
    return model


def load_tokenizer(token_vocab_path=None):
    """
    Creates a SimpleTokenizer. 
    Optionally, if you saved the tokenizer's vocabulary somewhere, load it here.
    
    If you didn't explicitly save it, you'll need to ensure you replicate the 
    same add_tokens(...) process as training. 
    For demonstration, we'll assume we just create a blank one 
    with the same vocab size that was used in training.
    """
    tokenizer = SimpleTokenizer()
    # If you have a saved tokenizer vocab, load it here,
    # or replicate the exact code that was used to build it during training.
    
    # Example: 
    #   with open("tokenizer_vocab.json", "r") as f:
    #       vocab_dict = json.load(f)
    #   tokenizer.vocab = vocab_dict["vocab"]
    #   tokenizer.reverse_vocab = vocab_dict["reverse_vocab"]
    #
    # But for now, we just return a new one. Make sure
    # your 'tokenizer.vocab' matches the size you used in ModelArgs above.
    return tokenizer


###############################################################################
# 2) Simple text generation
###############################################################################
def generate_text_greedy(model, tokenizer, prompt, device, max_new_tokens=20):
    """
    Generates text using a simple greedy approach.
    Appends one token at a time, repeatedly feeding the model.
    """
    # Convert prompt to token IDs
    input_ids = tokenizer.encode(prompt, max_length=128)  # or your choice
    # Chop off trailing PAD tokens if you want
    while input_ids and input_ids[-1] == 0:
        input_ids.pop()

    # Move to device
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    model.eval()
    generated = prompt  # We'll hold the text here

    # Generate new tokens
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # Forward pass
            logits = model(input_ids)  
            # logits shape: [batch_size=1, seq_len, vocab_size]
        
        # Get the logits for the last position
        next_logit = logits[0, -1, :]  # shape: (vocab_size,)
        
        # Greedy: pick argmax
        next_token_id = torch.argmax(next_logit).item()
        
        # Append token ID to current sequence
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
        
        # Convert the new token to text
        next_token_str = tokenizer.decode([next_token_id])
        generated += " " + next_token_str
        
        # Optionally, stop if we hit a special token or PAD, etc.
        if next_token_id == tokenizer.vocab.get("[PAD]", None):
            break

    return generated


###############################################################################
# 3) Main demonstration
###############################################################################
def main():
    # Parse optional args from command line if you want
    # e.g., python generate_text.py "My prompt"
    if len(sys.argv) > 1:
        user_prompt = sys.argv[1]
    else:
        user_prompt = "Hello world"

    # Suppose you saved the model checkpoint as 'checkpoint_epoch2.pt'
    checkpoint_path = "checkpoint_epoch2.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The vocab_size must match what you had during training
    vocab_size = 50000  # e.g. or len(tokenizer.vocab) if known

    # 1) Load the Model + Tokenizer
    model = load_model_and_tokenizer(
        checkpoint_path=checkpoint_path, 
        vocab_size=vocab_size, 
        device=device,
        dtype_str="bf16", # or "fp32"
    )
    tokenizer = load_tokenizer()

    # 2) Generate text from prompt
    generated_text = generate_text_greedy(
        model, tokenizer, user_prompt, device, max_new_tokens=30
    )

    print("PROMPT:", user_prompt)
    print("GENERATED TEXT:", generated_text)


if __name__ == "__main__":
    main()
