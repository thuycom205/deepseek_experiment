import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# Import your Transformer and ModelArgs
# (Adjust the import paths as needed based on your project structure)
from inference.model import Transformer, ModelArgs

###################################
# Copy or import SimpleTokenizer  #
###################################
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1}
        self.reverse_vocab = {0: "[PAD]", 1: "[UNK]"}
        self.unk_token = "[UNK]"

    def add_tokens(self, texts):
        for text in texts:
            for token in text.split():
                if token not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[token] = idx
                    self.reverse_vocab[idx] = token

    def encode(self, text, max_length=128):
        text = text.lower().replace('\n', ' ').replace('\t', ' ')
        tokens = text.split()[:max_length]
        # Map tokens to IDs or [UNK]
        token_ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        # Pad up to max_length
        token_ids += [0]*(max_length - len(token_ids))
        return token_ids

    def decode(self, token_ids):
        return " ".join(self.reverse_vocab.get(tid, self.unk_token) for tid in token_ids)

##############################################################
# Optional: Recreate / load tokenizer the same way as train. #
# If you just want a quick test, you can re-add vocab        #
##############################################################

def build_or_load_tokenizer():
    # In training, you used wikitext2 to build the vocab.
    # For a quick demonstration, we might just do a minimal example
    # or replicate exactly how you built it in `load_wikitext2`:
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Filter and keep some texts
    texts = [text for text in dataset["train"]["text"] 
             if text.strip() and len(text.split()) > 3]
    
    # Create tokenizer & add tokens
    tokenizer = SimpleTokenizer()
    tokenizer.add_tokens(texts)  # fill tokenizer vocab
    
    return tokenizer

###########################################################
# Load model checkpoint and set to eval mode for inference
###########################################################
def load_model(checkpoint_path, tokenizer, device='cuda'):
    # Create the same ModelArgs you used for training
    args = ModelArgs(
        max_batch_size=1,
        max_seq_len=512,
        vocab_size=len(tokenizer.vocab),  # crucial to match the tokenizer vocab size
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
        dtype="bf16" if torch.cuda.is_available() else "fp32",
    )
    
    model = Transformer(args)
    # Load the trained weights (map_location ensures compatibility if CPU is used)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    return model
def load_checkpoint(checkpoint_path, map_location="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    state_dict = ckpt["model_state_dict"]
    metadata = ckpt["metadata"]
    return state_dict, metadata

def load_model_and_tokenizer(checkpoint_path, device="cuda"):
    # 1) Load the checkpoint dictionary
    state_dict, metadata = load_checkpoint(checkpoint_path, map_location=device)
    
    # 2) Extract ModelArgs
    saved_args = metadata["args"]
    # 3) Create the model using those exact arguments
    model = Transformer(saved_args).to(device)
    model.load_state_dict(state_dict)
    
    model = model.to(torch.bfloat16)  # Make sure weights are in bf16
    model.eval()
    
    # 4) Reconstruct the tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.vocab = metadata["vocab"]
    tokenizer.reverse_vocab = metadata["reverse_vocab"]
    
    return model, tokenizer
#####################################
# A simple greedy text-generation   #
#####################################
def generate_text(model, tokenizer, prompt, max_new_tokens=30, device='cuda'):
    """
    Generate text from a given prompt using greedy decoding.
    """
    # Encode prompt to IDs (batch_size=1)
    input_ids = tokenizer.encode(prompt, max_length=512)
    # Trim trailing PAD from the prompt if it’s short
    while len(input_ids) > 0 and input_ids[-1] == 0:
        input_ids.pop()
    # We want a running list so we can keep appending predicted tokens
    generated_ids = input_ids[:]
    
    # Move to tensor
    
    input_tensor = torch.tensor([generated_ids], dtype=torch.bfloat16, device=device)
    
    for _ in range(max_new_tokens):
        # Forward pass
        with torch.no_grad():
            # shape: (batch=1, seq_len, vocab_size)
            logits = model(input_tensor)
        
        # Grab the logits for the last token
        next_token_logits = logits[0, -1, :]  # (vocab_size,)
        
        # Greedy decode: pick argmax
        next_token_id = torch.argmax(next_token_logits).item()
        
        # Append next_token_id to generated sequence
        generated_ids.append(next_token_id)
        
        # Update input_tensor for next iteration
        input_tensor = torch.tensor([generated_ids], dtype=torch.bfloat16, device=device)
    
    # Decode the resulting token IDs to string
    output_text = tokenizer.decode(generated_ids)
    return output_text

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoint_epoch2.pt"
    
    # Load the model and tokenizer in one go
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device=device)
    
    # Provide a prompt
    prompt = "Once upon a time"
    
    # Generate text
    output = generate_text(model, tokenizer, prompt, max_new_tokens=20, device=device)
    
    print("PROMPT:", prompt)
    print("GENERATED:", output)

if __name__ == "__main__":
    main()
