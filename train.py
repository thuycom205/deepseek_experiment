import torch
from torch.utils.data import DataLoader, Dataset
import sys
from inference.model import Transformer, ModelArgs

from datasets import load_dataset  # Using Hugging Face Datasets

sys.path.append('/content/deepseek_experiment')


# ---- SimpleTokenizer Class (updated) ----
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
        tokens = text.split()[:max_length]
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens] + [0]*(max_length - len(tokens))

    def decode(self, token_ids):
        return " ".join(self.reverse_vocab.get(id, self.unk_token) for id in token_ids)

# ---- Dataset Class (no change) ----
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.tokenizer.encode(self.texts[idx], self.max_seq_len), dtype=torch.long)

# ---- New Data Loading with Hugging Face ----
def load_wikitext2(max_seq_len=128):
    # Load dataset from Hugging Face
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    texts = [text for text in dataset["train"]["text"] if text.strip()]
    
     
    # Take first 200 non-empty texts
    # texts = [text for text in dataset["train"]["text"] if text.strip()][:200]  # <-- Changed here
    
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.add_tokens(texts)
    
    # Create dataloader
    dataset = TextDataset(texts, tokenizer, max_seq_len=max_seq_len)
    return DataLoader(dataset, batch_size=32, shuffle=True), tokenizer

# Rest of the code remains the same as previous version...
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_items = 0
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(batch[:, :-1])
        targets = batch[:, 1:].contiguous().view(-1)
        
        loss = criterion(logits.view(-1, logits.size(-1)), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()
        total_items += 1
        
        # Print every 100 batches
        if batch_idx % 100 == 0:
            avg_loss = total_loss / total_items
            print(f"Batch {batch_idx}: Loss {avg_loss:.4f}")
            total_loss = 0
            total_items = 0

    
    # Final epoch loss
    epoch_loss = total_loss / len(dataloader)
    print(f"\nEpoch Average Loss: {epoch_loss:.4f}\n") 
    
    # for batch in dataloader:
    #     batch = batch.to(device)
    #     optimizer.zero_grad()
        
    #     # Forward pass with full sequence
    #     logits = model(batch[:, :-1])  # Exclude last token for inputs
    #     targets = batch[:, 1:].contiguous().view(-1)  # Predict next tokens
        
    #     loss = criterion(logits.view(-1, logits.size(-1)), targets)
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #     optimizer.step()

    #     print(f"Loss: {loss.item()}")
# ---- Modified Main Function ----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load WikiText-2 instead of local file
    dataloader, tokenizer = load_wikitext2()

    # Rest of the original model setup
    args = ModelArgs(
        max_batch_size=1,
        max_seq_len=512,
        vocab_size=len(tokenizer.vocab),
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
        dtype="bf16" if dtype == torch.bfloat16 else "fp32",
    )

    model = Transformer(args).to(device).to(dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        train_one_epoch(model, dataloader, optimizer, criterion, device)
        torch.save(model.state_dict(), f'checkpoint_epoch{epoch}.pt')

if __name__ == "__main__":
    main()