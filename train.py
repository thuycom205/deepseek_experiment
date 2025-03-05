import torch
from torch.utils.data import DataLoader, Dataset
import sys

sys.path.append('/content/deepseek_experiment')

from inference.model import Transformer, ModelArgs
from inference.kernel import act_quant, weight_dequant, fp8_gemm

# ---- SimpleTokenizer Class ----
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1}
        self.reverse_vocab = {0: "[PAD]", 1: "[UNK]"}

    def add_tokens(self, texts):
        for text in texts:
            for token in text.split():
                if token not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[token] = idx
                    self.reverse_vocab[idx] = token

    def encode(self, text, max_length=128):
        tokens = text.split()[:max_length]
        ids = [self.vocab.get(token, 1) for token in tokens]
        ids += [0] * (max_length - len(ids))  # Pad to max length
        return ids

    def decode(self, token_ids):
        return " ".join(self.reverse_vocab.get(id, "[UNK]") for id in token_ids)

# ---- Dataset Class ----
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.tokenizer.encode(self.texts[idx], self.max_seq_len), dtype=torch.long)

# ---- Data Loading ----
def load_data(file_path, max_seq_len=128):
    with open(file_path, 'r') as f:
        texts = [line.strip() for line in f.readlines()]

    tokenizer = SimpleTokenizer()
    tokenizer.add_tokens(texts)

    dataset = TextDataset(texts, tokenizer, max_seq_len=max_seq_len)
    return DataLoader(dataset, batch_size=1, shuffle=True), tokenizer

# ---- Training Loop ----
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    for batch in dataloader:
        batch = batch.cuda()
        optimizer.zero_grad()
        logits = model(batch)

        # Shift batch to predict next token (causal language modeling)
        loss = criterion(logits.view(-1, logits.size(-1)), batch.view(-1))
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")

# ---- Main ----
def main():
    dataloader, tokenizer = load_data('sample_data.txt')
    args = ModelArgs(
    max_batch_size=1,
    max_seq_len=512,
    vocab_size=len(tokenizer.vocab),
    dim=256,
    inter_dim=512,
    moe_inter_dim=128,
    n_layers=4,
    n_heads=4,
    n_routed_experts=4,
    n_shared_experts=1,
    n_activated_experts=2,
    dtype="bf16",  # <-- This was missing
)


    
    model = Transformer(args).cuda().to(torch.bfloat16)


    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        train_one_epoch(model, dataloader, optimizer, criterion)
        torch.save(model.state_dict(), f'checkpoint_epoch{epoch}.pt')

if __name__ == "__main__":
    main()
