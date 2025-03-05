import torch
from torch.utils.data import DataLoader, Dataset
import sys
from torchtext.datasets import WikiText2

sys.path.append('/content/deepseek_experiment')

# ---- Modified Data Loading using WikiText-2 ----
def load_wikitext2(max_seq_len=128):
    # Load WikiText-2 dataset
    train_iter, val_iter, test_iter = WikiText2()
    
    # Combine all training data
    texts = []
    for line in train_iter:
        if line.strip():  # Skip empty lines
            texts.append(line.strip())
    
    # Initialize tokenizer and build vocabulary
    tokenizer = SimpleTokenizer()
    tokenizer.add_tokens(texts)
    
    # Create dataset
    dataset = TextDataset(texts, tokenizer, max_seq_len=max_seq_len)
    return DataLoader(dataset, batch_size=1, shuffle=True), tokenizer

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