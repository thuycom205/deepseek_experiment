import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


from inference.model import Transformer, ModelArgs
from inference.kernel import act_quant, weight_dequant, fp8_gemm

import sys
sys.path.append('/content/deepseek_experiment')

def load_data(tokenizer, file_path, max_seq_len=128):
    with open(file_path, 'r') as f:
        texts = [line.strip() for line in f.readlines()]
    tokenized = [tokenizer.encode(t, max_length=max_seq_len, truncation=True, padding='max_length') for t in texts]
    tokenized.add_special_tokens({'pad_token': '[PAD]'})

    tensor_data = torch.tensor(tokenized, dtype=torch.long)
    return DataLoader(tensor_data, batch_size=4, shuffle=True)

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    for batch in dataloader:
        batch = batch.cuda()
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits.view(-1, logits.size(-1)), batch.view(-1))  # simple LM loss
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")

def main():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')  # or custom tokenizer
    dataloader = load_data(tokenizer, 'sample_data.txt')

    args = ModelArgs()
    model = Transformer(args).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(3):  # 3 epochs for demo
        train_one_epoch(model, dataloader, optimizer, criterion)
        torch.save(model.state_dict(), f'checkpoint_epoch{epoch}.pt')

if __name__ == "__main__":
    main()
