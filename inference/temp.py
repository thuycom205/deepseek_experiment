# ---- Modified Data Loading with 200 samples ----
def load_wikitext2(max_seq_len=128):
    # Load dataset from Hugging Face
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Take first 200 non-empty texts
    texts = [text for text in dataset["train"]["text"] if text.strip()][:200]  # <-- Changed here
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.add_tokens(texts)
    
    # Create dataloader
    dataset = TextDataset(texts, tokenizer, max_seq_len=max_seq_len)
    return DataLoader(dataset, batch_size=1, shuffle=True), tokenizer