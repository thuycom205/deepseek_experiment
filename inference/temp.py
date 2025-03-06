import torch
import torch.nn.functional as F

def sample_next_token(logits, top_k=50, temperature=1.0):
    # 1) Divide logits by temperature
    logits = logits / temperature
    
    # 2) Filter to top_k tokens
    top_k = min(top_k, logits.size(-1))
    top_logits, top_indices = torch.topk(logits, top_k)
    
    # 3) Convert to probabilities
    probs = F.softmax(top_logits, dim=-1)
    
    # 4) Sample from the top_k
    next_token = torch.multinomial(probs, 1)
    
    # 5) Map back to the original vocab index
    next_token_id = top_indices[next_token]
    
    return next_token_id.item()

def generate_text(model, tokenizer, prompt, max_new_tokens=30, device='cuda',
                  top_k=50, temperature=1.0):
    # Encode as integer IDs
    input_ids = tokenizer.encode(prompt)
    # Trim trailing PAD
    while input_ids and input_ids[-1] == 0:
        input_ids.pop()

    generated_ids = input_ids[:]
    # Move to tensor (use `torch.long` for embedding indices)
    input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_tensor)  # (1, seq_len, vocab_size)
        next_token_logits = logits[0, -1, :]  # last token

        # Use sampling instead of greedy
        next_token_id = sample_next_token(next_token_logits, top_k, temperature)
        
        generated_ids.append(next_token_id)
        # update input_tensor
        input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)

    # Decode
    output_text = tokenizer.decode(generated_ids)
    return output_text
