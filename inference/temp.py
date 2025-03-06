def train_one_epoch(...):
    scaler = torch.cuda.amp.GradScaler()  # Add this
    
    for batch in dataloader:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(batch[:, :-1])
            loss = criterion(...)
        
        scaler.scale(loss).backward()  # Instead of loss.backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()