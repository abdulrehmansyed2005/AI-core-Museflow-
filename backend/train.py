import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path

# --- 1. THE NEW SEQ2SEQ DATA LOADER ---
class Seq2SeqDataset(Dataset):
    def __init__(self, prompt_len=128, target_len=384):
        self.data = []
        self.vocab_size = 0
        
        print("Loading Melody-to-Band pairs into memory...")
        token_files = list(Path("tokens_seq2seq").glob("*.json"))
        
        for file in token_files:
            with open(file, 'r') as f:
                content = json.load(f)
                prompt = content['prompt']
                target = content['target']
                
                # Pad sequences with 0s if they are too short, or chop them if they are too long
                prompt = (prompt + [0] * prompt_len)[:prompt_len]
                target = (target + [0] * target_len)[:target_len]
                
                # Stitch them together: [128 Melody Tokens] + [384 Genre & Band Tokens]
                seq = prompt + target
                self.data.append(seq)
                
                # Dynamically calculate vocabulary size (making sure to catch our 10001 genre tags)
                self.vocab_size = max(self.vocab_size, max(seq) + 1)
                
        print(f"Loaded {len(self.data)} Seq2Seq pairs. Vocabulary size: {self.vocab_size}")
        print(f"Sequence length: {prompt_len} (prompt) + {target_len} (target) = {prompt_len + target_len}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.data[idx], dtype=torch.long)
        x = sequence[:-1] 
        y = sequence[1:]  
        return x, y

# --- 2. THE TRANSFORMER BRAIN ---
class MuseFlowTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        embedded = self.embedding(x)
        out = self.transformer(embedded, mask=mask, is_causal=True)
        return self.fc_out(out)

# --- 3. THE TRAINING LOOP (Now with Mixed Precision & LR Scheduling) ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Booting up the Translator on: {device.type.upper()}")

    dataset = Seq2SeqDataset(prompt_len=128, target_len=384)
    
    # Batch size 4 with mixed precision fits in 6GB VRAM for 512-token sequences
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = MuseFlowTransformer(vocab_size=dataset.vocab_size).to(device)
    
    # Lower LR + cosine decay = much more stable training for Transformers
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding zeros
    
    # Cosine annealing scheduler — smoothly decays LR to near-zero
    epochs = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Mixed precision scaler — halves VRAM usage by using float16 where safe
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"🚀 Starting Seq2Seq Training ({epochs} epochs, mixed precision)...")
    print(f"   Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                predictions = model(x)
                loss = criterion(predictions.transpose(1, 2), y)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
    torch.save(model.state_dict(), "museflow_translator.pth")
    print("✅ Training Complete! Model saved as 'museflow_translator.pth'")

if __name__ == "__main__":
    train()