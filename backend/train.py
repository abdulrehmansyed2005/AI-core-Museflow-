import torch
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path

# Sequence to Sequence Data Loader for training on melody and band pairs
class Seq2SeqDataset(Dataset):
    def __init__(self, prompt_len=128, target_len=384, max_per_genre=4000):
        self.data = []
        self.vocab_size = 0
        
        print("Loading Melody-to-Band pairs into memory...")
        token_dir = Path("tokens_seq2seq")
        
        # Group files by genre prefix and cap each at the maximum per genre count
        genres = ["classical", "lofi", "rock"]
        selected_files = []
        for genre in genres:
            genre_files = sorted(token_dir.glob(f"{genre}_*.json"))
            capped = genre_files[:max_per_genre]
            selected_files.extend(capped)
            print(f"  {genre.upper()}: {len(capped)}/{len(genre_files)} files selected")
        
        for file in selected_files:
            with open(file, 'r') as f:
                content = json.load(f)
                prompt = content['prompt']
                target = content['target']
                
                # Pad sequences with zeros if they are too short or truncate if they are too long
                prompt = (prompt + [0] * prompt_len)[:prompt_len]
                target = (target + [0] * target_len)[:target_len]
                
                # Combine the melody tokens and band tokens into one sequence sequence of one hundred twenty eight melody tokens plus three hundred eighty four genre and band tokens
                seq = prompt + target
                self.data.append(seq)
                
                # Calculate vocabulary size based on the highest token value in the data
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

# Transformer Brain defines the model architecture
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

# Training loop handles the model optimization with precision and scheduling features
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # Enable mixed precision only when a CUDA GPU is available
    print(f"🔥 Booting up the Translator on: {device.type.upper()}")
    if not use_amp:
        print("⚠️  No CUDA GPU detected — training on CPU (slower, no mixed precision)")

    dataset = Seq2SeqDataset(prompt_len=128, target_len=384)
    
    # Batch size fits within memory constraints for the sequence length
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = MuseFlowTransformer(vocab_size=dataset.vocab_size).to(device)
    
    # Using AdamW optimizer with weight decay for better transformer stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding zeros during loss calculation
    
    # Higher epoch count to process more data during training
    epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Mixed precision scaler reduces memory usage by using lower precision math
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    mode_str = "mixed precision" if use_amp else "float32 (CPU)"
    print(f"🚀 Starting Seq2Seq Training ({epochs} epochs, {mode_str})...")
    print(f"   Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                # Mixed precision forward pass for GPU acceleration
                with torch.amp.autocast('cuda'):
                    predictions = model(x)
                    loss = criterion(predictions.transpose(1, 2), y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass for CPU fallback
                predictions = model(x)
                loss = criterion(predictions.transpose(1, 2), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
    torch.save(model.state_dict(), "museflow_translator.pth")
    print("✅ Training Complete! Model saved as 'museflow_translator.pth'")

if __name__ == "__main__":
    train()