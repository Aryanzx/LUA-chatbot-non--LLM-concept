"""
ENGRAM CHATBOT - GOOGLE COLAB TRAINING NOTEBOOK
================================================

This notebook trains an AI chatbot with:
✓ Engram Memory (conditional N-gram lookup)
✓ Mixture of Experts (MoE)
✓ Multi-branch Architecture
✓ Training on Hugging Face datasets

Instructions:
1. Upload this notebook to Google Colab
2. Runtime -> Change runtime type -> T4 GPU
3. Run all cells
4. Download the exported model
"""

# ============================================================================
# SETUP
# ============================================================================

# Install dependencies
!pip install -q datasets transformers torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
import json
import os
from tqdm.auto import tqdm

print("✓ Dependencies installed")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Model architecture
    'vocab_size': 5000,
    'd_model': 256,
    'num_layers': 4,
    'num_heads': 4,
    'num_branches': 4,
    
    # Engram memory
    'use_engram': True,
    'engram_layers': [1, 3],  # Apply at layers 1 and 3
    'hash_size_2gram': 5000,
    'hash_size_3gram': 20000,
    'compression_vocab_size': 1000,
    
    # Mixture of Experts
    'use_moe': True,
    'num_experts': 4,
    'top_k_experts': 2,
    
    # Training
    'batch_size': 8,
    'num_epochs': 2,
    'learning_rate': 3e-4,
    'max_seq_length': 256,
    
    # Dataset
    'dataset_name': 'daily_dialog',
    'dataset_split': 'train[:2000]',  # Small subset for quick training
}

print("\n" + "="*60)
print("CONFIGURATION")
print("="*60)
for key, value in CONFIG.items():
    print(f"{key:25s} = {value}")
print("="*60 + "\n")

# ============================================================================
# SIMPLIFIED MODEL (Optimized for Quick Training)
# ============================================================================

class SimpleEngramLayer(nn.Module):
    """Lightweight Engram memory for quick training"""
    
    def __init__(self, d_model, hash_size_2gram=5000, hash_size_3gram=20000):
        super().__init__()
        self.d_model = d_model
        
        # Embedding tables
        self.embed_2gram = nn.Embedding(hash_size_2gram, d_model)
        self.embed_3gram = nn.Embedding(hash_size_3gram, d_model)
        
        # Fusion layers
        self.project = nn.Linear(2 * d_model, d_model)
        self.gate = nn.Linear(d_model, 1)
        
        self.hash_2 = hash_size_2gram
        self.hash_3 = hash_size_3gram
    
    def hash_ngram(self, ids):
        """Simple hash function"""
        return (ids.sum(dim=-1) * 2654435761)
    
    def forward(self, hidden, input_ids):
        B, L, D = hidden.shape
        device = hidden.device
        
        # Create 2-grams and 3-grams
        padded = F.pad(input_ids, (2, 0), value=0)
        
        bigrams = torch.stack([padded[:, i:i+L] for i in range(2)], dim=-1)
        trigrams = torch.stack([padded[:, i:i+L] for i in range(3)], dim=-1)
        
        # Hash to indices
        idx_2 = self.hash_ngram(bigrams) % self.hash_2
        idx_3 = self.hash_ngram(trigrams) % self.hash_3
        
        # Retrieve and concatenate
        e2 = self.embed_2gram(idx_2)
        e3 = self.embed_3gram(idx_3)
        e = torch.cat([e2, e3], dim=-1)
        
        # Project
        e_proj = self.project(e)
        
        # Gating
        alpha = torch.sigmoid(self.gate(hidden))
        
        # Fuse
        return hidden + alpha * e_proj


class SimpleMoE(nn.Module):
    """Simplified Mixture of Experts"""
    
    def __init__(self, d_model, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.router = nn.Linear(d_model, num_experts)
        
        # Experts (simple FFNs)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        B, L, D = x.shape
        
        # Routing
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-K
        topk_probs, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        output = torch.zeros_like(x)
        
        for i in range(self.num_experts):
            expert_mask = (topk_idx == i).any(dim=-1)
            if expert_mask.sum() == 0:
                continue
            
            # Process tokens for this expert
            expert_input = x[expert_mask]
            expert_output = self.experts[i](expert_input)
            
            # Weight and add
            weights = torch.zeros(expert_mask.sum(), device=x.device)
            for k in range(self.top_k):
                mask = topk_idx[expert_mask, :, k] == i
                weights[mask.any(dim=1)] = topk_probs[expert_mask, :, k][mask.any(dim=1)].mean(dim=1)
            
            output[expert_mask] += weights.unsqueeze(-1) * expert_output
        
        return output


class TransformerBlock(nn.Module):
    """Simplified transformer block with optional Engram and MoE"""
    
    def __init__(self, d_model, num_heads, use_engram=False, use_moe=False, **kwargs):
        super().__init__()
        
        # Attention
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Engram (optional)
        self.use_engram = use_engram
        if use_engram:
            self.engram = SimpleEngramLayer(d_model, **kwargs)
        
        # FFN or MoE
        self.use_moe = use_moe
        if use_moe:
            self.ffn = SimpleMoE(d_model, **kwargs)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, input_ids=None):
        # Attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Engram
        if self.use_engram and input_ids is not None:
            x = self.engram(x, input_ids)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class SimpleEngramChatbot(nn.Module):
    """Simplified chatbot for quick training"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embed = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_embed = nn.Embedding(config['max_seq_length'], config['d_model'])
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config['d_model'],
                num_heads=config['num_heads'],
                use_engram=(i in config['engram_layers']),
                use_moe=config['use_moe'],
                num_experts=config['num_experts'],
                top_k=config['top_k_experts'],
                hash_size_2gram=config['hash_size_2gram'],
                hash_size_3gram=config['hash_size_3gram'],
            ) for i in range(config['num_layers'])
        ])
        
        # Output
        self.norm = nn.LayerNorm(config['d_model'])
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embed.weight
    
    def forward(self, input_ids):
        B, L = input_ids.shape
        
        # Embeddings
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        
        # Blocks
        for block in self.blocks:
            x = block(x, input_ids)
        
        # Output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits

# ============================================================================
# DATASET
# ============================================================================

class SimpleDataset(Dataset):
    def __init__(self, hf_dataset, vocab_size, max_length):
        self.data = []
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Simple char-level tokenization
        for item in hf_dataset:
            if 'dialog' in item:
                text = ' '.join(item['dialog'])
            elif 'text' in item:
                text = item['text']
            else:
                text = str(item)
            
            # Convert to IDs (simple char-level)
            ids = [ord(c) % vocab_size for c in text[:max_length]]
            ids += [0] * (max_length - len(ids))
            self.data.append(torch.tensor(ids[:max_length], dtype=torch.long))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# ============================================================================
# TRAINING
# ============================================================================

def train(model, dataloader, config, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    model.train()
    
    for epoch in range(config['num_epochs']):
        total_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch in progress:
            batch = batch.to(device)
            
            # Shift for next-token prediction
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            # Forward
            logits = model(inputs)
            
            # Loss
            loss = F.cross_entropy(
                logits.reshape(-1, config['vocab_size']),
                targets.reshape(-1),
                ignore_index=0
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} complete | Average Loss: {avg_loss:.4f}\n")
    
    return model

# ============================================================================
# EXPORT
# ============================================================================

def export_model(model, config, output_dir='lua_export'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(f'{output_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save weights
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        np_array = param.cpu().detach().numpy()
        np.save(f'{output_dir}/{name.replace(".", "_")}.npy', np_array)
    
    # Create download archive
    !zip -r lua_export.zip {output_dir}/
    
    print(f"\n✓ Model exported to {output_dir}/")
    print(f"✓ Config saved to {output_dir}/config.json")
    print(f"✓ Archive created: lua_export.zip")
    print(f"\n📥 Download lua_export.zip to use with Lua inference engine")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\n")

# Load dataset
print("Loading dataset...")
dataset = load_dataset(CONFIG['dataset_name'], split=CONFIG['dataset_split'])
print(f"✓ Loaded {len(dataset)} examples\n")

# Prepare data
train_dataset = SimpleDataset(dataset, CONFIG['vocab_size'], CONFIG['max_seq_length'])
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
print(f"✓ Dataset prepared: {len(train_dataset)} samples\n")

# Initialize model
print("Initializing model...")
model = SimpleEngramChatbot(CONFIG)
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created: {total_params:,} parameters\n")

# Train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device}\n")

trained_model = train(model, train_loader, CONFIG, device)

# Export
print("\nExporting model for Lua inference...")
export_model(trained_model, CONFIG)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Download lua_export.zip from the files panel")
print("2. Extract on your local machine")
print("3. Run: lua engram_chatbot.lua")
print("\n✓ All done!\n")
