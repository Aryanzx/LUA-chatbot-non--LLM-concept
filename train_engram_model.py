"""
Engram-Enhanced Chatbot Training Script
Trains on Hugging Face dataset with MoE, Engram memory, and exports for Lua inference
Optimized for Google Colab T4 GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
import json
import hashlib
from collections import Counter
from typing import Optional, Tuple, List
import math

# ==============================================================================
# ENGRAM MEMORY MODULE (Based on the paper)
# ==============================================================================

class EngramMemory(nn.Module):
    """
    Conditional Memory via Scalable Lookup
    Implements sparse N-gram memory with context-aware gating
    """
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        hash_size_2gram: int = 10000,
        hash_size_3gram: int = 50000,
        compression_vocab_size: int = 2000,
        num_branches: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.hash_size_2gram = hash_size_2gram
        self.hash_size_3gram = hash_size_3gram
        self.compression_vocab_size = compression_vocab_size
        self.num_branches = num_branches
        
        # Tokenizer compression: raw vocab -> compressed vocab
        self.vocab_projection = nn.Parameter(
            torch.randint(0, compression_vocab_size, (vocab_size,))
        )
        self.vocab_projection.requires_grad = False
        
        # Static N-gram embedding tables
        self.embedding_2gram = nn.Embedding(hash_size_2gram, d_model)
        self.embedding_3gram = nn.Embedding(hash_size_3gram, d_model)
        
        # Fusion layers
        self.W_e = nn.Linear(2 * d_model, d_model)  # Concat projection
        self.W_v = nn.Linear(d_model, d_model)  # Shared value projection
        
        # Branch-specific key projections (for multi-branch architecture)
        if num_branches > 1:
            self.W_k = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(num_branches)
            ])
        else:
            self.W_k = nn.Linear(d_model, d_model)
        
        # Lightweight refinement (1D conv)
        self.refine_conv = nn.Conv1d(d_model, d_model, kernel_size=3, 
                                     padding=1, groups=d_model)
        
        self.norm = nn.RMSNorm(d_model)
        
    def hash_ngram(self, ngram: torch.Tensor, hash_size: int) -> torch.Tensor:
        """Deterministic hashing of N-gram to embedding table index"""
        # Simple hash: sum of token IDs with prime modulo
        return (ngram.sum(dim=-1) * 2654435761) % hash_size
    
    def retrieve_ngrams(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Phase 1: Retrieval
        Extract and retrieve N-gram embeddings
        """
        batch_size, seq_len = input_ids.shape
        
        # Compress vocabulary
        compressed_ids = self.vocab_projection[input_ids]
        
        # Construct N-grams (pad beginning with zeros)
        padded = F.pad(compressed_ids, (2, 0), value=0)
        
        # Extract 2-grams and 3-grams
        bigrams = torch.stack([padded[:, i:i+seq_len] for i in range(2)], dim=-1)
        trigrams = torch.stack([padded[:, i:i+seq_len] for i in range(3)], dim=-1)
        
        # Hash to indices
        bigram_indices = self.hash_ngram(bigrams, self.hash_size_2gram)
        trigram_indices = self.hash_ngram(trigrams, self.hash_size_3gram)
        
        # Retrieve embeddings
        e_2gram = self.embedding_2gram(bigram_indices)
        e_3gram = self.embedding_3gram(trigram_indices)
        
        # Concatenate
        e_concat = torch.cat([e_2gram, e_3gram], dim=-1)
        
        # Project to d_model
        e_t = self.W_e(e_concat)
        
        return e_t
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        branch_idx: int = 0
    ) -> torch.Tensor:
        """
        Phase 2: Fusion
        Modulate retrieved memory with context-aware gating
        """
        # Retrieve N-gram embeddings
        e_t = self.retrieve_ngrams(input_ids)
        
        # Normalize
        h_norm = self.norm(hidden_states)
        e_norm = self.norm(e_t)
        
        # Select appropriate key projection
        if self.num_branches > 1:
            W_k = self.W_k[branch_idx]
        else:
            W_k = self.W_k
        
        # Context-aware gating (equation 6 from paper)
        k_e = W_k(e_norm)
        alpha = torch.sigmoid(
            (h_norm * k_e).sum(dim=-1, keepdim=True) / math.sqrt(self.d_model)
        )
        
        # Modulation
        v_e = self.W_v(e_t)
        u_t = alpha * v_e
        
        # Lightweight refinement via 1D convolution
        u_t = u_t.transpose(1, 2)  # [B, D, L]
        u_t = self.refine_conv(u_t)
        u_t = u_t.transpose(1, 2)  # [B, L, D]
        
        # Residual integration
        return hidden_states + u_t


# ==============================================================================
# MIXTURE OF EXPERTS (MoE) LAYER
# ==============================================================================

class MoELayer(nn.Module):
    """
    Sparse Mixture of Experts with Top-K routing
    """
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        expert_hidden_dim: int = None,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_hidden_dim = expert_hidden_dim or 4 * d_model
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # Router network
        self.router = nn.Linear(d_model, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, self.expert_hidden_dim),
                nn.GELU(),
                nn.Linear(self.expert_hidden_dim, d_model)
            ) for _ in range(num_experts)
        ])
        
        # Load balancing
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        
        # Flatten for routing
        x_flat = x.view(-1, d_model)
        
        # Route tokens to experts
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-K selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert
        for i in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == i).any(dim=-1)
            if expert_mask.sum() == 0:
                continue
            
            # Get tokens and their weights
            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            expert_input = x_flat[token_indices]
            
            # Get routing weights for this expert
            weights = torch.zeros(len(token_indices), device=x.device)
            for k_idx in range(self.top_k):
                mask = top_k_indices[token_indices, k_idx] == i
                weights[mask] = top_k_probs[token_indices, k_idx][mask]
            
            # Compute expert output
            expert_output = self.experts[i](expert_input)
            
            # Weighted contribution
            output[token_indices] += weights.unsqueeze(-1) * expert_output
            
            # Track expert usage
            self.expert_counts[i] += len(token_indices)
        
        # Reshape output
        output = output.view(batch_size, seq_len, d_model)
        
        # Load balancing loss
        router_probs_mean = router_probs.mean(dim=0)
        load_balance_loss = self.num_experts * (router_probs_mean * router_probs_mean.log()).sum()
        
        return output, load_balance_loss


# ==============================================================================
# MULTI-BRANCH TRANSFORMER BLOCK
# ==============================================================================

class MultiBranchTransformerBlock(nn.Module):
    """
    Transformer block with multi-branch architecture
    Supports Manifold-Constrained Hyper-Connections (M branches)
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_branches: int = 4,
        use_moe: bool = True,
        use_engram: bool = False,
        **engram_kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.num_branches = num_branches
        
        # Attention
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attn_norm = nn.RMSNorm(d_model)
        
        # MoE or standard FFN
        if use_moe:
            self.ffn = MoELayer(d_model, **engram_kwargs)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )
        self.ffn_norm = nn.RMSNorm(d_model)
        
        # Engram memory (optional)
        self.use_engram = use_engram
        if use_engram:
            self.engram = EngramMemory(d_model, num_branches=num_branches, **engram_kwargs)
        
        # Multi-branch connections (if num_branches > 1)
        if num_branches > 1:
            self.branch_weights = nn.Parameter(torch.ones(num_branches) / num_branches)
    
    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        load_balance_loss = None
        
        # Self-attention
        residual = x
        x = self.attn_norm(x)
        attn_out, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = residual + attn_out
        
        # Engram memory (if enabled)
        if self.use_engram and input_ids is not None:
            if self.num_branches > 1:
                # Multi-branch fusion
                engram_outs = []
                for branch_idx in range(self.num_branches):
                    engram_out = self.engram(x, input_ids, branch_idx)
                    engram_outs.append(engram_out)
                
                # Weighted combination
                x = sum(w * out for w, out in zip(self.branch_weights, engram_outs))
            else:
                x = self.engram(x, input_ids)
        
        # FFN (with MoE)
        residual = x
        x = self.ffn_norm(x)
        
        if isinstance(self.ffn, MoELayer):
            ffn_out, load_balance_loss = self.ffn(x)
        else:
            ffn_out = self.ffn(x)
        
        x = residual + ffn_out
        
        return x, load_balance_loss


# ==============================================================================
# MAIN TRANSFORMER MODEL
# ==============================================================================

class EngramChatbot(nn.Module):
    """
    Transformer-based chatbot with Engram memory and MoE
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_branches: int = 4,
        use_moe: bool = True,
        engram_layers: List[int] = [2, 4],  # Apply Engram at specific layers
        **engram_kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Standard embedding (unchanged)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(2048, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            MultiBranchTransformerBlock(
                d_model,
                num_heads,
                num_branches=num_branches,
                use_moe=use_moe,
                use_engram=(i in engram_layers),
                **engram_kwargs
            ) for i in range(num_layers)
        ])
        
        # Output (unchanged)
        self.norm = nn.RMSNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.output_projection.weight = self.token_embedding.weight
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Process through blocks
        total_load_balance_loss = 0
        for block in self.blocks:
            x, lb_loss = block(x, input_ids, attention_mask)
            if lb_loss is not None:
                total_load_balance_loss += lb_loss
        
        # Output
        x = self.norm(x)
        logits = self.output_projection(x)
        
        return logits, total_load_balance_loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Generate text using top-k and top-p sampling"""
        
        self.eval()
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS
                if next_token.item() == 0:  # Assuming 0 is EOS
                    break
        
        return generated


# ==============================================================================
# DATASET PREPARATION
# ==============================================================================

class ChatDataset(Dataset):
    """Prepare Hugging Face dataset for training"""
    
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as conversational data
        if 'text' in item:
            text = item['text']
        elif 'conversation' in item:
            text = self.format_conversation(item['conversation'])
        else:
            text = str(item)
        
        # Tokenize
        tokens = self.tokenizer.encode(text, max_length=self.max_length, truncation=True)
        
        # Pad
        input_ids = tokens + [0] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids[:self.max_length], dtype=torch.long)
        
        return input_ids
    
    def format_conversation(self, conversation):
        """Format conversation turns"""
        formatted = []
        for turn in conversation:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    device: str = 'cuda',
    save_path: str = 'engram_chatbot.pt'
):
    """Training loop with Engram memory and MoE"""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(train_loader))
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_lb_loss = 0
        
        for batch_idx, input_ids in enumerate(train_loader):
            input_ids = input_ids.to(device)
            
            # Shift for next-token prediction
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            # Forward pass
            logits, lb_loss = model(inputs)
            
            # Language modeling loss
            lm_loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                targets.reshape(-1),
                ignore_index=0
            )
            
            # Total loss (with load balancing)
            loss = lm_loss + 0.01 * lb_loss if lb_loss > 0 else lm_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += lm_loss.item()
            total_lb_loss += lb_loss if isinstance(lb_loss, float) else lb_loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | "
                      f"LM Loss: {lm_loss.item():.4f} | LB Loss: {lb_loss:.4f}")
        
        avg_loss = total_loss / len(train_loader)
        avg_lb_loss = total_lb_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Complete | Avg LM Loss: {avg_loss:.4f} | Avg LB Loss: {avg_lb_loss:.4f}\n")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': model.vocab_size,
        'd_model': model.d_model,
    }, save_path)
    
    print(f"Model saved to {save_path}")


# ==============================================================================
# EXPORT FOR LUA INFERENCE
# ==============================================================================

def export_for_lua(model: nn.Module, tokenizer, save_dir: str = 'lua_export'):
    """Export model weights and config for Lua inference"""
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Export configuration
    config = {
        'vocab_size': model.vocab_size,
        'd_model': model.d_model,
        'num_layers': len(model.blocks),
        'num_heads': 8,  # Adjust as needed
    }
    
    with open(f'{save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Export all weights as numpy arrays
    state_dict = model.state_dict()
    
    for name, param in state_dict.items():
        # Convert to numpy and save
        np_param = param.cpu().detach().numpy()
        np.save(f'{save_dir}/{name.replace(".", "_")}.npy', np_param)
    
    print(f"Model exported to {save_dir}/")
    print(f"Config saved to {save_dir}/config.json")
    print(f"Weights saved as .npy files")


# ==============================================================================
# MAIN EXECUTION (FOR GOOGLE COLAB)
# ==============================================================================

if __name__ == "__main__":
    
    # Simple tokenizer (replace with proper tokenizer)
    class SimpleTokenizer:
        def __init__(self, vocab_size=10000):
            self.vocab_size = vocab_size
            self.char_to_id = {}
            self.id_to_char = {}
        
        def encode(self, text, max_length=512, truncation=True):
            # Simple character-level tokenization
            ids = [ord(c) % self.vocab_size for c in text]
            if truncation:
                ids = ids[:max_length]
            return ids
        
        def decode(self, ids):
            return ''.join(chr(i) for i in ids if i > 0)
    
    # Configuration
    VOCAB_SIZE = 10000
    D_MODEL = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    BATCH_SIZE = 16
    NUM_EPOCHS = 3
    
    # Load dataset from Hugging Face
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("daily_dialog", split="train[:1000]")  # Small subset for demo
    
    # Initialize tokenizer and dataset
    tokenizer = SimpleTokenizer(VOCAB_SIZE)
    train_dataset = ChatDataset(dataset, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model with Engram and MoE
    print("Initializing model with Engram memory and MoE...")
    model = EngramChatbot(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_branches=4,
        use_moe=True,
        engram_layers=[2, 4],  # Apply Engram at layers 2 and 4
        num_experts=8,
        top_k=2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nStarting training...")
    train_model(
        model,
        train_loader,
        num_epochs=NUM_EPOCHS,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Export for Lua
    print("\nExporting for Lua inference...")
    export_for_lua(model, tokenizer)
    
    print("\n✓ Training complete!")
    print("✓ Model exported for Lua inference")
