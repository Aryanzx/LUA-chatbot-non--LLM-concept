# Engram Chatbot: AI with Conditional Memory & Markov Thinking

A from-scratch implementation of an advanced chatbot combining:
- **Engram Memory**: Sparse N-gram lookup for conditional memory (based on research paper)
- **Mixture of Experts (MoE)**: Efficient sparse computation
- **Markov Chain Thinking**: Internal lookahead for better predictions
- **Top-K Sampling**: Controlled generation quality

## 🏗️ Architecture

### Overall Design

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING (Python/PyTorch)               │
│                     Google Colab T4 GPU                      │
├─────────────────────────────────────────────────────────────┤
│  1. Load Hugging Face Dataset                               │
│  2. Train Transformer + Engram + MoE                        │
│  3. Export weights as .npy files                            │
│  4. Save config.json                                        │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ Export
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE (Lua)                           │
│                   CPU-optimized                              │
├─────────────────────────────────────────────────────────────┤
│  1. Load exported weights                                   │
│  2. Run inference with Engram memory                        │
│  3. Use Markov chain for internal thinking                  │
│  4. Generate with Top-K sampling                            │
└─────────────────────────────────────────────────────────────┘
```

### Engram Memory Module

Based on the paper: "Conditional Memory via Scalable Lookup"

**Two-Phase Operation:**

#### Phase 1: Retrieval (Static Memory Lookup)
```
Input: Token sequence [x₁, x₂, ..., xₜ]

1. Tokenizer Compression
   ├─ Map raw vocab → compressed vocab
   └─ Increase semantic density

2. N-gram Construction
   ├─ Extract 2-grams: (xₜ₋₁, xₜ)
   └─ Extract 3-grams: (xₜ₋₂, xₜ₋₁, xₜ)

3. Deterministic Hashing
   ├─ Hash each N-gram → embedding index
   └─ No learned retrieval (pure lookup)

4. Static Embedding Retrieval
   ├─ E₂[hash(2-gram)] → e₂ ∈ ℝᵈ
   └─ E₃[hash(3-gram)] → e₃ ∈ ℝᵈ

Output: Concatenated embeddings [e₂; e₃]
```

#### Phase 2: Fusion (Dynamic Modulation)
```
Input: Retrieved memory eₜ, Hidden state hₜ

1. Linear Projection
   êₜ = Wₑ · [e₂; e₃]

2. Context-Aware Gating
   αₜ = σ(RMSNorm(hₜ)ᵀ · RMSNorm(Wₖ·êₜ) / √d)
   
3. Modulation
   uₜ = αₜ · (Wᵥ · êₜ)

4. Lightweight Refinement
   uₜ' = Conv1D(uₜ)  // Local smoothing

5. Residual Integration
   hₜ' = hₜ + uₜ'

Output: Enhanced hidden state hₜ'
```

**Key Properties:**
- ✅ **Sparse**: Only retrieves relevant N-grams
- ✅ **Deterministic**: No attention over memory (faster)
- ✅ **Scalable**: Constant time lookup regardless of memory size
- ✅ **Conditional**: Context-aware gating adapts to input

### Mixture of Experts (MoE)

```
Input: Hidden state h ∈ ℝᵈ

1. Routing
   router_logits = Router(h)
   probs = softmax(router_logits)
   
2. Top-K Selection
   selected_experts, weights = topk(probs, k=2)

3. Expert Computation
   outputs = [Expert_i(h) for i in selected_experts]
   
4. Weighted Combination
   output = Σ(weights[i] * outputs[i])

Output: Expert-mixed representation
```

**Benefits:**
- 🚀 **Efficient**: Only 2 out of 8 experts active per token
- 🎯 **Specialized**: Each expert learns different patterns
- ⚖️ **Load Balanced**: Auxiliary loss ensures even expert usage

### Markov Chain Internal Thinking

```
Context: Last N tokens [xₜ₋ₙ, ..., xₜ]

1. Build Transition Table
   P(xₜ₊₁ | context) from training data

2. Multi-Step Lookahead
   For depth = 1 to 3:
     ├─ Predict: x̂ₜ₊depth using Markov chain
     └─ Calculate: confidence = P(x̂ₜ₊depth | context)

3. Confidence Scoring
   overall_confidence = Σ(prob_i / depth_i) / total_depth

4. Probability Adjustment
   If confidence > 0.3:
     transformer_probs[markov_prediction] *= (1 + confidence)

Output: Enhanced probability distribution
```

**Purpose:**
- 🧠 **Lookahead**: Internal "thinking" about likely continuations
- 📊 **Confidence Boosting**: Reinforces consistent patterns
- 🔄 **Complementary**: Combines statistical patterns with neural predictions

### Multi-Branch Architecture

For advanced models, multiple parallel branches process information:

```
                    Input
                      |
        ┌─────────────┼─────────────┐
        │             │             │
    Branch 1      Branch 2      Branch 3
        │             │             │
   (W_k⁽¹⁾)       (W_k⁽²⁾)       (W_k⁽³⁾)
        │             │             │
    Gate α₁       Gate α₂       Gate α₃
        │             │             │
        └─────────────┴─────────────┘
                      |
              Weighted Combination
                      |
                   Output
```

**Shared**: Embedding tables E₂, E₃ and value projection Wᵥ
**Branch-specific**: Key projections Wₖ⁽ᵐ⁾ for independent gating

## 📦 Installation

### Python (Training)

```bash
pip install torch datasets transformers numpy
```

### Lua (Inference)

```bash
# Install Lua 5.4
sudo apt-get install lua5.4

# Install JSON library
luarocks install json
```

## 🚀 Quick Start

### Step 1: Train the Model (Google Colab)

```python
# In Google Colab
!git clone <your-repo>
%cd <your-repo>

# Run training script
%run train_engram_model.py

# This will:
# 1. Load dataset from Hugging Face
# 2. Train model with Engram + MoE
# 3. Export to lua_export/ directory
```

### Step 2: Download Exported Weights

```python
# In Colab
from google.colab import files
import shutil

# Compress exports
shutil.make_archive('lua_export', 'zip', 'lua_export')

# Download
files.download('lua_export.zip')
```

### Step 3: Run Lua Inference

```bash
# Extract downloaded weights
unzip lua_export.zip

# Run chatbot
lua engram_chatbot.lua
```

## 💡 Usage Examples

### Training with Custom Dataset

```python
from datasets import load_dataset
from train_engram_model import EngramChatbot, train_model

# Load your dataset
dataset = load_dataset("your-dataset-name", split="train")

# Initialize model
model = EngramChatbot(
    vocab_size=10000,
    d_model=512,
    num_layers=6,
    num_heads=8,
    num_branches=4,
    use_moe=True,
    engram_layers=[2, 4],  # Apply Engram at layers 2 and 4
    num_experts=8,
    top_k=2
)

# Train
train_model(model, train_loader, num_epochs=5)
```

### Interactive Inference

```lua
-- In Lua
local EngramChatbot = require("engram_chatbot")

-- Load model
local model = EngramChatbot:new(
    "lua_export/config.json",
    "lua_export/"
)

-- Generate
local response = model:generate(
    "Hello, how are you?",
    100,   -- max_length
    0.8,   -- temperature
    50     -- top_k
)

print(response)
```

### Adjusting Generation Parameters

```lua
-- More creative (higher temperature)
model:generate(prompt, 100, 1.2, 50)

-- More focused (lower temperature, smaller top-k)
model:generate(prompt, 100, 0.5, 20)

-- Longer responses
model:generate(prompt, 500, 0.8, 50)
```

## 🔧 Configuration

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 10000 | Vocabulary size |
| `d_model` | 512 | Model dimension |
| `num_layers` | 6 | Number of transformer layers |
| `num_heads` | 8 | Attention heads per layer |
| `num_branches` | 4 | Parallel branches (multi-branch arch) |
| `num_experts` | 8 | Total MoE experts |
| `top_k` | 2 | Active experts per token |
| `engram_layers` | [2, 4] | Layers with Engram memory |

### Engram Memory Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hash_size_2gram` | 10000 | Size of 2-gram embedding table |
| `hash_size_3gram` | 50000 | Size of 3-gram embedding table |
| `compression_vocab_size` | 2000 | Compressed vocabulary size |

### Generation Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_length` | 100 | Maximum tokens to generate |
| `temperature` | 0.8 | Sampling temperature (higher = more random) |
| `top_k` | 50 | Top-K sampling parameter |
| `top_p` | 0.9 | Nucleus sampling threshold |

## 🎯 Key Features Explained

### 1. Engram Memory: Why It Matters

Traditional transformers process each token independently through attention. Engram adds **conditional memory**:

- **Static Pattern Storage**: N-grams capture frequent patterns without recomputation
- **Context-Aware Retrieval**: Gating decides when to use stored patterns
- **Efficiency**: O(1) lookup vs O(n²) attention over memory

**Example:**
```
Input: "The cat sat on the"
2-gram retrieval: ("the", "_") → stored completion patterns
3-gram retrieval: ("on", "the", "_") → more specific context
Gating: Current context decides how much to trust these patterns
```

### 2. Markov Chain Thinking: Internal Lookahead

Before generating each token, the model "thinks ahead" using Markov chains:

```
Current: "I will go to the"
Markov lookahead:
  Depth 1: "store" (p=0.4)
  Depth 2: "to" (p=0.3)
  Depth 3: "buy" (p=0.2)
Confidence: 0.3 → Boost "store" probability
```

This combines:
- **Neural predictions**: From transformer
- **Statistical patterns**: From Markov chain
- **Confidence weighting**: Only boost when Markov is confident

### 3. Top-K Sampling: Quality Control

Instead of always picking the most likely token (greedy) or sampling uniformly (random):

```
Logits: [0.3, 0.25, 0.2, 0.15, 0.05, 0.03, 0.02]
Top-K (k=3): [0.3, 0.25, 0.2, 0, 0, 0, 0]
Renormalize: [0.4, 0.33, 0.27, 0, 0, 0, 0]
Sample: Higher quality, controlled randomness
```

## 📊 Performance Tips

### GPU Training (Python)

```python
# Use mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits, _ = model(inputs)
    loss = F.cross_entropy(logits, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### CPU Inference (Lua)

```lua
-- Batch processing for efficiency
local function process_batch(prompts)
    local responses = {}
    for _, prompt in ipairs(prompts) do
        table.insert(responses, model:generate(prompt, 100, 0.8, 50))
    end
    return responses
end
```

### Memory Optimization

1. **Reduce hash table sizes** for lower memory:
   ```python
   hash_size_2gram=5000,  # Instead of 10000
   hash_size_3gram=20000  # Instead of 50000
   ```

2. **Fewer experts** for faster inference:
   ```python
   num_experts=4,  # Instead of 8
   top_k=1         # Instead of 2
   ```

3. **Smaller model dimension**:
   ```python
   d_model=256,    # Instead of 512
   num_layers=4    # Instead of 6
   ```

## 🐛 Troubleshooting

### Issue: Out of memory during training

**Solution**: Reduce batch size or model size
```python
BATCH_SIZE = 8  # Instead of 16
d_model = 256   # Instead of 512
```

### Issue: Lua can't load .npy files

**Solution**: The current implementation uses simplified NPY loading. For production:
1. Use a proper NPY library like `npy4th`
2. Or export weights as binary files with custom format

### Issue: Generation is repetitive

**Solution**: Adjust temperature and top_k
```lua
-- Higher temperature for more variety
model:generate(prompt, 100, 1.2, 50)

-- Or use nucleus (top-p) sampling instead
```

### Issue: Slow inference on CPU

**Solution**:
1. Reduce model size
2. Use quantization (convert float32 to int8)
3. Implement batch processing
4. Consider using LuaJIT instead of standard Lua

## 📚 Research References

1. **Engram Memory**: "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"
2. **Mixture of Experts**: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (Shazeer et al., 2017)
3. **Multi-branch Architecture**: "Manifold-Constrained Hyper-Connections" (Xie et al., 2025)
4. **Top-K Sampling**: "Hierarchical Neural Story Generation" (Fan et al., 2018)

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Better NPY file handling in Lua
- [ ] Quantization for faster inference
- [ ] More sophisticated tokenization
- [ ] Streaming generation
- [ ] Multi-GPU training support
- [ ] ONNX export for cross-platform deployment

## 📄 License

MIT License - Feel free to use for research and commercial applications

## 🙏 Acknowledgments

- Anthropic for the Engram architecture research
- Hugging Face for datasets and transformers library
- PyTorch team for the deep learning framework
- Lua community for the lightweight scripting language

---

**Built with ❤️ for AI research and experimentation**
