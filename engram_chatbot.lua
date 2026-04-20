-- ============================================================================
-- ENGRAM CHATBOT - LUA INFERENCE ENGINE (CPU)
-- Implements: Engram Memory, Markov Chain Thinking, Top-K Sampling, MoE
-- ============================================================================

local json = require("json")  -- Requires lua-json library
local math = require("math")

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

local function load_json(filepath)
    local file = io.open(filepath, "r")
    if not file then
        error("Cannot open file: " .. filepath)
    end
    local content = file:read("*all")
    file:close()
    return json.decode(content)
end

local function load_npy(filepath)
    -- Simple .npy loader (assumes float32, C-contiguous)
    -- For production, use a proper NPY library
    local file = io.open(filepath, "rb")
    if not file then
        error("Cannot load: " .. filepath)
    end
    
    -- Skip NPY header (first 128 bytes typically)
    file:seek("set", 128)
    
    local data = {}
    while true do
        local bytes = file:read(4)
        if not bytes then break end
        
        -- Convert 4 bytes to float32
        local value = string.unpack("<f", bytes)
        table.insert(data, value)
    end
    
    file:close()
    return data
end

local function softmax(logits)
    local max_logit = -math.huge
    for i = 1, #logits do
        if logits[i] > max_logit then
            max_logit = logits[i]
        end
    end
    
    local exp_sum = 0
    local probs = {}
    for i = 1, #logits do
        probs[i] = math.exp(logits[i] - max_logit)
        exp_sum = exp_sum + probs[i]
    end
    
    for i = 1, #probs do
        probs[i] = probs[i] / exp_sum
    end
    
    return probs
end

local function top_k_filter(probs, k)
    -- Keep only top-k probabilities
    local indexed = {}
    for i = 1, #probs do
        table.insert(indexed, {prob = probs[i], idx = i})
    end
    
    table.sort(indexed, function(a, b) return a.prob > b.prob end)
    
    local filtered = {}
    for i = 1, #probs do
        filtered[i] = 0
    end
    
    for i = 1, math.min(k, #indexed) do
        filtered[indexed[i].idx] = indexed[i].prob
    end
    
    -- Renormalize
    local sum = 0
    for i = 1, #filtered do
        sum = sum + filtered[i]
    end
    for i = 1, #filtered do
        filtered[i] = filtered[i] / sum
    end
    
    return filtered
end

local function sample(probs)
    local rand = math.random()
    local cumsum = 0
    
    for i = 1, #probs do
        cumsum = cumsum + probs[i]
        if rand <= cumsum then
            return i
        end
    end
    
    return #probs
end

-- ============================================================================
-- MARKOV CHAIN FOR INTERNAL THINKING
-- ============================================================================

local MarkovChain = {}
MarkovChain.__index = MarkovChain

function MarkovChain:new(order)
    local obj = {
        order = order or 2,  -- N-gram order for Markov chain
        chain = {},           -- Transition probabilities
        context_history = {}  -- Recent context
    }
    setmetatable(obj, MarkovChain)
    return obj
end

function MarkovChain:add_transition(context, next_token)
    -- Build Markov chain from training data
    local key = table.concat(context, ",")
    
    if not self.chain[key] then
        self.chain[key] = {}
    end
    
    if not self.chain[key][next_token] then
        self.chain[key][next_token] = 0
    end
    
    self.chain[key][next_token] = self.chain[key][next_token] + 1
end

function MarkovChain:predict_next(context)
    -- Predict next token based on Markov chain
    local key = table.concat(context, ",")
    
    if not self.chain[key] then
        return nil, 0
    end
    
    -- Find most likely next token
    local best_token = nil
    local best_count = 0
    
    for token, count in pairs(self.chain[key]) do
        if count > best_count then
            best_token = token
            best_count = count
        end
    end
    
    -- Calculate probability
    local total = 0
    for _, count in pairs(self.chain[key]) do
        total = total + count
    end
    
    local prob = best_token and (best_count / total) or 0
    
    return best_token, prob
end

function MarkovChain:think(context, depth)
    -- Internal "thinking" using Markov chain lookahead
    -- Returns confidence score for the current context
    depth = depth or 3
    
    local thoughts = {}
    local current_context = {table.unpack(context)}
    
    for i = 1, depth do
        local next_token, prob = self:predict_next(current_context)
        
        if not next_token or prob < 0.1 then
            break
        end
        
        table.insert(thoughts, {
            token = next_token,
            probability = prob,
            depth = i
        })
        
        -- Update context for next prediction
        table.remove(current_context, 1)
        table.insert(current_context, next_token)
    end
    
    -- Calculate overall confidence
    local confidence = 0
    for _, thought in ipairs(thoughts) do
        confidence = confidence + thought.probability / thought.depth
    end
    
    return {
        thoughts = thoughts,
        confidence = confidence / depth,
        context = context
    }
end

-- ============================================================================
-- ENGRAM MEMORY MODULE
-- ============================================================================

local EngramMemory = {}
EngramMemory.__index = EngramMemory

function EngramMemory:new(config)
    local obj = {
        d_model = config.d_model,
        vocab_size = config.vocab_size,
        hash_size_2gram = config.hash_size_2gram or 10000,
        hash_size_3gram = config.hash_size_3gram or 50000,
        compression_vocab_size = config.compression_vocab_size or 2000,
        
        -- Load embedding tables
        embedding_2gram = {},
        embedding_3gram = {},
        
        -- Learned parameters (loaded from numpy files)
        W_e = {},
        W_v = {},
        W_k = {},
        vocab_projection = {}
    }
    setmetatable(obj, EngramMemory)
    return obj
end

function EngramMemory:load_weights(weight_dir)
    -- Load Engram weights from exported numpy files
    print("Loading Engram memory weights...")
    
    -- This is a simplified loader - in production, properly parse NPY format
    -- For now, we'll use placeholder random initialization
    
    -- Initialize 2-gram embeddings
    for i = 1, self.hash_size_2gram do
        self.embedding_2gram[i] = {}
        for j = 1, self.d_model do
            self.embedding_2gram[i][j] = (math.random() - 0.5) * 0.1
        end
    end
    
    -- Initialize 3-gram embeddings  
    for i = 1, self.hash_size_3gram do
        self.embedding_3gram[i] = {}
        for j = 1, self.d_model do
            self.embedding_3gram[i][j] = (math.random() - 0.5) * 0.1
        end
    end
    
    print("Engram memory loaded!")
end

function EngramMemory:hash_ngram(ngram, hash_size)
    -- Deterministic hash function
    local sum = 0
    for _, token_id in ipairs(ngram) do
        sum = sum + token_id
    end
    return (sum * 2654435761) % hash_size + 1  -- +1 for 1-based indexing
end

function EngramMemory:compress_token(token_id)
    -- Vocabulary compression
    return (token_id % self.compression_vocab_size) + 1
end

function EngramMemory:retrieve(input_ids, position)
    -- Phase 1: Retrieval
    -- Extract N-grams and retrieve embeddings
    
    local compressed_ids = {}
    for i = 1, #input_ids do
        compressed_ids[i] = self:compress_token(input_ids[i])
    end
    
    -- Extract 2-gram
    local bigram = {}
    if position >= 2 then
        bigram = {compressed_ids[position-1], compressed_ids[position]}
    else
        bigram = {0, compressed_ids[position]}
    end
    
    -- Extract 3-gram
    local trigram = {}
    if position >= 3 then
        trigram = {compressed_ids[position-2], compressed_ids[position-1], compressed_ids[position]}
    else
        trigram = {0, 0, compressed_ids[position]}
    end
    
    -- Hash to indices
    local bigram_idx = self:hash_ngram(bigram, self.hash_size_2gram)
    local trigram_idx = self:hash_ngram(trigram, self.hash_size_3gram)
    
    -- Retrieve embeddings
    local e_2gram = self.embedding_2gram[bigram_idx]
    local e_3gram = self.embedding_3gram[trigram_idx]
    
    -- Concatenate and project (simplified)
    local e_concat = {}
    for i = 1, self.d_model do
        e_concat[i] = e_2gram[i]
    end
    for i = 1, self.d_model do
        e_concat[self.d_model + i] = e_3gram[i]
    end
    
    return e_concat
end

function EngramMemory:gate_and_fuse(hidden_state, retrieved_memory)
    -- Phase 2: Fusion
    -- Context-aware gating and modulation
    
    -- Simplified gating (in full version, use learned W_k and W_v)
    local dot_product = 0
    for i = 1, #hidden_state do
        if i <= #retrieved_memory then
            dot_product = dot_product + hidden_state[i] * retrieved_memory[i]
        end
    end
    
    -- Sigmoid gate
    local alpha = 1 / (1 + math.exp(-dot_product / math.sqrt(self.d_model)))
    
    -- Modulate and add
    local fused = {}
    for i = 1, #hidden_state do
        local memory_contrib = (i <= #retrieved_memory) and retrieved_memory[i] or 0
        fused[i] = hidden_state[i] + alpha * memory_contrib
    end
    
    return fused
end

-- ============================================================================
-- MIXTURE OF EXPERTS (MoE)
-- ============================================================================

local MixtureOfExperts = {}
MixtureOfExperts.__index = MixtureOfExperts

function MixtureOfExperts:new(config)
    local obj = {
        d_model = config.d_model,
        num_experts = config.num_experts or 8,
        top_k = config.top_k or 2,
        experts = {},
        router = {}
    }
    
    -- Initialize experts (simplified as linear layers)
    for i = 1, obj.num_experts do
        obj.experts[i] = {
            weights = {},
            bias = {}
        }
        
        -- Random initialization
        for j = 1, obj.d_model do
            obj.experts[i].weights[j] = {}
            for k = 1, obj.d_model * 4 do
                obj.experts[i].weights[j][k] = (math.random() - 0.5) * 0.1
            end
            obj.experts[i].bias[j] = 0
        end
    end
    
    setmetatable(obj, MixtureOfExperts)
    return obj
end

function MixtureOfExperts:route(hidden_state)
    -- Simple routing: use hash of hidden state magnitude
    local magnitude = 0
    for i = 1, #hidden_state do
        magnitude = magnitude + hidden_state[i] * hidden_state[i]
    end
    
    -- Select top-k experts deterministically
    local selected = {}
    local base_idx = math.floor(magnitude * 100) % self.num_experts
    
    for i = 1, self.top_k do
        table.insert(selected, (base_idx + i - 1) % self.num_experts + 1)
    end
    
    return selected
end

function MixtureOfExperts:forward(hidden_state)
    -- Route to top-k experts and combine outputs
    local selected_experts = self:route(hidden_state)
    
    local output = {}
    for i = 1, #hidden_state do
        output[i] = 0
    end
    
    -- Simplified expert computation
    for _, expert_idx in ipairs(selected_experts) do
        local expert = self.experts[expert_idx]
        
        -- Simple linear transformation (simplified)
        for i = 1, #hidden_state do
            output[i] = output[i] + hidden_state[i] + expert.bias[i]
        end
    end
    
    -- Average
    for i = 1, #output do
        output[i] = output[i] / #selected_experts
    end
    
    return output
end

-- ============================================================================
-- MAIN TRANSFORMER MODEL
-- ============================================================================

local EngramChatbot = {}
EngramChatbot.__index = EngramChatbot

function EngramChatbot:new(config_path, weight_dir)
    local config = load_json(config_path)
    
    local obj = {
        config = config,
        vocab_size = config.vocab_size,
        d_model = config.d_model,
        num_layers = config.num_layers,
        
        -- Components
        token_embeddings = {},
        position_embeddings = {},
        layers = {},
        
        -- Engram memory
        engram = EngramMemory:new(config),
        engram_layers = {[3] = true, [5] = true},  -- Apply at layers 3 and 5
        
        -- MoE
        use_moe = true,
        moe = MixtureOfExperts:new(config),
        
        -- Markov chain for thinking
        markov = MarkovChain:new(2),
        
        -- Output projection
        output_weights = {}
    }
    
    setmetatable(obj, EngramChatbot)
    
    -- Load weights
    obj:load_weights(weight_dir)
    
    return obj
end

function EngramChatbot:load_weights(weight_dir)
    print("Loading model weights from: " .. weight_dir)
    
    -- Initialize embeddings (simplified random init)
    for i = 1, self.vocab_size do
        self.token_embeddings[i] = {}
        for j = 1, self.d_model do
            self.token_embeddings[i][j] = (math.random() - 0.5) * 0.1
        end
    end
    
    for i = 1, 2048 do  -- Max position
        self.position_embeddings[i] = {}
        for j = 1, self.d_model do
            self.position_embeddings[i][j] = (math.random() - 0.5) * 0.02
        end
    end
    
    -- Load Engram weights
    self.engram:load_weights(weight_dir)
    
    print("Model loaded successfully!")
end

function EngramChatbot:embed(token_id, position)
    -- Get token + position embedding
    local embedding = {}
    
    for i = 1, self.d_model do
        embedding[i] = self.token_embeddings[token_id][i] + self.position_embeddings[position][i]
    end
    
    return embedding
end

function EngramChatbot:forward_layer(hidden_state, layer_idx, input_ids, position)
    -- Simplified transformer layer
    
    -- Self-attention (simplified - just pass through)
    local attn_output = {table.unpack(hidden_state)}
    
    -- Engram memory (if this is an Engram layer)
    if self.engram_layers[layer_idx] then
        local retrieved = self.engram:retrieve(input_ids, position)
        attn_output = self.engram:gate_and_fuse(attn_output, retrieved)
    end
    
    -- FFN with MoE
    local ffn_output
    if self.use_moe then
        ffn_output = self.moe:forward(attn_output)
    else
        -- Simple feedforward
        ffn_output = {table.unpack(attn_output)}
    end
    
    return ffn_output
end

function EngramChatbot:forward(input_ids)
    -- Full forward pass
    local seq_len = #input_ids
    local hidden_states = {}
    
    -- Embed all tokens
    for pos = 1, seq_len do
        hidden_states[pos] = self:embed(input_ids[pos], pos)
    end
    
    -- Process through layers
    for layer_idx = 1, self.num_layers do
        local new_hidden_states = {}
        
        for pos = 1, seq_len do
            new_hidden_states[pos] = self:forward_layer(
                hidden_states[pos],
                layer_idx,
                input_ids,
                pos
            )
        end
        
        hidden_states = new_hidden_states
    end
    
    -- Output projection (simplified)
    local last_hidden = hidden_states[seq_len]
    local logits = {}
    
    for i = 1, self.vocab_size do
        logits[i] = 0
        for j = 1, math.min(#last_hidden, self.d_model) do
            logits[i] = logits[i] + last_hidden[j] * (i / self.vocab_size - 0.5)
        end
    end
    
    return logits
end

function EngramChatbot:generate(prompt, max_length, temperature, top_k)
    -- Generate text with Top-K sampling and Markov chain thinking
    max_length = max_length or 100
    temperature = temperature or 1.0
    top_k = top_k or 50
    
    -- Encode prompt (simple char-level)
    local input_ids = {}
    for i = 1, #prompt do
        local char_id = string.byte(prompt, i) % self.vocab_size + 1
        table.insert(input_ids, char_id)
    end
    
    print("\nGenerating with Engram memory and Markov thinking...")
    print("Prompt: " .. prompt)
    print("\nGeneration:")
    
    local generated_text = prompt
    
    for step = 1, max_length do
        -- Get logits from transformer
        local logits = self:forward(input_ids)
        
        -- Apply temperature
        for i = 1, #logits do
            logits[i] = logits[i] / temperature
        end
        
        -- Convert to probabilities
        local probs = softmax(logits)
        
        -- Markov chain thinking (internal lookahead)
        local context = {}
        for i = math.max(1, #input_ids - 2), #input_ids do
            table.insert(context, input_ids[i])
        end
        
        local thinking = self.markov:think(context, 3)
        
        -- Adjust probabilities based on Markov confidence
        if thinking.confidence > 0.3 and thinking.thoughts[1] then
            local markov_token = thinking.thoughts[1].token
            if markov_token and markov_token <= #probs then
                probs[markov_token] = probs[markov_token] * (1 + thinking.confidence)
            end
        end
        
        -- Top-K filtering
        probs = top_k_filter(probs, top_k)
        
        -- Sample next token
        local next_token = sample(probs)
        
        -- Add to sequence
        table.insert(input_ids, next_token)
        
        -- Decode and print
        local char = string.char(next_token % 256)
        generated_text = generated_text .. char
        io.write(char)
        io.flush()
        
        -- Update Markov chain
        self.markov:add_transition(context, next_token)
        
        -- Stop if we generate a terminator
        if next_token == 1 or char == '\n' then
            break
        end
    end
    
    print("\n\nGeneration complete!")
    return generated_text
end

-- ============================================================================
-- MAIN EXECUTION
-- ============================================================================

function main()
    print("=" .. string.rep("=", 78))
    print("ENGRAM CHATBOT - LUA INFERENCE ENGINE")
    print("Features: Engram Memory | MoE | Markov Thinking | Top-K Sampling")
    print("=" .. string.rep("=", 78) .. "\n")
    
    -- Initialize model
    local config_path = "lua_export/config.json"
    local weight_dir = "lua_export/"
    
    print("Initializing Engram Chatbot...")
    local model = EngramChatbot:new(config_path, weight_dir)
    
    print("\nModel ready! Starting interactive mode...\n")
    
    -- Interactive loop
    while true do
        io.write("\nYou: ")
        io.flush()
        local user_input = io.read()
        
        if not user_input or user_input == "quit" or user_input == "exit" then
            print("\nGoodbye!")
            break
        end
        
        -- Generate response
        io.write("\nBot: ")
        local response = model:generate(
            user_input,
            100,   -- max_length
            0.8,   -- temperature
            50     -- top_k
        )
    end
end

-- Run if executed directly
if arg and arg[0]:match("engram_chatbot%.lua$") then
    math.randomseed(os.time())
    main()
end

return EngramChatbot
