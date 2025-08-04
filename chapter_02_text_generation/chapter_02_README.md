# Chapter 2: Text Generation Engine

*From statistical patterns to neural probability distributions*

## 🎯 Learning Objectives

After completing this chapter, you'll understand:
- How language modeling works at a fundamental level
- The evolution from n-gram models to neural approaches
- Character-level tokenization and vocabulary construction
- Neural network architectures for sequence prediction
- Training dynamics and optimization for language tasks

## 📚 Theoretical Foundation

### Language Modeling Problem

Given a sequence of characters/tokens `x₁, x₂, ..., xₜ`, predict the next token `xₜ₊₁`:

```
P(x_{t+1} | x_1, x_2, ..., x_t)
```

This probability distribution is the core of all language models, from simple n-grams to modern transformers.

### Statistical vs Neural Approaches

| Aspect | Statistical (N-gram) | Neural |
|--------|---------------------|--------|
| **Memory** | Explicit count tables | Distributed parameters |
| **Generalization** | Limited to seen contexts | Learned representations |
| **Capacity** | Scales with data size | Fixed parameter count |
| **Smoothing** | Manual techniques | Implicit regularization |

## 🔧 Architecture Overview

### Character-Level Tokenization

```python
# Simple but effective approach
text = "hello world"
chars = sorted(set(text))  # [' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w']
char_to_idx = {ch: i for i, ch in enumerate(chars)}
```

**Benefits:**
- Small vocabulary size
- No out-of-vocabulary issues
- Language-agnostic approach
- Good for understanding fundamentals

### Neural Architecture

```
Input: [c₁, c₂, ..., cₜ] (context characters)
  ↓
Embedding Layer: chars → dense vectors
  ↓
Concatenation: flatten all embeddings
  ↓
MLP: Hidden layers with nonlinearities
  ↓
Output: logits over vocabulary
  ↓
Softmax: probability distribution
```

## 🚀 Running the Code

```bash
cd chapter_02_text_generation/
python language_model.py
```

**Expected Output:**
```
=== Language Model Comparison ===

Vocabulary size: 32
Characters: ['<START>', '<END>', ' ', '.', 'S', 'T', 'a', 'b', 'c', 'd', ...]

Training statistical n-gram model...
Training neural language model...
Epoch 0: Loss = 3.4521
Epoch 10: Loss = 2.1847
Epoch 20: Loss = 1.8392
Epoch 30: Loss = 1.6458
Epoch 40: Loss = 1.5234

=== Model Outputs ===

Statistical Model Samples:
Sample 1: The quick brown fox jumps over the lazy dog was sleeping...
Sample 2: under the warm sun began to set As evening approached they...
Sample 3: forest dwellers The owl had seen many seasons come and go...

Neural Model Samples:
Sample 1: The wise old owl watched the forest creatures and shared stories...
Sample 2: Together under the oak tree they became friends in the meadow...
Sample 3: The butterfly landed softly while the fox played in the sunshine...
```

## 🧪 Experiments

### 1. Context Length Impact

```python
# Test different context lengths
for context_len in [3, 6, 9, 12]:
    model = NeuralLanguageModel(vocab_size, context_len)
    # Train and compare generation quality
```

### 2. Temperature Sampling

```python
# Explore creativity vs coherence trade-off
temperatures = [0.3, 0.7, 1.0, 1.5, 2.0]
for temp in temperatures:
    sample = model.generate(tokenizer, temperature=temp)
    print(f"T={temp}: {sample}")
```

### 3. Architecture Variations

```python
# Compare different hidden dimensions
configs = [
    {'embedding_dim': 8, 'hidden_dim': 32},
    {'embedding_dim': 16, 'hidden_dim': 64},
    {'embedding_dim': 32, 'hidden_dim': 128},
    {'embedding_dim': 64, 'hidden_dim': 256}
]
```

## 📊 Analysis Tools

### Training Dynamics

Monitor key metrics during training:

```python
def analyze_training(trainer, dataloader):
    """Comprehensive training analysis."""
    
    # Loss components
    losses = []
    perplexities = []
    
    for epoch in range(num_epochs):
        loss = trainer.train_epoch(dataloader)
        perplexity = torch.exp(torch.tensor(loss))
        
        losses.append(loss)
        perplexities.append(perplexity.item())
        
        # Log metrics
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss:.4f}, Perplexity={perplexity:.2f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(losses)
    ax1.set_title('Cross-Entropy Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2.plot(perplexities)
    ax2.set_title('Perplexity')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    
    plt.tight_layout()
    plt.show()
```

### Generation Quality Metrics

```python
def evaluate_generation_quality(model, tokenizer, test_contexts):
    """Evaluate generation quality across different metrics."""
    
    metrics = {
        'diversity': [],
        'coherence': [],
        'fluency': []
    }
    
    for context in test_contexts:
        generations = []
        
        # Generate multiple samples
        for _ in range(10):
            sample = model.generate(tokenizer, context=context, temperature=0.8)
            generations.append(sample)
        
        # Compute diversity (unique n-grams)
        all_text = ' '.join(generations)
        words = all_text.split()
        bigrams = set(zip(words[:-1], words[1:]))
        diversity = len(bigrams) / len(words) if words else 0
        
        metrics['diversity'].append(diversity)
        # Add more sophisticated metrics as needed
    
    return metrics
```

## 🎛️ Hyperparameter Tuning

### Key Parameters

1. **Learning Rate**: Start with 1e-3, adjust based on loss curves
2. **Context Length**: Balance between memory and long-range dependencies
3. **Hidden Dimensions**: Scale with vocabulary size and data complexity
4. **Temperature**: Control generation randomness (0.7-1.0 for creative text)

### Systematic Search

```python
def hyperparameter_search():
    """Grid search over key hyperparameters."""
    
    results = []
    
    configs = {
        'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3],
        'hidden_dim': [64, 128, 256],
        'embedding_dim': [16, 32, 64]
    }
    
    for lr in configs['learning_rate']:
        for hidden_dim in configs['hidden_dim']:
            for emb_dim in configs['embedding_dim']:
                
                # Create and train model
                model = NeuralLanguageModel(
                    vocab_size=vocab_size,
                    context_length=context_length,
                    embedding_dim=emb_dim,
                    hidden_dim=hidden_dim
                )
                
                trainer = LanguageModelTrainer(model, tokenizer, learning_rate=lr)
                
                # Train and evaluate
                final_loss = train_model(trainer, dataloader, epochs=30)
                
                results.append({
                    'lr': lr,
                    'hidden_dim': hidden_dim,
                    'embedding_dim': emb_dim,
                    'final_loss': final_loss
                })
    
    # Find best configuration
    best_config = min(results, key=lambda x: x['final_loss'])
    return best_config, results
```

## 🐛 Common Issues and Solutions

### 1. Training Instability

**Problem**: Loss spikes or doesn't decrease
```python
# ❌ Common issues
- Learning rate too high
- No gradient clipping
- Poor weight initialization

# ✅ Solutions
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
```

### 2. Poor Generation Quality

**Problem**: Repetitive or incoherent text
```python
# ❌ Issues
- Context length too short
- Temperature too low/high
- Insufficient training data

# ✅ Improvements
- Increase context window
- Experiment with sampling strategies
- Add more diverse training data
```

### 3. Overfitting

**Problem**: Perfect training loss, poor generalization
```python
# ❌ Signs of overfitting
- Training loss << validation loss
- Generated text memorizes training data

# ✅ Regularization techniques
- Dropout layers
- Weight decay
- Early stopping
- Data augmentation
```

## 🎯 Advanced Techniques

### 1. Nucleus (Top-p) Sampling

```python
def nucleus_sampling(logits, p=0.9):
    """Sample from the top-p probability mass."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Find cutoff point
    cutoff = (cumulative_probs <= p).sum().item()
    
    # Keep only top-p tokens
    top_p_logits = sorted_logits[:cutoff+1]
    top_p_indices = sorted_indices[:cutoff+1]
    
    # Sample from filtered distribution
    probs = F.softmax(top_p_logits, dim=-1)
    selected_idx = torch.multinomial(probs, num_samples=1)
    
    return top_p_indices[selected_idx]
```

### 2. Beam Search

```python
def beam_search(model, tokenizer, context, beam_width=5, max_length=50):
    """Generate text using beam search decoding."""
    
    # Initialize beams
    beams = [(context, 0.0)]  # (sequence, log_prob)
    
    for _ in range(max_length):
        candidates = []
        
        for sequence, score in beams:
            if len(sequence) >= max_length:
                candidates.append((sequence, score))
                continue
            
            # Get next token probabilities
            logits = model(torch.tensor([sequence[-model.context_length:]]))
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Add top-k candidates
            top_k_probs, top_k_indices = torch.topk(log_probs, beam_width)
            
            for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
                new_sequence = sequence + [idx.item()]
                new_score = score + prob.item()
                candidates.append((new_sequence, new_score))
        
        # Keep top beams
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    return beams[0][0]  # Return best sequence
```

## 🔗 Connections to Modern LLMs

### Scaling Laws

The principles learned here apply to large language models:

1. **Data Scaling**: More text → better language understanding
2. **Model Scaling**: More parameters → increased capacity
3. **Compute Scaling**: More training → better optimization

### Architecture Evolution

```
Character-level RNN → Word-level RNN → Transformer → GPT → ChatGPT
```

Each step maintains the same core prediction objective while improving:
- **Context length**: From fixed windows to long sequences
- **Architecture**: From RNNs to attention mechanisms
- **Scale**: From thousands to billions of parameters

## 📈 Next Steps

In Chapter 3, we'll build upon these foundations to create:
1. Multi-layer perceptron architectures
2. Advanced training techniques
3. Better handling of longer sequences
4. Introduction to attention mechanisms

The character-level language modeling skills you've developed here are directly applicable to:
- Code completion systems
- Text generation applications
- Understanding modern transformer architectures
- Building domain-specific language models

## 📚 Further Reading

- **"A Mathematical Framework for Transformer Circuits"** - Understanding attention patterns
- **"Language Models are Few-Shot Learners"** - Scaling and emergent capabilities  
- **"Training language models to follow instructions with human feedback"** - RLHF techniques
- **Character-level Convolutional Networks for Text Classification** - Alternative architectures
