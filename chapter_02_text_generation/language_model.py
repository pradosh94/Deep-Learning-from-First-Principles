# chapter_02_text_generation/language_model.py
"""
Character-Level Language Models
From n-gram statistics to neural probability distributions
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional


class CharacterTokenizer:
    """Simple character-level tokenizer with special tokens."""
    
    def __init__(self, text: str):
        # Extract unique characters and add special tokens
        chars = sorted(list(set(text)))
        self.start_token = '<START>'
        self.end_token = '<END>'
        
        self.chars = [self.start_token, self.end_token] + chars
        self.vocab_size = len(self.chars)
        
        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text: str) -> List[int]:
        """Convert text to list of integers."""
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices: List[int]) -> str:
        """Convert list of integers back to text."""
        return ''.join([self.idx_to_char[i] for i in indices])
    
    def get_start_token_id(self) -> int:
        return self.char_to_idx[self.start_token]
    
    def get_end_token_id(self) -> int:
        return self.char_to_idx[self.end_token]


class StatisticalLanguageModel:
    """N-gram based language model for baseline comparison."""
    
    def __init__(self, n: int = 2):
        self.n = n  # Context length
        self.counts = defaultdict(Counter)
        self.context_counts = Counter()
    
    def train(self, text: str, tokenizer: CharacterTokenizer):
        """Train n-gram model on text."""
        # Tokenize and add special tokens
        tokens = [tokenizer.get_start_token_id()] + tokenizer.encode(text) + [tokenizer.get_end_token_id()]
        
        # Count n-grams
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i+self.n-1])
            next_char = tokens[i+self.n-1]
            
            self.counts[context][next_char] += 1
            self.context_counts[context] += 1
    
    def get_probabilities(self, context: Tuple[int, ...]) -> Dict[int, float]:
        """Get probability distribution for next character given context."""
        if context in self.counts:
            total = self.context_counts[context]
            return {char: count / total for char, count in self.counts[context].items()}
        else:
            # Uniform distribution if context not seen
            return {}
    
    def generate(self, tokenizer: CharacterTokenizer, max_length: int = 100) -> str:
        """Generate text using the trained model."""
        context = tuple([tokenizer.get_start_token_id()] * (self.n - 1))
        generated = []
        
        for _ in range(max_length):
            probs = self.get_probabilities(context)
            if not probs:
                break
            
            # Sample next character
            chars, weights = zip(*probs.items())
            next_char = random.choices(chars, weights=weights)[0]
            
            if next_char == tokenizer.get_end_token_id():
                break
            
            generated.append(next_char)
            context = context[1:] + (next_char,)
        
        return tokenizer.decode(generated)


class NeuralLanguageModel(torch.nn.Module):
    """Simple neural language model using character embeddings."""
    
    def __init__(self, vocab_size: int, context_length: int, embedding_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        
        # Character embedding table
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # Neural network layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(context_length * embedding_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, vocab_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        """Forward pass through the neural language model."""
        batch_size, context_len = x.shape
        
        # Get embeddings and flatten
        embeddings = self.embedding(x)  # (batch_size, context_len, embedding_dim)
        embeddings_flat = embeddings.view(batch_size, -1)  # (batch_size, context_len * embedding_dim)
        
        # Pass through neural network
        logits = self.layers(embeddings_flat)  # (batch_size, vocab_size)
        
        return logits
    
    def generate(self, tokenizer, context=None, max_length=100, temperature=1.0, top_k=None):
        """Generate text using the trained neural model."""
        self.eval()
        
        if context is None:
            # Start with special start tokens
            context = [tokenizer.get_start_token_id()] * self.context_length
        else:
            context = tokenizer.encode(context)
            if len(context) < self.context_length:
                context = [tokenizer.get_start_token_id()] * (self.context_length - len(context)) + context
            else:
                context = context[-self.context_length:]
        
        generated = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # Convert context to tensor
                x = torch.tensor([context], dtype=torch.long)
                
                # Get predictions
                logits = self.forward(x)[0]  # Remove batch dimension
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[-1]] = -float('inf')
                
                # Sample from probability distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Stop if we hit end token
                if next_token == tokenizer.get_end_token_id():
                    break
                
                generated.append(next_token)
                
                # Update context (sliding window)
                context = context[1:] + [next_token]
        
        return tokenizer.decode(generated)


class LanguageModelTrainer:
    """Training utilities for language models."""
    
    def __init__(self, model, tokenizer, learning_rate=1e-3):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.loss_history = []
    
    def create_training_data(self, text, batch_size=32):
        """Create training batches from text."""
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        # Create context-target pairs
        contexts, targets = [], []
        
        for i in range(len(tokens) - self.model.context_length):
            context = tokens[i:i + self.model.context_length]
            target = tokens[i + self.model.context_length]
            contexts.append(context)
            targets.append(target)
        
        # Convert to tensors
        X = torch.tensor(contexts, dtype=torch.long)
        Y = torch.tensor(targets, dtype=torch.long)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (X, Y) in enumerate(dataloader):
            # Forward pass
            logits = self.model(X)
            loss = F.cross_entropy(logits, Y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        self.loss_history.append(avg_loss)
        return avg_loss
    
    def evaluate(self, dataloader):
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X, Y in dataloader:
                logits = self.model(X)
                loss = F.cross_entropy(logits, Y)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def plot_training_curve(self):
        """Plot training loss curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.grid(True)
        plt.show()


def compare_models(text_data, context_length=8):
    """Compare statistical vs neural language models."""
    print("=== Language Model Comparison ===\n")
    
    # Create tokenizer
    tokenizer = CharacterTokenizer(text_data)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Characters: {tokenizer.chars[:20]}{'...' if len(tokenizer.chars) > 20 else ''}\n")
    
    # Train statistical model
    print("Training statistical n-gram model...")
    stat_model = StatisticalLanguageModel(n=context_length)
    stat_model.train(text_data, tokenizer)
    
    # Train neural model
    print("Training neural language model...")
    neural_model = NeuralLanguageModel(
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        embedding_dim=24,
        hidden_dim=128
    )
    
    trainer = LanguageModelTrainer(neural_model, tokenizer, learning_rate=1e-3)
    dataloader = trainer.create_training_data(text_data, batch_size=64)
    
    # Train for several epochs
    for epoch in range(50):
        loss = trainer.train_epoch(dataloader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    print("\n=== Model Outputs ===")
    
    # Generate samples from both models
    print("\nStatistical Model Samples:")
    for i in range(3):
        sample = stat_model.generate(tokenizer, max_length=80)
        print(f"Sample {i+1}: {sample}")
    
    print("\nNeural Model Samples:")
    for i in range(3):
        sample = neural_model.generate(tokenizer, max_length=80, temperature=0.8)
        print(f"Sample {i+1}: {sample}")
    
    # Plot training curve
    trainer.plot_training_curve()
    
    return stat_model, neural_model, tokenizer


# Example usage and experiments
if __name__ == "__main__":
    # Sample text data (you can replace with your own dataset)
    sample_text = """
    The quick brown fox jumps over the lazy dog. The dog was sleeping peacefully 
    under the warm sun. Suddenly, a butterfly landed on the dog's nose, causing 
    it to wake up with a start. The fox, seeing this, decided to play a game of 
    chase with both the dog and the butterfly. They ran through the meadow, 
    laughing and playing until the sun began to set. As evening approached, they 
    all settled down together under a large oak tree, becoming the best of friends.
    
    In another part of the forest, a wise old owl watched over all the woodland 
    creatures. The owl had seen many seasons come and go, and knew the secrets 
    of the forest better than anyone. When young animals came to ask for advice, 
    the owl would share stories of courage, friendship, and the importance of 
    working together. These stories were passed down from generation to generation, 
    creating a rich tradition of wisdom and community among all the forest dwellers.
    """
    
    # Clean up the text
    sample_text = ' '.join(sample_text.split())
    
    print("Starting language model comparison...")
    stat_model, neural_model, tokenizer = compare_models(sample_text, context_length=6)
    
    # Interactive generation
    print("\n=== Interactive Generation ===")
    print("Enter a starting context (or press Enter for random generation):")
    
    while True:
        user_input = input("\nContext: ").strip()
        if user_input.lower() == 'quit':
            break
        
        context = user_input if user_input else None
        
        print("\nNeural model generation:")
        neural_output = neural_model.generate(tokenizer, context=context, 
                                             max_length=100, temperature=0.7, top_k=20)
        print(f"Output: {neural_output}")
        
        if len(user_input) >= 2:  # Statistical model needs some context
            print("\nStatistical model generation:")
            try:
                stat_output = stat_model.generate(tokenizer, max_length=100)
                print(f"Output: {stat_output}")
            except:
                print("Statistical model couldn't generate from this context")
    
    print("\nTraining complete! Check the generated samples above.")
