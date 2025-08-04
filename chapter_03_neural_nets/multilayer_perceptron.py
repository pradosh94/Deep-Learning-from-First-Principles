
# chapter_03_neural_nets/multilayer_perceptron.py
"""
Multi-Layer Perceptrons for Language Modeling
Advanced architectures with proper initialization and training
"""

import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class ModernMLP(torch.nn.Module):
    """Multi-layer perceptron with modern training techniques."""
    
    def __init__(self, vocab_size: int, context_length: int, n_embd: int = 64, 
                 n_hidden: int = 128, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.n_embd = n_embd
        
        # Embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, n_embd)
        
        # Build MLP layers
        layers = []
        input_dim = context_length * n_embd
        
        for i in range(n_layers):
            layers.extend([
                torch.nn.Linear(input_dim if i == 0 else n_hidden, n_hidden),
                torch.nn.BatchNorm1d(n_hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout)
            ])
        
        # Output projection
        layers.append(torch.nn.Linear(n_hidden, vocab_size))
        
        self.mlp = torch.nn.Sequential(*layers)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Proper weight initialization for stable training."""
        if isinstance(module, torch.nn.Linear):
            # Xavier/He initialization based on activation
            if hasattr(module, 'activation') and module.activation == 'relu':
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            else:
                torch.nn.init.xavier_normal_(module.weight)
            
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        elif isinstance(module, torch.nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through the MLP."""
        B, T = x.shape
        
        # Get embeddings
        embeddings = self.embedding(x)  # (B, T, n_embd)
        
        # Flatten for MLP
        embeddings_flat = embeddings.view(B, T * self.n_embd)  # (B, T * n_embd)
        
        # Pass through MLP
        logits = self.mlp(embeddings_flat)  # (B, vocab_size)
        
        return logits
