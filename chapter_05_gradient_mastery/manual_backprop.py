
# chapter_05_gradient_mastery/manual_backprop.py
"""
Manual Backpropagation Implementation
Understanding gradient computation at the tensor level
"""

class ManualBackpropMLP:
    """
    MLP with manual backpropagation for educational purposes.
    Shows exactly how gradients flow through each operation.
    """
    
    def __init__(self, vocab_size: int, context_length: int, n_embd: int = 32, n_hidden: int = 64):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.n_embd = n_embd
        self.n_hidden = n_hidden
        
        # Initialize parameters
        self.C = torch.randn((vocab_size, n_embd)) * 0.02  # Embedding matrix
        self.W1 = torch.randn((context_length * n_embd, n_hidden)) * 0.02  # First layer weights
        self.b1 = torch.randn(n_hidden) * 0.02  # First layer bias
        self.W2 = torch.randn((n_hidden, vocab_size)) * 0.02  # Output weights
        self.b2 = torch.randn(vocab_size) * 0.02  # Output bias
        
        # Store for gradient computation
        self.cache = {}
    
    def forward(self, X, Y=None):
        """Forward pass with intermediate value caching."""
        B, T = X.shape
        
        # Embedding lookup
        emb = self.C[X]  # (B, T, n_embd)
        self.cache['emb'] = emb
        
        # Flatten embeddings
        embcat = emb.view(B, -1)  # (B, T * n_embd)
        self.cache['embcat'] = embcat
        
        # First linear layer
        h_preact = embcat @ self.W1 + self.b1  # (B, n_hidden)
        self.cache['h_preact'] = h_preact
        
        # Activation function (tanh)
        h = torch.tanh(h_preact)  # (B, n_hidden)
        self.cache['h'] = h
        
        # Output layer
        logits = h @ self.W2 + self.b2  # (B, vocab_size)
        self.cache['logits'] = logits
        
        # Loss computation
        if Y is not None:
            loss = F.cross_entropy(logits, Y)
            return logits, loss
        
        return logits
    
    def backward_manual(self, X, Y):
        """Manual backpropagation through the entire network."""
        B, T = X.shape
        
        # Forward pass to populate cache
        logits, loss = self.forward(X, Y)
        
        # Start backpropagation from loss
        # Cross-entropy backward: dlogits = softmax(logits) - one_hot(Y)
        dlogits = F.softmax(logits, dim=1)
        dlogits[range(B), Y] -= 1  # Subtract 1 from correct class
        dlogits /= B  # Average over batch
        
        # Output layer backward: logits = h @ W2 + b2
        dh = dlogits @ self.W2.T  # (B, n_hidden)
        dW2 = self.cache['h'].T @ dlogits  # (n_hidden, vocab_size)
        db2 = dlogits.sum(0)  # (vocab_size,)
        
        # Tanh activation backward: h = tanh(h_preact)
        dh_preact = (1 - self.cache['h']**2) * dh  # (B, n_hidden)
        
        # First linear layer backward: h_preact = embcat @ W1 + b1
        dembcat = dh_preact @ self.W1.T  # (B, T * n_embd)
        dW1 = self.cache['embcat'].T @ dh_preact  # (T * n_embd, n_hidden)
        db1 = dh_preact.sum(0)  # (n_hidden,)
        
        # Reshape embedding gradients
        demb = dembcat.view(B, T, self.n_embd)  # (B, T, n_embd)
        
        # Embedding backward: emb = C[X]
        dC = torch.zeros_like(self.C)
        for b in range(B):
            for t in range(T):
                ix = X[b, t]
                dC[ix] += demb[b, t]
        
        # Store gradients
        gradients = {
            'dC': dC,
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
        
        return loss, gradients
    
    def update_parameters(self, gradients, learning_rate=0.01):
        """Update parameters using computed gradients."""
        with torch.no_grad():
            self.C -= learning_rate * gradients['dC']
            self.W1 -= learning_rate * gradients['dW1']
            self.b1 -= learning_rate * gradients['db1']
            self.W2 -= learning_rate * gradients['dW2']
            self.b2 -= learning_rate * gradients['db2']
