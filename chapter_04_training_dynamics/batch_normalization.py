
# chapter_04_training_dynamics/batch_normalization.py
"""
Advanced Training Dynamics: Batch Normalization and Gradient Flow
Understanding internal covariate shift and training stability
"""

class CustomBatchNorm1d(torch.nn.Module):
    """
    Custom implementation of Batch Normalization for educational purposes.
    Shows the internal mechanics of normalization and gradient flow.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = torch.nn.Parameter(torch.ones(num_features))  # Scale
        self.beta = torch.nn.Parameter(torch.zeros(num_features))  # Shift
        
        # Running statistics (not learned, updated during training)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x):
        """Forward pass with detailed statistics tracking."""
        
        if self.training:
            # Training mode: use batch statistics
            batch_mean = x.mean(dim=0, keepdim=True)
            batch_var = x.var(dim=0, keepdim=True, unbiased=False)
            
            # Update running statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze()
                self.num_batches_tracked += 1
            
            # Normalize using batch statistics
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            
        else:
            # Inference mode: use running statistics
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        # Scale and shift
        output = self.gamma * x_normalized + self.beta
        
        return output
    
    def extra_repr(self):
        return f'num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}'


class TrainingDiagnostics:
    """Tools for analyzing training dynamics and gradient flow."""
    
    def __init__(self, model):
        self.model = model
        self.stats_history = []
    
    def collect_statistics(self, epoch: int):
        """Collect comprehensive training statistics."""
        stats = {'epoch': epoch, 'layers': {}}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                # Weight statistics
                weights = module.weight.data
                stats['layers'][name] = {
                    'weight_mean': weights.mean().item(),
                    'weight_std': weights.std().item(),
                    'weight_abs_max': weights.abs().max().item(),
                }
                
                # Gradient statistics (if available)
                if module.weight.grad is not None:
                    grads = module.weight.grad.data
                    stats['layers'][name].update({
                        'grad_mean': grads.mean().item(),
                        'grad_std': grads.std().item(),
                        'grad_abs_max': grads.abs().max().item(),
                        'grad_norm': grads.norm().item(),
                    })
        
        self.stats_history.append(stats)
    
    def plot_gradient_flow(self):
        """Visualize gradient magnitudes across layers and training."""
        if not self.stats_history:
            print("No statistics collected yet!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract layer names and epochs
        layer_names = list(self.stats_history[0]['layers'].keys())
        epochs = [s['epoch'] for s in self.stats_history]
        
        # Plot 1: Gradient norms over time
        ax1 = axes[0, 0]
        for layer_name in layer_names:
            grad_norms = [s['layers'][layer_name].get('grad_norm', 0) for s in self.stats_history]
            ax1.plot(epochs, grad_norms, label=layer_name, marker='o', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Gradient Norm')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Weight statistics
        ax2 = axes[0, 1]
        for layer_name in layer_names:
            weight_stds = [s['layers'][layer_name]['weight_std'] for s in self.stats_history]
            ax2.plot(epochs, weight_stds, label=layer_name, marker='s', markersize=3)
        ax2.set_title('Weight Standard Deviations')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Weight Std')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Gradient-to-weight ratios
        ax3 = axes[1, 0]
        for layer_name in layer_names:
            ratios = []
            for s in self.stats_history:
                if 'grad_norm' in s['layers'][layer_name]:
                    weight_norm = s['layers'][layer_name]['weight_std'] * math.sqrt(s['layers'][layer_name].get('num_params', 1))
                    grad_norm = s['layers'][layer_name]['grad_norm']
                    ratios.append(grad_norm / (weight_norm + 1e-8))
                else:
                    ratios.append(0)
            ax3.plot(epochs, ratios, label=layer_name, marker='^', markersize=3)
        ax3.set_title('Gradient-to-Weight Ratios')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Grad/Weight Ratio')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Activation magnitudes (if tracked)
        ax4 = axes[1, 1]
        ax4.set_title('Layer Health Summary')
        ax4.text(0.1, 0.8, 'Green: Healthy gradients (1e-3 to 1e-1)', transform=ax4.transAxes, color='green')
        ax4.text(0.1, 0.6, 'Yellow: Moderate issues (1e-4 to 1e-3, 1e-1 to 1)', transform=ax4.transAxes, color='orange')
        ax4.text(0.1, 0.4, 'Red: Serious problems (<1e-4 or >1)', transform=ax4.transAxes, color='red')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
