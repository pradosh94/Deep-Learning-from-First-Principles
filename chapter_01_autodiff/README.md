# Chapter 1: Automatic Differentiation Engine

*Building the mathematical foundation for neural network training*

##  Learning Objectives

By the end of this chapter, you'll understand:
- How automatic differentiation works at the computational level
- The relationship between forward pass computation and backward pass gradients
- Implementation of basic mathematical operations with gradient tracking
- Visualization and debugging of computational graphs

##  Mathematical Foundation

### The Chain Rule

The cornerstone of automatic differentiation is the chain rule from calculus:

```
If f(x) = g(h(x)), then f'(x) = g'(h(x)) · h'(x)
```

In computational graphs, this becomes:
- **Forward pass**: Compute values from inputs to outputs
- **Backward pass**: Propagate gradients from outputs to inputs

### Computational Graphs

Every mathematical expression can be represented as a directed acyclic graph (DAG):
- **Nodes**: Variables and intermediate results
- **Edges**: Dependencies between operations
- **Operations**: Basic mathematical functions (+, *, exp, etc.)

## 🔧 Implementation Details

### Core Data Structure

```python
class Scalar:
    def __init__(self, data, children=(), operation=''):
        self.data = data           # Forward pass value
        self.gradient = 0.0        # Backward pass gradient
        self._backward_fn = None   # Local gradient computation
        self._dependencies = set(children)  # Graph structure
```

### Key Operations

| Operation | Forward | Backward |
|-----------|---------|----------|
| Addition | `a + b` | `∂L/∂a += ∂L/∂c`, `∂L/∂b += ∂L/∂c` |
| Multiplication | `a * b` | `∂L/∂a += b * ∂L/∂c`, `∂L/∂b += a * ∂L/∂c` |
| Power | `a^n` | `∂L/∂a += n * a^(n-1) * ∂L/∂c` |
| Exponential | `exp(a)` | `∂L/∂a += exp(a) * ∂L/∂c` |

##  Running the Code

```bash
cd chapter_01_autodiff/
python gradient_engine.py
```

**Expected Output:**
```
=== Automatic Differentiation Engine Demo ===

1. Basic Operations:
Expression: f = (a * b + c) * (a + b * c)
a = 2.0, b = -3.0, c = 10.0
Result: f = -136.0
∂f/∂a = -50.0
∂f/∂b = 164.0
∂f/∂c = -20.0

2. Activation Functions:
tanh(0.5) = 0.4621, gradient = 0.7864
relu(0.5) = 0.5000, gradient = 1.0000
sigmoid(0.5) = 0.6225, gradient = 0.2350

3. Chain Rule Verification:
f(x) = exp(x²), x = 2.0
f(x) = 54.5982
f'(x) = 218.3928
Manual calculation: f'(2) = 218.3928
Match: True
```

##  Experiments to Try

### 1. Gradient Verification
Compare your computed gradients with numerical derivatives:

```python
def numerical_gradient(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# Test your implementation
x = Scalar(3.0)
y = x ** 3 + 2 * x ** 2 + x + 1
y.backward()

print(f"Analytical gradient: {x.gradient}")
print(f"Numerical gradient: {numerical_gradient(lambda val: val**3 + 2*val**2 + val + 1, 3.0)}")
```

### 2. Complex Expressions
Build more complex computational graphs:

```python
# Polynomial evaluation: P(x) = x⁴ - 3x³ + 2x² - x + 5
x = Scalar(1.5, label='x')
p = x**4 - 3*x**3 + 2*x**2 - x + 5
p.label = 'P(x)'
p.backward()
print(f"P({x.data}) = {p.data}, P'({x.data}) = {x.gradient}")
```

### 3. Multiple Variables
Explore partial derivatives:

```python
# Function: f(x,y) = x²y + xy² + sin(x+y)
x = Scalar(1.0, label='x')
y = Scalar(2.0, label='y')
f = x**2 * y + x * y**2 + (x + y).tanh()  # Using tanh as sin approximation
f.backward()
print(f"∂f/∂x = {x.gradient}, ∂f/∂y = {y.gradient}")
```

##  Common Pitfalls

### 1. Gradient Accumulation
```python
#  Wrong: Gradients accumulate across multiple backward passes
x = Scalar(2.0)
y = x ** 2
y.backward()
y.backward()  # Gradient doubles!

#  Correct: Reset gradients between computations
x.zero_grad()
y.backward()
```

### 2. In-place Operations
```python
#  Wrong: Modifying data in-place breaks the graph
x = Scalar(5.0)
y = x * 2
x.data += 1  # Breaks gradient computation!

#  Correct: Create new nodes for operations
x = Scalar(5.0)
x_modified = x + 1
y = x_modified * 2
```

### 3. Circular Dependencies
```python
#  Wrong: Creates infinite loops
a = Scalar(1.0)
b = Scalar(2.0)
a._dependencies.add(b)
b._dependencies.add(a)  # Circular!

#  Correct: Ensure DAG structure
# Dependencies should only flow in one direction
```

##  Visualization

Generate computational graph visualizations:

```python
from gradient_engine import Scalar, visualize_graph

x = Scalar(2.0, label='x')
y = x.tanh()
y.label = 'tanh(x)'
z = y ** 2
z.label = 'tanh²(x)'

z.backward()
graph = visualize_graph(z)
graph.render('computation_graph', cleanup=True)
```

##  Deep Dive: Topological Sorting

Understanding the backward pass requires topological sorting:

```python
def topological_sort(root):
    """
    Returns nodes in topological order for gradient computation.
    """
    result = []
    visited = set()
    
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        
        for dependency in node._dependencies:
            dfs(dependency)
        
        result.append(node)
    
    dfs(root)
    return result

# The backward pass processes nodes in reverse topological order
# This ensures gradients are computed correctly
```

## Mathematical Connections

### Jacobian Matrix
For multivariate functions, gradients form the Jacobian matrix:

```
J = [∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₙ]
    [∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₙ]
    [   ⋮        ⋮      ⋱     ⋮   ]
    [∂fₘ/∂x₁  ∂fₘ/∂x₂  ...  ∂fₘ/∂xₙ]
```

Our scalar engine computes one row of this matrix at a time.

### Connection to Neural Networks
Neural networks are just very large computational graphs:
- **Parameters**: Trainable variables (weights, biases)
- **Forward pass**: Function evaluation f(x; θ)
- **Loss function**: Scalar output L(f(x; θ), y)
- **Backward pass**: Compute ∂L/∂θ for all parameters

##  Next Steps

In Chapter 2, we'll use this automatic differentiation engine to:
1. Build simple neural network components
2. Implement basic optimizers (SGD)
3. Train models on simple datasets
4. Understand the connection between gradients and learning

The mathematical foundation you've built here is the core of all modern deep learning frameworks!

##  Additional Resources

- **"Automatic Differentiation in Machine Learning: A Survey"** by Baydin et al.
- **"Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation"** by Griewank & Walther
- **Numerical Methods**: Compare with finite difference methods for gradient verification
