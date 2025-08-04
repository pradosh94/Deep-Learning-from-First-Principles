# chapter_01_autodiff/gradient_engine.py
"""
Automatic Differentiation Engine
Building a scalar-valued computational graph with gradient computation
"""

import math
import random
from typing import Set, List, Callable, Optional
from graphviz import Digraph


class Scalar:
    """
    A scalar value with automatic differentiation capabilities.
    
    This class represents a node in a computational graph, storing both
    the value and gradient information needed for backpropagation.
    """
    
    def __init__(self, data: float, children: tuple = (), operation: str = '', label: str = ''):
        self.data = data
        self.gradient = 0.0
        self.label = label
        
        # Graph construction metadata
        self._backward_fn: Callable = lambda: None
        self._dependencies: Set['Scalar'] = set(children)
        self._operation = operation
    
    def __repr__(self):
        return f"Scalar(data={self.data:.4f}, grad={self.gradient:.4f})"
    
    def __add__(self, other):
        """Addition operation with gradient computation"""
        other = other if isinstance(other, Scalar) else Scalar(other)
        result = Scalar(self.data + other.data, (self, other), '+')
        
        def _backward():
            # Gradient of addition: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
            self.gradient += result.gradient
            other.gradient += result.gradient
        
        result._backward_fn = _backward
        return result
    
    def __mul__(self, other):
        """Multiplication operation with gradient computation"""
        other = other if isinstance(other, Scalar) else Scalar(other)
        result = Scalar(self.data * other.data, (self, other), '*')
        
        def _backward():
            # Gradient of multiplication: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
            self.gradient += other.data * result.gradient
            other.gradient += self.data * result.gradient
        
        result._backward_fn = _backward
        return result
    
    def __pow__(self, exponent):
        """Power operation with gradient computation"""
        assert isinstance(exponent, (int, float)), "Only numeric exponents supported"
        result = Scalar(self.data ** exponent, (self,), f'**{exponent}')
        
        def _backward():
            # Gradient of power: ∂(a^n)/∂a = n * a^(n-1)
            self.gradient += exponent * (self.data ** (exponent - 1)) * result.gradient
        
        result._backward_fn = _backward
        return result
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return other * (self ** -1)
    
    def exp(self):
        """Exponential function with gradient computation"""
        result = Scalar(math.exp(self.data), (self,), 'exp')
        
        def _backward():
            # Gradient of exp: ∂(e^a)/∂a = e^a
            self.gradient += result.data * result.gradient
        
        result._backward_fn = _backward
        return result
    
    def log(self):
        """Natural logarithm with gradient computation"""
        result = Scalar(math.log(self.data), (self,), 'log')
        
        def _backward():
            # Gradient of log: ∂(ln(a))/∂a = 1/a
            self.gradient += (1.0 / self.data) * result.gradient
        
        result._backward_fn = _backward
        return result
    
    def tanh(self):
        """Hyperbolic tangent activation function"""
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        result = Scalar(t, (self,), 'tanh')
        
        def _backward():
            # Gradient of tanh: ∂(tanh(a))/∂a = 1 - tanh²(a)
            self.gradient += (1 - t**2) * result.gradient
        
        result._backward_fn = _backward
        return result
    
    def relu(self):
        """Rectified Linear Unit activation function"""
        result = Scalar(0 if self.data < 0 else self.data, (self,), 'ReLU')
        
        def _backward():
            # Gradient of ReLU: ∂(ReLU(a))/∂a = 1 if a > 0, else 0
            self.gradient += (result.data > 0) * result.gradient
        
        result._backward_fn = _backward
        return result
    
    def sigmoid(self):
        """Sigmoid activation function"""
        s = 1 / (1 + math.exp(-self.data))
        result = Scalar(s, (self,), 'sigmoid')
        
        def _backward():
            # Gradient of sigmoid: ∂(σ(a))/∂a = σ(a) * (1 - σ(a))
            self.gradient += s * (1 - s) * result.gradient
        
        result._backward_fn = _backward
        return result
    
    def backward(self):
        """
        Perform backpropagation through the computational graph.
        
        Uses topological sorting to ensure gradients are computed in the correct order.
        """
        # Build topological ordering of computation graph
        topology = []
        visited = set()
        
        def build_topology(node):
            if node not in visited:
                visited.add(node)
                for dependency in node._dependencies:
                    build_topology(dependency)
                topology.append(node)
        
        build_topology(self)
        
        # Initialize gradient of output node
        self.gradient = 1.0
        
        # Propagate gradients backward through the graph
        for node in reversed(topology):
            node._backward_fn()
    
    def zero_grad(self):
        """Reset gradient to zero"""
        self.gradient = 0.0


def visualize_graph(root: Scalar, filename: str = 'computation_graph'):
    """
    Create a visual representation of the computational graph using Graphviz.
    """
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    
    nodes, edges = set(), set()
    
    def build_graph(node):
        if node not in nodes:
            uid = str(id(node))
            dot.node(name=uid, label=f"{node.label} | data {node.data:.4f} | grad {node.gradient:.4f}", shape='record')
            nodes.add(node)
            
            if node._operation:
                op_uid = uid + node._operation
                dot.node(name=op_uid, label=node._operation)
                dot.edge(op_uid, uid)
            
            for child in node._dependencies:
                build_graph(child)
                edge = (str(id(child)), op_uid if node._operation else uid)
                if edge not in edges:
                    dot.edge(*edge)
                    edges.add(edge)
    
    build_graph(root)
    return dot


# Example usage and testing
if __name__ == "__main__":
    print("=== Automatic Differentiation Engine Demo ===\n")
    
    # Example 1: Basic operations
    print("1. Basic Operations:")
    a = Scalar(2.0, label='a')
    b = Scalar(-3.0, label='b')
    c = Scalar(10.0, label='c')
    
    # Complex expression: f = (a * b + c) * (a + b * c)
    e = a * b
    e.label = 'a*b'
    d = e + c
    d.label = 'a*b+c'
    f = c * b
    f.label = 'c*b'
    g = a + f
    g.label = 'a+c*b'
    h = d * g
    h.label = 'final'
    
    print(f"Expression: f = (a * b + c) * (a + b * c)")
    print(f"a = {a.data}, b = {b.data}, c = {c.data}")
    print(f"Result: f = {h.data}")
    
    # Compute gradients
    h.backward()
    print(f"∂f/∂a = {a.gradient}")
    print(f"∂f/∂b = {b.gradient}")
    print(f"∂f/∂c = {c.gradient}")
    
    # Example 2: Activation functions
    print("\n2. Activation Functions:")
    x = Scalar(0.5, label='x')
    
    y_tanh = x.tanh()
    y_tanh.label = 'tanh(x)'
    y_tanh.backward()
    print(f"tanh({x.data}) = {y_tanh.data:.4f}, gradient = {x.gradient:.4f}")
    
    x.zero_grad()  # Reset gradient
    y_relu = x.relu()
    y_relu.label = 'relu(x)'
    y_relu.backward()
    print(f"relu({x.data}) = {y_relu.data:.4f}, gradient = {x.gradient:.4f}")
    
    x.zero_grad()
    y_sigmoid = x.sigmoid()
    y_sigmoid.label = 'sigmoid(x)'
    y_sigmoid.backward()
    print(f"sigmoid({x.data}) = {y_sigmoid.data:.4f}, gradient = {x.gradient:.4f}")
    
    # Example 3: Chain rule verification
    print("\n3. Chain Rule Verification:")
    x = Scalar(2.0, label='x')
    y = x ** 2
    y.label = 'x²'
    z = y.exp()
    z.label = 'exp(x²)'
    
    z.backward()
    print(f"f(x) = exp(x²), x = {x.data}")
    print(f"f(x) = {z.data:.4f}")
    print(f"f'(x) = {x.gradient:.4f}")
    
    # Manual verification: f'(x) = 2x * exp(x²)
    manual_derivative = 2 * x.data * math.exp(x.data ** 2)
    print(f"Manual calculation: f'(2) = {manual_derivative:.4f}")
    print(f"Match: {abs(x.gradient - manual_derivative) < 1e-6}")
