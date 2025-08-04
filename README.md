# Deep Learning from First Principles 🧠

*Building Neural Networks Through Mathematical Intuition*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Philosophy

> "The best way to understand something is to build it from scratch."

This repository takes you on a journey from basic calculus to modern transformer architectures, implementing every component from the ground up. No magic frameworks, no black boxes - just pure mathematical understanding translated into clean, readable Python.

## Learning Journey

### Part I: Mathematical Foundations (Chapters 1-7)

**Chapter 1: Autodiff Engine** - Build a scalar-based automatic differentiation system
**Chapter 2: Text Generation** - Create character-level language models  
**Chapter 3: Neural Architectures** - Implement feedforward networks with proper training
**Chapter 4: Training Dynamics** - Master gradient flow and normalization techniques
**Chapter 5: Gradient Mastery** - Hand-implement backpropagation for deep understanding
**Chapter 6: Hierarchical Learning** - Build convolutional and tree-structured networks
**Chapter 7: Attention Mechanisms** - Implement the transformer architecture from scratch

### Part II: Advanced Techniques (Chapters 8-10)

**Chapter 8: Optimization Strategies** - Advanced training techniques and regularization
**Chapter 9: Scaling Laws** - Distributed training and efficient architectures  
**Chapter 10: Production Systems** - Deployment patterns and real-world considerations

## Quick Start

```bash
# Clone the repository
git clone https://github.com/pradosh94/deep-learning-from-first-principles.git
cd deep-learning-from-first-principles

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run examples
python chapter_01_autodiff/gradient_engine.py
```

## What You'll Learn

- **Mathematical Intuition**: Understanding backpropagation through computational graphs
- **Implementation Skills**: Building neural networks without high-level frameworks
- **Debugging Techniques**: Identifying and fixing training issues
- **Architecture Design**: Creating efficient and scalable models
- **Optimization Theory**: Advanced training strategies and convergence analysis

##  Repository Structure

```
deep-learning-from-first-principles/
├── chapter_01_autodiff/           # Automatic differentiation engine
├── chapter_02_text_generation/    # Language modeling fundamentals
├── chapter_03_neural_nets/        # Multi-layer perceptrons
├── chapter_04_training_dynamics/  # Gradient flow and normalization
├── chapter_05_gradient_mastery/   # Manual backpropagation
├── chapter_06_hierarchical/       # Convolutional architectures
├── chapter_07_attention/          # Transformer implementation
├── chapter_08_optimization/       # Advanced training techniques
├── chapter_09_scaling/            # Distributed and efficient training
├── chapter_10_production/         # Deployment and serving
├── utils/                         # Shared utilities and visualizations
└── notebooks/                     # Interactive Jupyter examples
```

##  Prerequisites

- **Programming**: Intermediate Python knowledge
- **Mathematics**: Basic calculus and linear algebra
- **Mindset**: Curiosity and willingness to implement everything from scratch

##  Contributing

This is an educational resource built for the community. Contributions, suggestions, and improvements are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-improvement`)
3. Commit your changes (`git commit -m 'Add amazing improvement'`)
4. Push to the branch (`git push origin feature/amazing-improvement`)
5. Open a Pull Request

##  Citation

If you find this educational material helpful in your research or learning, please cite:

```bibtex
@misc{deep_learning_first_principles,
  title={Deep Learning from First Principles: A Mathematical Journey},
  author={Pradosh P. Dash},
  year={2025},
  url={https://github.com/pradosh94/deep-learning-from-first-principles}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with inspiration from mathematical elegance and pedagogical clarity from [Andrej Karpathy][https://github.com/karpathy]
- Special thanks to the open-source community for foundational tools
- Dedicated to learners who prefer understanding over memorization

---

*"In the end, we want to understand not just how to use these tools, but how they work at the deepest level."*
