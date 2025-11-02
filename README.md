# AI & Machine Learning Portfolio

A comprehensive portfolio of **19 AI/ML projects**, progressing from fundamental concepts to advanced techniques. Built with a focus on deep understanding through implementation from scratch.

## 📚 Projects Overview

### Foundation (Projects 1-5)

1. **Neural Networks from Scratch** - Build neural networks using only NumPy
   - 5 activation functions (ReLU, Sigmoid, Tanh, Softmax, Linear)
   - 5 loss functions (MSE, MAE, CrossEntropy, etc.)
   - Single neuron, dense layers, and full networks
   - Backpropagation and gradient checking

2. **Classification Pipeline** - Complete end-to-end ML pipeline
   - Data preprocessing and feature engineering
   - Model training and evaluation
   - Hyperparameter tuning

3. **Computer Vision with CNN** - Deep CNNs for image recognition
   - Convolutional layers from scratch
   - Image classification on CIFAR-10
   - Transfer learning with pre-trained models

4. **NLP Text Classification** - Natural language processing basics
   - Text preprocessing and tokenization
   - Word embeddings
   - RNN/LSTM for text

5. **RNN/LSTM for Sequences** - Recurrent neural networks
   - Sequence modeling
   - Time series prediction
   - Attention mechanisms

### Intermediate (Projects 6-10)

6. **Recommendation System** - Collaborative filtering & matrix factorization
7. **Reinforcement Learning** - Q-Learning and policy gradient methods
8. **GANs** - Generative adversarial networks
9. **Transfer Learning** - Fine-tuning pre-trained models
10. **Ensemble Methods** - Boosting, bagging, stacking

### Advanced (Projects 11-19)

11. **Autoencoders** - Anomaly detection with autoencoders
12. **Attention & Transformers** - Self-attention mechanisms and BERT
13. **Time Series Forecasting** - LSTM and statistical methods
14. **Bayesian Inference** - Probabilistic deep learning
15. **Graph Neural Networks** - GNNs for graph-structured data
16. **Meta-Learning** - Learning to learn
17. **Federated Learning** - Distributed machine learning
18. **Explainable AI (XAI)** - Model interpretability and feature importance
19. **Capstone: Humanoid Robotics** - Integration of all concepts

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/AI-Portfolio.git
cd AI-Portfolio

# Create virtual environment
python -m venv venv

# Activate virtual environment

# On Windows:
venv\Scripts\Activate.ps1

# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
pytest tests/ -v
```

---

## 📁 Project Structure

Each project follows this standardized structure:

```
project-name/
├── src/                 # Source code
│   ├── __init__.py
│   ├── main_module.py
│   └── utilities.py
├── tests/               # Unit tests
│   ├── __init__.py
│   ├── test_main.py
│   └── test_utilities.py
├── notebooks/           # Jupyter notebooks for exploration
│   └── exploration.ipynb
├── data/                # Datasets and data files
│   ├── raw/
│   ├── processed/
│   └── README.md
├── models/              # Trained model files
│   └── README.md
├── results/             # Output, results, visualizations
│   ├── plots/
│   ├── metrics/
│   └── README.md
├── assets/              # Images, diagrams, documentation
│   └── README.md
└── README.md            # Project-specific documentation
```

---

## 📊 Technologies Used

### Core Libraries

- **NumPy** (1.24.3) - Numerical computing
- **Pandas** (2.0.3) - Data manipulation and analysis
- **SciPy** (1.11.1) - Scientific computing

### Machine Learning & Deep Learning

- **Scikit-learn** (1.3.0) - Traditional ML algorithms
- **TensorFlow/Keras** - Deep learning framework
- **PyTorch** - PyTorch-based models

### Visualization & Analysis

- **Matplotlib** (3.7.1) - 2D plotting
- **Seaborn** (0.12.2) - Statistical data visualization

### Development & Testing

- **Jupyter** (1.0.0) - Interactive notebooks
- **Pytest** (7.4.0) - Unit testing framework
- **IPython** (8.16.0) - Enhanced Python shell

---

## 🎯 Learning Outcomes

After completing this portfolio, you will understand:

✅ **Fundamentals**

- How neural networks work at a mathematical level
- Backpropagation and gradient descent
- Different activation functions and loss functions

✅ **Deep Learning**

- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs/LSTMs)
- Transformer architectures

✅ **Advanced Topics**

- Generative models (GANs)
- Reinforcement learning
- Transfer learning
- Attention mechanisms

✅ **Production Skills**

- Code organization and testing
- Git and version control
- Documentation and best practices
- Model evaluation and deployment

---

## 📋 Project Status

| Project | Status | Completion |
|---------|--------|-----------|
| 01 - Neural Networks from Scratch | ✅ Complete | 100% |
| 02 - Classification Pipeline | 🚧 In Progress | 0% |
| 03 - Computer Vision CNN | ⏳ Planned | 0% |
| 04-19 | ⏳ Planned | 0% |

---

## 🔧 Running Tests

Each project includes unit tests to verify correctness:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_activations.py -v

# Run with coverage report
pytest tests/ --cov

# Run specific test
pytest tests/test_activations.py::TestReLU::test_forward -v
```

---

## 📖 Documentation

Each project contains:

- ✅ Complete source code with docstrings
- ✅ Comprehensive unit tests
- ✅ Jupyter notebooks with examples
- ✅ README with project-specific details
- ✅ Usage examples and API documentation

---

## 💡 Key Principles

This portfolio emphasizes:

1. **Learning from First Principles** - Build things from scratch with NumPy
2. **Clean Code** - Well-organized, documented, and tested
3. **Best Practices** - Professional development patterns
4. **Progressive Difficulty** - Start simple, progress to advanced
5. **Practical Application** - Real datasets and problems

---

## 🤝 Contributing

This is a personal learning portfolio, but suggestions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

The MIT License allows you to:

- ✅ Use this code for any purpose
- ✅ Modify and distribute
- ✅ Use commercially
- ⚠️ Must include the original license

---

## 👤 Author

**Maâmar M** - AI & Machine Learning Enthusiast

- 📧 Email: <maamar.m@gmail.com>
- 🔗 GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

---

## 📞 Questions & Support

For questions or issues:

1. Check the specific project's README
2. Review the code comments and docstrings
3. Look at the Jupyter notebooks for examples
4. Open an issue on GitHub

---

## 🎓 References & Resources

- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Fast.ai - Practical Deep Learning](https://course.fast.ai/)
- [Stanford CS231n - CNN for Visual Recognition](http://cs231n.stanford.edu/)
- [MIT 6.S191 - Introduction to Deep Learning](http://introtodeeplearning.com/)

---

## 🎉 Getting Started

**Next Steps:**

1. Clone the repository
2. Setup virtual environment
3. Install dependencies
4. Start with Project 1 - Neural Networks from Scratch
5. Read the code and run the tests
6. Experiment with the Jupyter notebooks

---

**Last Updated:** November 2025

**Portfolio Status:** 🚧 In Progress (Project 1 Complete, 18 projects to come)

⭐ If you find this useful, please consider starring the repository!
