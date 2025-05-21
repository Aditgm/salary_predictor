 # 🚀 Salary Predictor: Linear Regression from Scratch

<div align="center">
  <img src="https://github.com/yourusername/salary-predictor/raw/main/assets/regression_plot.gif" width="600" alt="Animated Training Process">
  <p><em>Live training visualization showing loss reduction and line fitting</em></p>
</div>

## 📌 Table of Contents
- [✨ Features](#-features)
- [📦 Installation](#-installation)
- [🧠 Algorithm Deep Dive](#-algorithm-deep-dive)
- [📊 Dataset Overview](#-dataset-overview)
- [🚀 Quick Start](#-quick-start)
- [📈 Advanced Usage](#-advanced-usage)
- [🔍 Results](#-results)
- [🛠️ Project Structure](#️-project-structure)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

## ✨ Features

<div align="center">
  <img src="https://github.com/yourusername/salary-predictor/raw/main/assets/feature_comparison.png" width="700" alt="Feature Comparison">
</div>

| Feature                | Our Implementation | scikit-learn |
|------------------------|-------------------|-------------|
| Custom Learning Rates  | ✅ Yes            | ❌ No        |
| Training Visualization | ✅ Live Plots     | ❌ No        |
| Early Stopping         | ✅ Threshold      | ❌ No        |
| Normalization          | ✅ Built-in       | ❌ Requires preprocessing |

## 📦 Installation

### Method 1: Local Installation
```bash
# Clone with SSH
git clone git@github.com:yourusername/salary-predictor.git

# Or with HTTPS
git clone https://github.com/yourusername/salary-predictor.git

# Navigate to project
cd salary-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```
## 🧠 Algorithm

### Core Mathematics
<div align="center">
  <img src="https://latex.codecogs.com/svg.latex?\bg_white&space;J(w,b)=\frac{1}{2m}\sum_{i=1}^m(y_i-(wx_i+b))^2" width="250" style="background:white;padding:10px;border-radius:5px;">
  <img src="https://latex.codecogs.com/svg.latex?\bg_white&space;\frac{\partial%20J}{\partial%20w}=\frac{1}{m}\sum_{i=1}^m(y_i-(wx_i+b))\cdot%20x_i" width="300" style="background:white;padding:10px;border-radius:5px;">
</div>

### Training Code
```python
for epoch in range(max_epochs):
    predictions = X * weights + bias
    error = predictions - y
    weights -= lr * (1/m) * np.dot(X.T, error)
    bias -= lr * (1/m) * np.sum(error)
```
