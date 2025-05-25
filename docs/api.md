# API Documentation

## 🧮 Core Functionality

### `LinearRegressor` Class
```python
class LinearRegressor(
    lr: float = 0.01,       # Learning rate (default: 0.01)
    max_iter: int = 1000,    # Maximum training iterations
    threshold: float = 1e-5  # Early stopping threshold
)
```
**Methods:**
| Method | Description |
|--------|-------------|
| `.fit(X, y)` | Train model on feature array `X` and targets `y` |
| `.predict(X)` | Generate predictions for input `X` |
| `.plot_training()` | Display loss curve (requires matplotlib) |

**Example:**
```python
from salary_predictor import LinearRegressor
import numpy as np

X = np.array([1, 2, 3])
y = np.array([2, 4, 6])
model = LinearRegressor(lr=0.1)
model.fit(X, y)
model.plot_training()
```

## 🔧 Preprocessing

### `normalize()`
```python
def normalize(X: np.ndarray) -> np.ndarray:
    """Normalize data to zero mean, unit variance"""
```

**Usage:**
```python
from salary_predictor.preprocessing import normalize
X_normalized = normalize(raw_data)
```

## 📊 Visualization

### `plot_results()`
```python
def plot_results(
    X: np.ndarray, 
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> None:
    """Plot actual vs predicted values"""
```

## ⚠️ Common Exceptions
| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError` | Non-numeric input | Check data types |
| `NotFittedError` | Predict before fit | Call `.fit()` first |

## 📁 Data Requirements
- Input `X`: 1D numpy array (n_samples,)
- Target `y`: 1D numpy array (n_samples,)

---

> 🔍 **Pro Tip**: Run examples in the included Jupyter notebooks for interactive demos.
