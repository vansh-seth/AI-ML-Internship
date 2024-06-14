## Neural Network Models (Supervised)

### Multi-layer Perceptron (MLP)

**Overview:**
Multi-layer Perceptron (MLP) is a supervised learning algorithm capable of approximating complex functions through training on input-output datasets. It differs from logistic regression by incorporating one or more non-linear hidden layers between input and output layers, enabling it to learn non-linear relationships.

**Structure:**
- **Input Layer:** Neurons representing input features.
- **Hidden Layers:** Each neuron applies weighted summation followed by a non-linear activation (e.g., tanh function).
- **Output Layer:** Transforms hidden layer outputs into final predictions.

![image](https://github.com/vansh-seth/AI-ML-Internship/assets/111755254/681aeeb0-b587-4820-8cae-7016c62ccdd2)


### MLPClassifier (Classification)

**Functionality:**
MLPClassifier trains on:
- `X`: Training samples (n_samples, n_features).
- `y`: Target values (class labels) for training samples.

**Example:**
```python
from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)
```

### MLPRegressor (Regression)

**Functionality:**
MLPRegressor performs multi-output regression using backpropagation. It minimizes square error loss function for continuous value predictions.

**Example:**
```python
from sklearn.neural_network import MLPRegressor
X = [[0., 0.], [2., 2.]]
y = [0.5, 2.5]
clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)
```

### Regularization

**Overview:**
Both MLPRegressor and MLPClassifier use `alpha` for L2 regularization to prevent overfitting by penalizing large weights.

**Example:**
```python
clf = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(10,), random_state=1)
```

### Algorithms

**Training Methods:**
MLP supports training via Stochastic Gradient Descent (SGD), Adam optimizer, or L-BFGS (Limited-memory BFGS).

**Example:**
```python
clf = MLPClassifier(solver='sgd', learning_rate='adaptive', max_iter=1000)
```

### Complexity

**Time Complexity:**
For MLP with n_samples, n_features, n_hidden layers, and n_output neurons, backpropagation complexity is O(iterations * (n_samples * n_features * n_hidden_layers * n_output_neurons)).

**Recommendation:** Start with fewer hidden neurons and layers to manage training complexity effectively.


