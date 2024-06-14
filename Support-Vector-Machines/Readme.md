# Support Vector Machines

## Classification

In machine learning, classification involves assigning instances to predefined groups. Examples include identifying whether an image contains a cat or dog, or determining the sentiment of a text as positive, negative, or neutral. The machine learns these patterns from labeled training data.

### Binary Classification

This type involves classifying instances into one of two classes (e.g., yes/no, true/false). Questions addressed in binary classification include:
- Does this image contain a human?
- Is this text positive?
- Will the stock price increase next month?

### Multiclass Classification

Here, instances are classified into one of three or more classes. Examples include:
- Classifying text as positive, negative, or neutral.
- Identifying the breed of a dog in an image.
- Categorizing news articles into sports, politics, economics, or social.

## Support Vector Machines (SVM)

SVM is a supervised learning algorithm used for classification and regression. It seeks to find the optimal boundary (hyperplane) between different classes.

### How SVM Works

SVM aims to maximize the margin between classes by transforming data using various kernel functions (Linear, Polynomial, Gaussian, RBF, Sigmoid). The support vectors are the data points closest to the hyperplane.

## Multiclass Classification Using SVM

SVM inherently supports binary classification. For multiclass problems, we use strategies to break down the problem into multiple binary classifications:

- **One-vs-One**: Constructs a binary classifier for each pair of classes, requiring \(\frac{m(m-1)}{2}\) SVMs for \(m\) classes.
- **One-vs-Rest**: Constructs a binary classifier for each class against all others, requiring \(m\) SVMs for \(m\) classes.

### Example

Consider a problem with three classes: green, red, and blue. In the One-vs-One approach, each class pair is separated by a hyperplane. In the One-vs-Rest approach, a hyperplane separates one class from all others.

## SVM Multiclass Classification in Python

The following Python example demonstrates training and testing an SVM classifier on the Iris dataset using Scikit-learn. Two kernel functions (Polynomial and RBF) are used to show performance differences.

### Setup

```python
from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score, f1_score
```

### Load Data

```python
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=101)
```

### Train Classifiers

```python
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
```

### Test Classifiers

```python
poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)
```

### Evaluate Performance

```python
poly_accuracy = accuracy_score(y_test, poly_pred)
poly_f1 = f1_score(y_test, poly_pred, average='weighted')
print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy * 100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1 * 100))

rbf_accuracy = accuracy_score(y_test, rbf_pred)
rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy * 100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1 * 100))
```

### Results

```
Accuracy (Polynomial Kernel): 70.00
F1 (Polynomial Kernel): 69.67
Accuracy (RBF Kernel): 76.67
F1 (RBF Kernel): 76.36
```

## Conclusion

This tutorial introduced classification, SVM, and their application in multiclass classification. We provided a Python example demonstrating the use of SVM with different kernels on the Iris dataset. The RBF kernel outperformed the Polynomial kernel in this case. Experiment with hyperparameters like C, gamma, and degree to optimize performance for your specific dataset.
