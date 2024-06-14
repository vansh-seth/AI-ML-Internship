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

![image](https://github.com/vansh-seth/AI-ML-Internship/assets/111755254/d5a31fdc-cdac-4f43-9499-c271aa8db294)


## Multiclass Classification Using SVM

SVM inherently supports binary classification. For multiclass problems, we use strategies to break down the problem into multiple binary classifications:

- **One-vs-One**: Constructs a binary classifier for each pair of classes, requiring \(\frac{m(m-1)}{2}\) SVMs for \(m\) classes.

![image](https://github.com/vansh-seth/AI-ML-Internship/assets/111755254/81697343-1421-45ee-b094-30e12372213d)

- **One-vs-Rest**: Constructs a binary classifier for each class against all others, requiring \(m\) SVMs for \(m\) classes.

![image](https://github.com/vansh-seth/AI-ML-Internship/assets/111755254/4bb9fb1a-57b5-47dd-a90b-7610cbcbcfa1)

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

### Support Vector Machines

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression, and outlier detection.

#### Advantages of Support Vector Machines:
- **Effective in high dimensional spaces.**
- **Still effective in cases where the number of dimensions is greater than the number of samples.**
- **Uses a subset of training points in the decision function (called support vectors), making it memory efficient.**
- **Versatile:** Different kernel functions can be specified for the decision function. Common kernels are provided, but custom kernels can also be specified.

#### Disadvantages of Support Vector Machines:
- **Overfitting Risk:** If the number of features is much greater than the number of samples, avoiding overfitting in choosing kernel functions and the regularization term is crucial.
- **Probability Estimates:** SVMs do not directly provide probability estimates; these are calculated using an expensive five-fold cross-validation.

SVMs in scikit-learn support both dense (numpy.ndarray) and sparse (scipy.sparse) sample vectors as input. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.

#### Classification
`SVC`, `NuSVC`, and `LinearSVC` are classes capable of performing binary and multi-class classification on a dataset.

![image](https://github.com/vansh-seth/AI-ML-Internship/assets/111755254/d2b3663c-0600-444c-8855-bff88bc14b34)

`SVC` and `NuSVC` are similar methods but accept slightly different sets of parameters and have different mathematical formulations. `LinearSVC` is a faster implementation of Support Vector Classification for the case of a linear kernel and uses `squared_hinge` loss. 

Example:
```python
from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
clf.predict([[2., 2.]])
```

#### Multi-class Classification
`SVC` and `NuSVC` implement the “one-versus-one” approach for multi-class classification. `LinearSVC` implements the “one-vs-the-rest” strategy, thus training `n_classes` models.

Example:
```python
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y)
clf.decision_function([[1]]).shape[1]
```

#### Scores and Probabilities
The `decision_function` method of `SVC` and `NuSVC` gives per-class scores for each sample. When `probability=True`, class membership probability estimates are enabled. The probabilities are calibrated using Platt scaling.

#### Unbalanced Problems
In problems where it is desired to give more importance to certain classes or samples, the parameters `class_weight` and `sample_weight` can be used. `SVC` implements `class_weight` in the fit method, and all SVM variants implement `sample_weight`.

![image](https://github.com/vansh-seth/AI-ML-Internship/assets/111755254/eba9c290-c76e-4f77-b251-44c4fc69bdd0)

#### Regression
Support Vector Regression (SVR) extends support vector classification to solve regression problems. There are three different implementations of SVR: `SVR`, `NuSVR`, and `LinearSVR`.

Example:
```python
from sklearn import svm
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
regr = svm.SVR()
regr.fit(X, y)
regr.predict([[1, 1]])
```

#### Density Estimation and Novelty Detection
The class `OneClassSVM` implements a One-Class SVM used in outlier detection.

#### Complexity
SVMs have high computational and memory requirements. The core of an SVM is a quadratic programming problem (QP), separating support vectors from the rest of the training data. The algorithm in `LinearSVC` is more efficient and can scale almost linearly to millions of samples and/or features.

#### Practical Tips
- **Avoiding Data Copy:** Ensure data is C-ordered and double precision for optimal performance.
- **Kernel Cache Size:** Increasing the kernel cache size can improve run times.
- **Setting C:** A lower C value can help with noisy observations.
- **Data Scaling:** Scale your data to improve performance and ensure meaningful results.
- **Randomness Control:** Use `random_state` to control the randomness in `SVC`, `NuSVC`, and `LinearSVC`.

#### Kernel Functions
SVMs can use various kernel functions:
- **Linear:** 
- **Polynomial:** 
- **RBF (Gaussian):** 
- **Sigmoid:** 

Example:
```python
linear_svc = svm.SVC(kernel='linear')
rbf_svc = svm.SVC(kernel='rbf')
```

#### Custom Kernels
You can define custom kernels by passing a function to the `kernel` parameter or using pre-computed kernels.

Example:
```python
import numpy as np
from sklearn import svm

def my_kernel(X, Y):
    return np.dot(X, Y.T)

clf = svm.SVC(kernel=my_kernel)
```
