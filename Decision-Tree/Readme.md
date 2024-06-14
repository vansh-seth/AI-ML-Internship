# Decision Trees

Decision Trees are a powerful, non-parametric supervised learning method used for both classification and regression tasks. The goal of a decision tree is to create a model that predicts the value of a target variable by learning simple decision rules from the data features. This method can be seen as a piecewise constant approximation.

## Key Advantages

1. **Interpretability**: Decision trees are simple to understand and interpret. They can be visualized, making them easy to explain to non-experts.
2. **Minimal Data Preparation**: Requires little data preparation compared to other techniques. For instance, data normalization or creation of dummy variables is often unnecessary.
3. **Efficiency**: The cost of making predictions is logarithmic in the number of data points used to train the tree.
4. **Versatility**: Can handle both numerical and categorical data, and multi-output problems.
5. **Transparency**: Uses a white box model, where the results are easily interpretable through boolean logic.
6. **Validation**: Models can be validated using statistical tests, making it possible to account for the reliability of the model.
7. **Robustness**: Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.

## Key Disadvantages

1. **Overfitting**: Decision trees can create over-complex trees that do not generalize well to unseen data. Techniques such as pruning, setting a minimum number of samples per leaf, or setting maximum tree depth are necessary to mitigate this.
2. **Instability**: Small variations in the data can result in a completely different tree. This can be addressed by using ensemble methods.
3. **Piecewise Constant Predictions**: Predictions are not smooth or continuous, making them poor at extrapolation.
4. **Complexity**: Learning an optimal decision tree is NP-complete under several aspects of optimality. Heuristic algorithms like the greedy algorithm are used in practice.
5. **Bias**: Trees can be biased if some classes dominate. Balancing the dataset prior to fitting is recommended.

## Classification with Decision Trees

### Example Code

```python
from sklearn import tree

# Sample data
X = [[0, 0], [1, 1]]
Y = [0, 1]

# Initialize and train classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

# Predict
print(clf.predict([[2., 2.]]))  # Output: [1]

# Predict probabilities
print(clf.predict_proba([[2., 2.]]))  # Output: [[0., 1.]]
```

### Using the Iris Dataset

```python
from sklearn.datasets import load_iris
from sklearn import tree

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Initialize and train classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# Plot tree
tree.plot_tree(clf)
```

### Exporting Tree Visualization

```python
import graphviz

# Export in Graphviz format
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")
```

### Exporting Tree as Text

```python
from sklearn.tree import export_text

# Export as text
r = export_text(clf, feature_names=iris['feature_names'])
print(r)
```

## Regression with Decision Trees

### Example Code

```python
from sklearn import tree

# Sample data
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]

# Initialize and train regressor
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)

# Predict
print(clf.predict([[1, 1]]))  # Output: [0.5]
```

## Multi-output Problems

### Handling Multiple Outputs

```python
# If Y is a 2D array of shape (n_samples, n_outputs), the resulting estimator
# will predict multiple outputs simultaneously.
```

## Computational Complexity

The run-time cost to construct a balanced binary tree is \(O(n \log(n))\) and the query time is \(O(\log(n))\). The tree construction involves searching through the features to find the one that offers the largest reduction in impurity, leading to a total cost of \(O(n \log(n) d)\), where \(d\) is the number of features.

## References

- M. Dumont et al., "Fast multi-class image annotation with random subwindows and multiple output randomized trees," International Conference on Computer Vision Theory and Applications, 2009.
