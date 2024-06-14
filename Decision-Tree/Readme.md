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

## Computational Complexity

The run-time cost to construct a balanced binary tree is \(O(n \log(n))\) and the query time is \(O(\log(n))\). The tree construction involves searching through the features to find the one that offers the largest reduction in impurity, leading to a total cost of \(O(n \log(n) d)\), where \(d\) is the number of features.


## Code Output:

![image](https://github.com/vansh-seth/AI-ML-Internship/assets/111755254/98755e1e-484c-4e2e-bf76-2fcd1e1e6d80)

# Decision Tree Result Explained

A decision tree uses previous decisions to calculate the odds of whether you want to go see a comedian or not. Let’s break down the different aspects of the decision tree.

## Rank

### Root Node
- **Rank <= 6.5**: Comedians with a rank of 6.5 or lower follow the True arrow (left), while those with a rank higher than 6.5 follow the False arrow (right).
- **gini = 0.497**: Indicates the quality of the split. A value between 0.0 (perfect split) and 0.5 (no information gain).
- **samples = 13**: Number of comedians considered at this step (all comedians).
- **value = [6, 7]**: Among the 13 comedians, 6 will get a "NO" and 7 will get a "GO".

### Gini Calculation
The Gini impurity for this split is calculated as:
\[ \text{Gini} = 1 - \left(\frac{x}{n}\right)^2 - \left(\frac{y}{n}\right)^2 \]
Where:
- \( x \) = number of "GO" answers = 7
- \( y \) = number of "NO" answers = 6
- \( n \) = total number of samples = 13

\[ \text{Gini} = 1 - \left(\frac{7}{13}\right)^2 - \left(\frac{6}{13}\right)^2 = 0.497 \]

## First Split

### Left Branch (Rank <= 6.5)
- **gini = 0.0**: All samples have the same result.
- **samples = 5**: Number of comedians in this branch.
- **value = [5, 0]**: All 5 comedians get a "NO".

### Right Branch (Rank > 6.5)
- **gini = 0.219**: Indicates a good quality split.
- **samples = 8**: Number of comedians in this branch.
- **value = [1, 7]**: Among these 8 comedians, 1 gets a "NO" and 7 get a "GO".

## Second Split (Right Branch, Rank > 6.5)

### Left Branch (Nationality <= 0.5, e.g., UK)
- **gini = 0.375**: Indicates the quality of the split.
- **samples = 4**: Number of comedians from the UK.
- **value = [1, 3]**: Among these 4 comedians, 1 gets a "NO" and 3 get a "GO".

### Right Branch (Nationality > 0.5)
- **gini = 0.0**: All samples have the same result.
- **samples = 4**: Number of non-UK comedians.
- **value = [0, 4]**: All 4 comedians get a "GO".

## Third Split (Left Branch, Nationality <= 0.5)

### Left Branch (Age <= 35.5)
- **gini = 0.0**: All samples have the same result.
- **samples = 2**: Number of comedians aged 35.5 or younger.
- **value = [0, 2]**: Both comedians get a "GO".

### Right Branch (Age > 35.5)
- **gini = 0.5**: Split quality indicates equal distribution.
- **samples = 2**: Number of comedians older than 35.5.
- **value = [1, 1]**: One comedian gets a "NO" and one gets a "GO".

## Fourth Split (Right Branch, Age > 35.5)

### Left Branch (Experience <= 9.5)
- **gini = 0.0**: All samples have the same result.
- **samples = 1**: One comedian with 9.5 years of experience or less.
- **value = [0, 1]**: This comedian gets a "GO".

### Right Branch (Experience > 9.5)
- **gini = 0.0**: All samples have the same result.
- **samples = 1**: One comedian with more than 9.5 years of experience.
- **value = [1, 0]**: This comedian gets a "NO".

## Summary

- The decision tree helps determine whether you would go see a comedian based on their rank, nationality, age, and experience.
- Each split represents a decision point that filters comedians into "GO" or "NO" categories.
- The Gini index measures the purity of each split, aiming for values closer to 0 for pure nodes.
- The tree's structure allows for clear and interpretable decision rules based on the given criteria.
