import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

C = 1.0
clf = svm.LinearSVC(C=1.0, max_iter=10000)
clf.fit(X, y)

fig, ax = plt.subplots()
disp = DecisionBoundaryDisplay.from_estimator(clf,X,response_method="predict",ax=ax,xlabel=iris.feature_names[0],ylabel=iris.feature_names[1])
X0, X1 = X[:, 0], X[:, 1]
ax.scatter(X0, X1, c=y, edgecolors="k")
ax.set_title("LinearSVC")

plt.show()
