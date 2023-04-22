from collections import defaultdict
import numpy as np
from decision_tree import DecisionTree
import random as rand

class RandomForest:
    def __init__(self, params):
        self.forest = []
        self.params = defaultdict(lambda: None, params)


    def train(self, X, y):
        for _ in range(self.params["ntrees"]):
            X_bagging, y_bagging = self.bagging(X,y)
            tree = DecisionTree(self.params)
            tree.train(X_bagging, y_bagging)
            self.forest.append(tree)

    def evaluate(self, X, y):
        predicted = self.predict(X)
        predicted = [round(p) for p in predicted]
        print(f"Accuracy: {round(np.mean(predicted==y),2)}")

    def predict(self, X):
        tree_predictions = []
        for tree in self.forest:
            tree_predictions.append(tree.predict(X))
        forest_predictions = list(map(lambda x: sum(x)/len(x), zip(*tree_predictions)))
        return forest_predictions

    def bagging(self, X, y):
        X_selected, y_selected = None, None
        # TODO implement bagging
        x_rand = []
        y_rand = []
        for i in range(X.shape[1]):
            i_rand = rand.randint(0, X.shape[1] - 1)
            x_rand.append(X[i_rand])
            y_rand.append(y[i_rand])
            
        X_selected = np.array(x_rand)
        y_selected = np.array(y_rand)

        return X_selected, y_selected
