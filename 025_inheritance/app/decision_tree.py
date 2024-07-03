from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from app.generic_model import Basic_model

class Decision_tree(Basic_model):
    def __init__(self, df):
        super().__init__(df)
    
    def train_tree(self):
        X_train, X_test, y_train, y_test = train_test_split(self.rescaledX2, self.y, test_size=0.25, random_state = 42)
        tree = DecisionTreeClassifier(max_depth=10, random_state=0)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        return tree, X_train, X_test, y_train, y_test, y_pred