import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

class Decision_tree():
    def __init__(self, df):
        self.df = df
        self.X = self.df[['Gender', 'Diabetes','Hipertension', 'Scholarship', 'SMS_received',
        'Handicap_0','Handicap_1','Handicap_2','Handicap_3','Handicap_4', 'Num_App_Missed', 'Age', 'AwaitingTime']]
        self.y = df["No-show"]
        self.X_train = pd.get_dummies(self.X)
        scaler = StandardScaler().fit(self.X_train)
        self.rescaledX2 = scaler.transform(self.X_train)

    def get_inital_input(self):
        return self.X_train
    
    def train_tree(self):
        X_train, X_test, y_train, y_test = train_test_split(self.rescaledX2, self.y, test_size=0.25, random_state = 42)
        tree = DecisionTreeClassifier(max_depth=10, random_state=0)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        return tree, X_train, X_test, y_train, y_test, y_pred