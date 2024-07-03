from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Pre-process


class Pre_process():
    def __init__(self, df):
        self.df = df

    def preprocess(self):
        self.df['No-show'].replace("No", 0,inplace=True)
        self.df['No-show'].replace("Yes", 1,inplace=True) 
        #Convert to Categorical
        self.df['Handcap'] = pd.Categorical(self.df['Handcap'])
        #Convert to Dummy Variables
        Handicap = pd.get_dummies(self.df['Handcap'], prefix = 'Handicap')
        self.df = pd.concat([self.df, Handicap], axis=1)
        self.df = self.df[(self.df.Age >= 0) & (self.df.Age <= 100)]


    def convert_datetime(self):
        # Converts the two variables to datetime variables
        self.df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
        self.df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])


    def calc_waiting_time(self):
        # Create a variable called "AwaitingTime" by subtracting the date the patient made the appointment and the date of the appointment.
        self.df['AwaitingTime'] = self.df["AppointmentDay"].sub(self.df["ScheduledDay"], axis=0)
        # Convert the result "AwaitingTime" to number of days between appointment day and scheduled day. 
        self.df["AwaitingTime"] = (self.df["AwaitingTime"] / np.timedelta64(1, 'D')).abs()


    def calc_no_shows(self):
        # Number of Appointments Missed by Patient
        new_col = self.df.groupby('PatientId')['No-show'].apply(lambda x: x.cumsum())
        self.df["Num_App_Missed"] = new_col.reset_index(level=0, drop=True)
    
    def format_and_get_df(self):
        self.preprocess()
        self.convert_datetime()
        self.calc_waiting_time()
        self.calc_no_shows()

# Logistic model
    
class Logictic_model():
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
    
    def log_reg(self):
        X_train, X_test, y_train, y_test = train_test_split(self.rescaledX2, self.y, test_size=0.25)
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        return y_test,y_pred

# Decision tree

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
    
# Analysis

df = pd.read_csv("KaggleV2-May-2016.csv")

pre_processer = Pre_process(df)
df_proc = pre_processer.format_and_get_df()

log_reg = Logictic_model(df_proc)

l_y_test,l_y_pred = log_reg.log_reg()

print("Logistic regression")
print("Results:")
print("Accuracy", metrics.accuracy_score(l_y_test,l_y_pred))

# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(l_y_test, l_y_pred)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

#Specificity: When the actual value is negative, how often is the prediction correct?
print("Specificity:",TN / float(TN + FP))

#False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
print("False Positive Rate:",FP / float(TN + FP))

#Precision: When a positive value is predicted, how often is the prediction correct?
print("Precision:",metrics.precision_score(l_y_test, l_y_pred))

#Sensitivity:
print("Recall:",metrics.recall_score(l_y_test, l_y_pred))

print("Decision tree")
dev_tree = Decision_tree(df_proc)

tree, t_X_train, t_X_test, t_y_train, t_y_test, t_y_pred = dev_tree.train_tree()

print('Accuracy on the training subset: {:.3f}'.format(tree.score(t_X_train, t_y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(t_X_test, t_y_test)))


# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(t_y_test, t_y_pred)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

#Specificity: When the actual value is negative, how often is the prediction correct?
print("Specificity:",TN / float(TN + FP))

#False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
print("False Positive Rate:",FP / float(TN + FP))

#Precision: When a positive value is predicted, how often is the prediction correct?
print("Precision:",metrics.precision_score(t_y_test, t_y_pred))

#Sensitivity:
print("Recall:",metrics.recall_score(t_y_test, t_y_pred))


X_train = dev_tree.get_inital_input()

n_features = X_train.shape[1]
plt.barh(range(n_features), tree.feature_importances_, )
plt.yticks(np.arange(n_features), X_train)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()
