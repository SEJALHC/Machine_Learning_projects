import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def Diabetes(datapath):
    df = pd.read_csv(datapath)
    print("Data loaded Successfully")
    
    print(df.head())
    
    print("Null Values in Data:\n", df.isnull().sum())

    print ("Statistics Data : ")
    print(df.describe())
    
    df.hist(figsize=(10,5))
    plt.tight_layout()
    plt.show()
    
    missing= ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[missing] = df[missing].replace(0, np.nan)

    df.dropna(inplace=True)
    
    
    x = df.drop(columns=['Outcome'])
    y =df['Outcome']
    
    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)
    
    logisticRegression(x_scale,y)
    
def logisticRegression(x_scale,y):
    
    x_train, x_test, y_train, y_test = train_test_split(x_scale,y, test_size=0.2,random_state=42)
    model = LogisticRegression()
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    print("Accuracy is :",accuracy_score(y_test,y_pred)*100)
    
    cm = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix :",cm)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
    results.to_csv('LogisticRegressionPredictionsOutput.csv', index=False)
    print("Result Saved in LogisticRegressionPredictionsOutput.csv")
    
    DecisionTree(x_scale,y)
    
    
def DecisionTree(x_scale,y):
    
    x_train, x_test, y_train, y_test = train_test_split(x_scale,y, test_size=0.2,random_state=42)
    
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    print("Accuracy is :",accuracy_score(y_test,y_pred)*100)
    
    cm = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix :",cm)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
    results.to_csv('DecisionTreeClassifierPredictionsOutput.csv', index=False)
    print("Result Saved in DecisionTreeClassifierPredictionsOutput.csv")

def main():
    Diabetes("diabetes.csv")
    
    

if __name__ == "__main__":
    main()