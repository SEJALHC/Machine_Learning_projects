import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier


def BreastCancerCaseStudy ():
    
    load_data = load_breast_cancer()
    
    df = pd.DataFrame(load_data.data, columns=load_data.feature_names)
    df['target'] = load_data.target
    
    print("Data loaded Successfully :")
    
    print(df.head())
    
    print("Statistical Summary:")
    print(df.describe())
    
    print("Null Values in Data:\n", df.isnull().sum())
    
    x = df.drop(columns=['target'])
    y = df['target']
    
    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)
    
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), cmap='coolwarm',annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.show()
    
    x_train, x_test, y_train, y_test = train_test_split(x_scale,y, test_size=0.2,random_state=42)
    
    model = DecisionTreeClassifier(max_depth=7)
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    y_proba = model.predict_proba(x_test)[:, 1]
    
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    print("DecisionTree Classifier Accuracy is :",accuracy_score(y_test,y_pred)*100)
    
    cm = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix :",cm)
    
    roc_auc = roc_auc_score(y_test,y_proba)
    print ("ROC-AUC:", roc_auc*100)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("DecisionTree Classifier Confusion Matrix ")
    plt.show() 
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange',label=f'ROC curve (AUC = {roc_auc_score(y_test, y_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Decision Tree Classifier')
    plt.legend()
    plt.grid(True)
    plt.show()
    importance = pd.Series(model.feature_importances_,index=x.columns)
    importance = importance.sort_values(ascending=False)
    
    importance.plot(kind='bar', figsize=(10,7), title="Features Importance")
    plt.show()

def main():
    BreastCancerCaseStudy()

if __name__ == "__main__":
    main()