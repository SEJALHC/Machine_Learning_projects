import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def Advertising (datapath):
    df = pd.read_csv(datapath)
    print("Data Loaded Successfully")
    print(df.head())
    
    print("Clean the dataset :")
    df.drop(columns=['Unnamed: 0'],inplace= True)
    print(df.head())
    
    x = df.drop(columns=['sales'])
    y = df['sales']
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=42)
    
    model = LinearRegression()
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    mse = metrics.mean_squared_error(y_test,y_pred)
    
    rmse = np.sqrt(mse)
    
    r2 = metrics.r2_score(y_test,y_pred)
    
    print("Mean Squared Error is :",mse)
    print("Root mean Squared Error is:",rmse)
    print("R Square value :",r2)
    
    
def main():
    Advertising("Advertising.csv")

if __name__ =="__main__":
    main()