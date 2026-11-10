import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class walmartlinearmodel(nn.Module):
    def __init__(self,num_features:int):
        super().__init__()
        self.linear = nn.Linear(in_features = num_features,out_features = 1)
    def forward(self, x:torch.Tensor):
        return self.linear(x)

class Linearmodel:
    def __init__(self,num_features : int, lr:float=0.01):
        self.model = walmartlinearmodel(num_features)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def fit(self,X_df:pd.DataFrame, y_df:pd.DataFrame,epochs: int = 300):

        X_scaled = self.x_scaler.fit_transform(X_df)
        y_scaled = self.y_scaler.fit_transform(y_df)

        X_tensor = torch.tensor(X_scaled, dtype = torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype = torch.float32)

        self.model.train()

        for epoch in range(epochs):
            predictions = self.model(X_tensor)
            loss = self.criterion(predictions,y_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self,x_new_df:pd.DataFrame):
        self.model.eval()

        X_new_scaled = self.x_scaler.transform(x_new_df)
        X_tensor = torch.tensor(X_new_scaled,dtype = torch.float32)

        with torch.no_grad():
            pred_scaled = self.model(X_tensor).numpy()

        pred_final = self.y_scaler.inverse_transform(pred_scaled)
        return float(pred_final[0][0])


if __name__ == "__main__":
    df = pd.read_csv("Walmart_Sales.csv")

    features_columns = [
                "Store",
                "Holiday_Flag",
                "Temperature",
                "Fuel_Price",
                "CPI",
                "Unemployment"
            ]
    target_column = "Weekly_Sales"

    X = df[features_columns]
    y = df[[target_column]]

    predictor = Linearmodel(num_features=len(features_columns),lr = 0.02)

    print("---Training started---")
    predictor.fit(X,y,epochs=300)
    print("---Traning Done---")

    sample_data = pd.DataFrame([{
                "Store": 1,
                "Holiday_Flag": 0,
                "Temperature": 42.31,
                "Fuel_Price": 2.572,
                "CPI": 211.096,
                "Unemployment": 8.106
    }])

    predicted_sales = predictor.predict(sample_data)
    print(predicted_sales)
