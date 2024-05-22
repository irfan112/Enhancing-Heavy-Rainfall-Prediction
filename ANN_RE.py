import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from ANN_RE_model import ANN
from ANN_pre_model import ANN_2

    
class ann_re:
    def __init__(self, df):
        self.df = df
    
    def training_ANN_RE(self):
        
        print("ANN for Random Error mitigation")
        data_training = self.df.iloc[:,3:].copy()
        data_training_ = data_training.replace('--', 0)
        data_training__ = data_training_.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
        data_training__ = data_training__.dropna()
        
        data_training_cleaned = data_training__.copy()
        
        X = data_training_cleaned[['Precipitation', 'HQprecipitation', 'IRprecipitation', 'randomError']]
        y = data_training_cleaned[['Precipitation', 'HQprecipitation', 'IRprecipitation']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = MinMaxScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)
        X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
        
        model = ANN()
        print("ANN Model for Random Error mitigation")
        print("Model : ", model)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 10
        batch_size = 32
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, len(X_train_tensor), batch_size):
                optimizer.zero_grad()
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss/len(X_train_tensor):.4f}")
            
        with torch.no_grad():
            y_pred = model(X_test_tensor)
            test_loss = criterion(y_pred, y_test_tensor)
            print("Test Loss:", test_loss.item())
            
        torch.save(model.state_dict(), "ANN_RE_model.pt")
        print("Model Saved! named: ANN_RE_model.pt")

    
class ann_pre:
    def __init__(self, df):
        self.df = df
        
    def taining_ANN_Pre(self):
        data_training = self.df.iloc[:,3:].copy()
        data_training_ = data_training.replace('--', 0)
        data_training__ = data_training_.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
        data_training__ = data_training__.dropna()
        
        data_training_cleaned = data_training__.copy() 
        data2 = data_training_cleaned[['Precipitation', 'HQprecipitation', 'IRprecipitation']]
        X1 = data_training_cleaned[['Precipitation', 'HQprecipitation', 'IRprecipitation']]
        y1 = data_training_cleaned[['Precipitation']]  
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42) 
        scaler = MinMaxScaler()
        X_train_normalized1 = scaler.fit_transform(X_train1)
        X_test_normalized1 = scaler.transform(X_test1)
        
        X_train_tensor1 = torch.tensor(X_train_normalized1, dtype=torch.float32)
        X_test_tensor1 = torch.tensor(X_test_normalized1, dtype=torch.float32)
        y_train_tensor1 = torch.tensor(y_train1.values, dtype=torch.float32)
        y_test_tensor1 = torch.tensor(y_test1.values, dtype=torch.float32)
        
        model2 = ANN_2()

        criterion = nn.MSELoss()
        optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

        epochs = 10
        batch_size = 16
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, len(X_train_tensor1), batch_size):
                optimizer2.zero_grad()
                batch_X1 = X_train_tensor1[i:i+batch_size]
                batch_y1 = y_train_tensor1[i:i+batch_size]
                outputs1 = model2(batch_X1)
                loss1 = criterion(outputs1, batch_y1)
                loss1.backward()
                nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
                optimizer2.step()
                epoch_loss += loss1.item()
                
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss/len(X_train_tensor1):.4f}")
            
        with torch.no_grad():
            y_pred1 = model2(X_test_tensor1)
            test_loss1 = criterion(y_pred1, y_test_tensor1)
            print("Test Loss:", test_loss1.item())
            
        torch.save(model2.state_dict(), "ANN_pre_model.pt")
        