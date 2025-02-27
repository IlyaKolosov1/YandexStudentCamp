import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('field_train.csv')
example_data = pd.read_csv('field_example.csv')

X_train = train_data[['longitude', 'latitude']].values
y_train = train_data['intensity'].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)

class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.model(x)

model = GNet()
criter = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criter(outputs, y_train_tensor.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Эпоха [{epoch+1}/{num_epochs}], Потери: {loss.item():.4f}')

X_pred = example_data[['longitude', 'latitude']].values
X_pred_scaled = scaler_X.transform(X_pred)
X_pred_tensor = torch.FloatTensor(X_pred_scaled)

model.eval()
with torch.no_grad():
    predictions_scaled = model(X_pred_tensor).numpy()
predictions = scaler_y.inverse_transform(predictions_scaled)

answers = example_data.copy()
answers['intensity'] = predictions

answers.to_csv('answers.csv', index=False)