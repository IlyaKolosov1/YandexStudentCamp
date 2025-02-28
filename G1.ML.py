import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop(['target'], axis=1)
y = train['target']

cat_features = ['B']
num_features = ['A', 'C', 'D', 'E', 'F']

encoder = OneHotEncoder(
    drop='first', 
    sparse_output=False, 
    handle_unknown='ignore'
)

encoded_cat = encoder.fit_transform(X[cat_features])
X_numeric = X[num_features].values
X_combined = np.hstack([X_numeric, encoded_cat])  

encoded_cat_test = encoder.transform(test[cat_features])
X_numeric_test = test[num_features].values
X_combined_test = np.hstack([X_numeric_test, encoded_cat_test])

X_train, X_val, y_train, y_val = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

class ComplexNN(nn.Module):
    def __init__(self, input_dim):
        super(ComplexNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = ComplexNN(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val.values,   dtype=torch.float32).view(-1, 1)

num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_t)
    loss = criterion(y_pred, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_loss = criterion(val_preds, y_val_t)
    
    scheduler.step()

    if (epoch + 1) % 100 == 0:
        print(f"Эпоха [{epoch+1}/{num_epochs}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

val_preds = model(X_val_t).detach().numpy().squeeze()
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {rmse:.4f}")

X_test_t = torch.tensor(X_combined_test, dtype=torch.float32)
test_preds = model(X_test_t).detach().numpy().squeeze()

submission = pd.DataFrame({
    'A': test['A'],
    'B': test['B'],
    'C': test['C'],
    'D': test['D'],
    'E': test['E'],
    'F': test['F'],
    'target': test_preds
})

submission.to_csv('submission.csv', index=False)
