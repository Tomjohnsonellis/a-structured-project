from sklearn import model_selection
from data_handler import setup_dataset
import torch
import torch.nn
import insurance_models
import optuna

X_train, X_test, y_train, y_test = setup_dataset("dataset/train.csv",verbose=0)

# Need to train and validate
current_model = insurance_models.InsurancePriceNN([128, 64])
current_model.train()

an_optimizer = torch.optim.SGD(current_model.parameters(), lr=0.1)
criterion = torch.nn.MSELoss()

print(X_train[0].size())
# print(y_train[0])
print(y_train[0].size())
print(X_train[0])
print(y_train[0])

an_optimizer.zero_grad()
model_guess = current_model.forward(X_train[0])
loss_value = criterion(model_guess, y_train[0])
print(loss_value)




