from sklearn import model_selection
from data_handler import setup_dataset, InsuranceDataset, create_loaders
import torch
import torch.nn
import insurance_models
import optuna

data_path = "dataset/train.csv"

# Load up the data in a usable form
current_model = insurance_models.InsurancePriceNN([32, 32])
train_loader, test_loader = create_loaders(data_path)
# Define an optimiser and loss funciton
an_optimizer = torch.optim.SGD(current_model.parameters(), lr=0.1)
criterion = torch.nn.L1Loss() # MSE is producing NaNs today :(

def validate_model(model, criterion, test_loader, best_loss):
    model.eval()
    with torch.no_grad():
        running_loss = 0
        for X_test, y_test in iter(test_loader):
            model_guess = model.forward(X_test)
            loss_value = criterion(model_guess, y_test)
            running_loss += loss_value
    
    if running_loss < best_loss:
        print("New best loss!")
        torch.save(model, "best_model.pt")
        file = open("best_loss", "w")
        file.write(str(float(running_loss)))
        best_loss = float(running_loss)
    
    return

num_epochs = 10
best_loss = float('inf')
file = open("best_loss", "r")
best_loss = file.read()
best_loss = float(best_loss)
file.close()

# Rough training loop
for epoch in range(num_epochs):
    running_loss = 0
    print(f"Epoch {epoch+1}")
    for X, y in iter(train_loader):
        X = X.reshape(32, 10)
        an_optimizer.zero_grad()
        model_guess = current_model.forward(X)
        loss_value = criterion(model_guess, y)
        loss_value.backward()
        an_optimizer.step()
        running_loss += loss_value.item()
    print(f"Loss: {running_loss}")

    # Validation during training
    validate_model(current_model, criterion, test_loader, best_loss)






