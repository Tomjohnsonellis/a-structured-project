from data_handler import setup_dataset
import insurance_models


X_train, X_test, y_train, y_test = setup_dataset("dataset/insurance.csv",verbose=1)

# Need to train and validate
model = insurance_models.InsurancePriceNN([128, 64])
test = model.forward(X_train[20])
print(test)