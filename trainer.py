from datahandler import setup_dataset

X_train, X_test, y_train, y_test = setup_dataset("dataset/insurance.csv")

print(X_test)