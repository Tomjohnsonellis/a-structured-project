import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def setup_dataset(csv_path, bmi_precision=1, price_precision=2, verbose=0):
    df = pd.read_csv(csv_path)
    # NAs can leave
    df.dropna()
    # One hot encode sex and region
    OHE_sex = pd.get_dummies(df['sex'], prefix='sex')
    OHE_region = pd.get_dummies(df['region'], prefix='region')
    df = pd.merge(
        df,OHE_sex,left_index=True,right_index=True
    )
    df = pd.merge(
        df,OHE_region,left_index=True,right_index=True
    )
    df = df.drop(['sex','region'], axis=1)

    # Binary encode smoker
    df['smoker'] = df['smoker'].map(dict(yes=1,no=0))

    # Round BMI to 1dp
    df['bmi'] = df['bmi'].round(bmi_precision)

    # Seperate target from data
    target = df['charges'].round(price_precision).values
    df = df.drop(['charges'], axis=1)
    # print(target)

    # Data
    data = df.values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    if verbose:
        print(f"Dimensions:\nX_train:{X_train.shape}\ny_train:{y_train.shape}\n","="*50,f"\nX_test:{X_test.shape}\ny_test{y_test.shape}")

    # Convert to tensors
    X_train = torch.from_numpy(X_train).to(torch.float)
    X_test = torch.from_numpy(X_test).to(torch.float)
    # reshape -1 is a good tool to remember
    y_train = torch.from_numpy(y_train).to(torch.float).reshape(-1,1)
    y_test = torch.from_numpy(y_test).to(torch.float).reshape(-1,1)

    return  X_train, X_test, y_train, y_test


class InsuranceDataset(Dataset):
    def __init__(self, csv_path, train=True):
        self.X_train, self.X_test, self.y_train, self.y_test = setup_dataset(csv_path)
        self.training_samples = len(self.X_train)
        self.test_samples = len(self.X_test)
        if train:
            self.X = self.X_train
            self.y = self.y_train
            self.n_samples = self.training_samples
        else:
            self.X = self.X_test
            self.y = self.y_test
            self.n_samples = self.test_samples

    def __gettrainlen__(self):
        return self.training_samples
    def __gettestlen__(self):
        return self.test_samples
    def __gettrainitem__(self, index):
        return self.X_train[index], self.y_train[index]
    def __gettestitem__(self, index):
        return self.X_test[index], self.y_test[index]
    def __len__(self):
        return self.n_samples
    def __getitem__(self, index):
        return self.X[index], self.y[index]


def create_loaders(data_path):
    train_data = InsuranceDataset(data_path)
    test_data = InsuranceDataset(data_path, train=False)
    train_loader = DataLoader(dataset=train_data, batch_size=32, drop_last=True)
    # train_iter = iter(train_loader)
    test_loader = DataLoader(dataset=test_data, batch_size=32, drop_last=True)
    # test_iter = iter(test_loader)
    return train_loader, test_loader





if __name__ == '__main__':
    print("Usages:")
    print("X_train, X_test, y_train, y_test = setup_dataset('csv_path')")
    X_train, X_test, y_train, y_test = setup_dataset("dataset/train.csv",verbose=0)


    object = InsuranceDataset("dataset/insurance.csv")
    a_sample, a_label = object.__gettrainitem__(4)
    print(a_sample.size(), a_label.size())
    sample_size = object.__gettrainlen__()
    print(sample_size)
