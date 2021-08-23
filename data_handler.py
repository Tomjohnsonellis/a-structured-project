import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def setup_dataset(csv_path, bmi_precision=1, price_precision=2, verbose=0):
    df = pd.read_csv(csv_path)
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
    y_train = torch.from_numpy(y_train).to(torch.float)
    y_test = torch.from_numpy(y_test).to(torch.float)

    return  X_train, X_test, y_train, y_test



if __name__ == '__main__':
    print("Usages:")
    print("X_train, X_test, y_train, y_test = setup_dataset('csv_path')")