# Shrink vehicles.csv to only the columns we need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def shrink_vehicles_csv():
    # Importing the dataset
    dataset = pd.read_csv('vehicles.csv')
    # Cleaning the dataset
    dataset = dataset[dataset['price'] != 0]
    dataset = dataset[dataset['odometer'].notnull()]
    dataset = dataset[dataset['price'].notnull()]

    # Remove outliers
    dataset = dataset[dataset['odometer'] < 500000]
    dataset = dataset[dataset['price'] < 100000]
    dataset = dataset[dataset['price'] > 1000]
    # Remove any car with 0 miles under 10000 dollars
    dataset = dataset[~((dataset['price'] < 10000) & (dataset['odometer'] < 1235))]

    # Shrink dataset to only the columns we need
    dataset = dataset[['manufacturer', 'model', 'year', 'odometer', 'price']]

    # Export dataset to csv
    dataset.to_csv('vehicles_shrunk.csv', index=False)

shrink_vehicles_csv()