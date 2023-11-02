import numpy as np
import matplotlib.pyplot as plt
import pandas
from gradient_descent import gradientDescent
from gradient_descent import predict

def main():
    df = pandas.read_csv('../clean_data.csv')
    ndf = df.copy()
    print(df)

    for column in ndf.columns[1:]: 
        ndf[column] = (ndf[column] - ndf[column].min()) / (ndf[column].max() - ndf[column].min())

    print(ndf)

    prices = ndf['Price'].to_numpy().reshape(-1,1)

    print(prices)
    nArray = ndf.drop(['Unnamed: 0', 'Price','Others_F','Others_O','Second_O','Manual','Diesel','Petrol'], axis=1).to_numpy()
    print(nArray)

    weights = gradientDescent(nArray,prices,iterations=50000, alpha=0.01, epsilon=2e-3)
    print(f"Pesos: \n{weights}")

    X = np.append(np.ones((nArray.shape[0], 1)), nArray, axis=1)
    print(predict(X,weights))

    denormalized_prices = (predict(X,weights) * (df['Price'].max() - df['Price'].min())) + df['Price'].min()
    print(denormalized_prices)


main()