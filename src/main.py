import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gradient_descent import gradientDescent
from gradient_descent import predict
import math

#Funcion que elimina los valores que generan mas varianza
def shortVariance(X,df,ndf, w):
    
    #predicciones de todos los precios dado los pesos
    predictions = (predict(X,w) * (df['Price'].max() - df['Price'].min())) + df['Price'].min()

    #Varianza de cada prediccion
    l2 = []
    for x, y in zip(df['Price'],predictions):
        w = math.sqrt(((x-y)/x)*((x-y)/x))
        l2.append(w)

    df['aux'] = l2

    df = df.loc[df["aux"] < 3]
    df = df.drop(["aux"],axis = 1)
    
    return df


def variance(X,df,w):

    l = (predict(X,w) * (df['Price'].max() - df['Price'].min())) + df['Price'].min()

    l2 = []
    for x, y in zip(df['Price'],l):
        w = math.sqrt(((x-y)/x)*((x-y)/x))
        l2.append(w)

    return  sum(l2)/len(l2),(l2[1029],l2[1030])




def main():

    #Importamos la data limpia
    df = pd.read_csv('../clean_data.csv')
    ndf = df.copy()
    print(df)

    #Normalizamos la data
    for column in ndf.columns[1:]: 
        ndf[column] = (ndf[column] - ndf[column].min()) / (ndf[column].max() - ndf[column].min())
    print(ndf)



    #Extraemos la columna de precios
    prices = ndf['Price'].to_numpy().reshape(-1,1)

    #Eliminamos las columnas no relevantes
    nArray = ndf.drop(['Unnamed: 0', 'Price'], axis=1).to_numpy()

    w = gradientDescent(nArray,prices,iterations=5000, alpha=0.01, epsilon=2e-3)
    r1 = variance(nArray,df,w)

    #Eliminamos las filas con valores atípicos
    df = shortVariance(nArray,df,ndf,w)
    ndf = df.copy()
    print(df)

    #Normalizamos la data
    for column in ndf.columns[1:]: 
        ndf[column] = (ndf[column] - ndf[column].min()) / (ndf[column].max() - ndf[column].min())
    print(ndf)



    #Extraemos la columna de precios
    prices = ndf['Price'].to_numpy().reshape(-1,1)

    #Eliminamos las columnas no relevantes
    nArray = ndf.drop(['Unnamed: 0', 'Price'], axis=1).to_numpy()

    w2 = gradientDescent(nArray,prices,iterations=5000, alpha=0.01, epsilon=2e-3)

    r2 = variance(nArray,df,w2)
    
    print(r1,r2)

    #Validamos la predicción de los precios
    #denormalized_prices = (predict(nArray,w) * (df['Price'].max() - df['Price'].min())) + df['Price'].min()
    #denormalized_prices2 = (predict(nArray2,w2) * (df['Price'].max() - df['Price'].min())) + df['Price'].min()
    #print(denormalized_prices)
    #print(denormalized_prices2)


main()