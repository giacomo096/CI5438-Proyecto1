from operator import indexOf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gradient_descent import gradientDescent
from gradient_descent import predict
import math
from sklearn.model_selection import KFold

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
    for y, x in zip(df['Price'],l):
        w = abs(((y-x)/y))
        l2.append(w)

    return  sum(l2)/len(l2),(l2[1029],l2[1030])


#Calcula la varianza dado el vector de prueba y su predicción
def score(y_test, predictions):
    
    l = []
    for y, x in zip(y_test,predictions):
        l.append(abs(y-x))

    return sum(l)/len(l)


def main():

    #Data import   
    df = pd.read_csv('../clean_data.csv')
    X = df.copy()
    for column in X.columns[1:]:
            X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min())

    #Extraemos la columna de precio y eliminamos las que no son necesarias        
    y = X['Price'].to_numpy().reshape(-1,1)
    X = X.drop(['Unnamed: 0', 'Price', 'Others_F', 'Others_O'], axis=1).to_numpy()

    

    #Valores de cada conjunto
    scores = []
    weights = []

    kf = KFold(n_splits=5, shuffle=False )
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        # Se divide la data en conjunto de entrenamiento y de prueba
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Se entrena el modelo
        w = gradientDescent(X_train, y_train, iterations=5000, alpha=0.0001, epsilon=2e-3)
        weights.append(w)
        
        #Se evalua el modelo
        predictions = predict(X_test,w)
        
        #Calcula el puntaje de cada conjunto
        s = score(y_test, predictions)
        scores.append(s)

    #Promediamos los pesos de los conjuntos
    means = [np.mean(row) for row in zip(*weights)]
    print("\nPesos: ", means)
    print("\n Varianza: ", variance(X,df,means))
    print("\n Average score: ", np.mean(scores))

    
    #Hacemos el entrenamiento con el conjunto completo y medimos su varianza
    weights = gradientDescent(X, y, iterations=5000, alpha=0.01, epsilon=2e-3)
    print("\n Pesos: ", weights)
    print("\n Varianza: ", variance(X,df,weights))
    predictions = predict(X,weights)
    print("\n Score: ", score(y,predictions))


main()