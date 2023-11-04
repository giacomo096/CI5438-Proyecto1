from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import sys

#Funcion de perdida
def lossFn(y, h):
    return np.mean((y - h) ** 2)

#Evaluación de los valores con sus pesos
def predict(X, w):
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    return np.dot(X,w)

#
def gradientDescent(X, y, iterations=300000, alpha=0.05, epsilon=1e-6):

    X2 = np.append(np.ones((X.shape[0], 1)), X, axis=1)

    #Se toman las dimensiones del dataset
    m, n = X2.shape

    # Inicializar los pesos
    w = np.zeros((n, 1))
    
    losses = []
    
    for it in range(iterations):

        #Hipotesis
        h = predict(X,w)

        #Actualización de los pesos
        for i in range(n):
            for j in range(n):
                w[i] = w[i] + alpha * (X2[j][i] * (y[j] - h[j]))

        #Calculamos la perdida
        loss = lossFn(y,h)
        if loss <= epsilon:
            print("Converge")
            break
        
        sys.stdout.write(f'\r{loss}')
        sys.stdout.flush()


        losses.append(loss)    

    # Visualización de curva de aprendizaje
    """ fig = plt.figure(figsize = (14, 8))
    plt.plot(losses)
    plt.title("Loss vs. Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show() """

    return w

#Funcion para generar ejemplos en la recta
def generateSamples(w0, w1, w2, num_samples=1000):

    x1 = np.random.rand(num_samples, 1)
    x2 = np.random.rand(num_samples, 1)
    
    X = np.hstack((x1, x2))
    
    # Evaluate the linear function for all samples
    y = w1 * x1 + w2 * x2 + w0
    
    return X, y



def main():

    #Probamos la funcion con pesos fijos
    w0, w1, w2 = 3, 1, 2
    X,Y = generateSamples(w0, w1, w2)

    # Estimación de pesos usando descenso de gradiente
    weights = gradientDescent(X,Y)
    print(f"Pesos: \n{weights}")

    x1 = np.linspace(-200, 200, 10)
    y1 = w1*x1 + w2*x1 + w0
    x2 = np.linspace(-200, 200, 10)
    y2 = weights[1]*x2 + weights[2]*x2 + weights[0]

    plt.plot(x1, y1, color="red")
    plt.plot(x2, y2,'-.',color = "blue")
    plt.title("Test function vs Hypothesis")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()