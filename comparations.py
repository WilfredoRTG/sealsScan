import numpy as np

redAntigua = np.load('redAntigua.npy')
redNueva = np.load('redNueva.npy')


print("Numero de predicciones de Red Antigua: ", len(redAntigua))
print("Numero de predicciones de Red Nueva: ", len(redNueva))