import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

with open("params.pickle", "rb") as file:
	log = pickle.load(file)

def plot(X, y):
  l = []
  for i in range(y.shape[0]):
    if(y[i] == 1): 
    	l.append(i)
  np_succ = X[l]
  np_fail = np.delete(X, l, 0)

  plt.scatter(np_succ[:, 0], np_succ[:, 1], color = "blue")
  plt.scatter(np_fail[:, 0], np_fail[:, 1], color = "red")


def f(X, Y):
	params = log["params"]
	X = X.reshape(1, -1)
	Y = Y.reshape(1, -1)
	Z = np.dot(params["w1"].T, np.concatenate((X, Y), axis = 0))
	return Z

data = pd.read_csv("2d_dataset_3.csv")
trainx = data.drop(columns = "y").to_numpy()
scaler = StandardScaler()
trainx = scaler.fit_transform(trainx)
trainy = data.y.to_numpy()

plot(trainx, trainy)
plt.show()