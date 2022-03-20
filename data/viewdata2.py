import numpy as np 
from matplotlib import pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data = pd.read_csv("2d_dataset_4.csv")
X = data.drop(columns = "y").to_numpy()/1000
X = scaler.fit_transform(X)
y = data.y.to_numpy()

with open("params4.pickle", "rb") as file:
	log = pickle.load(file)
params = log['params']

def plot(X, y):
  l = []
  for i in range(y.shape[0]):
    if(y[i] == 1): 
    	l.append(i)
  np_succ = X[l]
  np_fail = np.delete(X, l, 0)
  # ax.scatter3D(np_succ[:, 0], np_succ[:, 1], np.zeros((np_succ.shape[0], 1)), color = "blue")
  # ax.scatter3D(np_fail[:, 0], np_fail[:, 1], np.zeros((np_fail.shape[0], 1)), color = "red")
  return np_succ, np_fail

def relu(z):
	return np.where(z > 0, z, 0)

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def f(X, Y):
  params = log["params"]
  X = X.reshape(1, -1)
  Y = Y.reshape(1, -1)
  z1 = np.dot(params["w1"].T, np.concatenate((X, Y), axis  = 0)) + params["b1"]
  a1 = relu(z1)
  z2 = np.dot(params["w2"].T, a1) + params["b2"]
  # Z = np.dot(params["w2"])
  # Z = relu(np.dot(params["w1"].T, np.concatenate((X, Y), axis = 0)))
  return a1

def plotSurface():
	gridx = np.linspace(-2, 2, 30)
	gridy = np.linspace(-2, 2, 30)
	gridX, gridY = np.meshgrid(gridx.T, gridy.T)
	gridZ = f(gridX, gridY)
	if(gridZ.shape[0] == 1):
		gridZ = np.concatenate([gridZ, np.zeros((2, gridZ.shape[1]))])
	ax.plot_wireframe(gridZ[0, :].reshape(30, 30), gridZ[1, :].reshape(30, 30), gridZ[2, :].reshape(30, 30), rstride=1, cstride=1,
                cmap='viridis', color = "black")

def plotDataPoints(X, y):
	np_succ, np_fail = plot(X, y)


	z1_succ = np.dot(params["w1"].T, np_succ.T) + params["b1"]
	z1_fail = np.dot(params["w1"].T, np_fail.T) + params["b1"]
	a1_succ = relu(z1_succ)
	a1_fail = relu(z1_fail)
	z2_succ = np.dot(params["w2"].T, a1_succ) + params["b2"]
	z2_fail = np.dot(params["w2"].T, a1_succ) + params["b2"]

	trans_succ = a1_succ
	trans_fail = a1_fail

	if(trans_succ.shape[0] == 1):
		trans_succ = np.concatenate([trans_succ, np.zeros((2, trans_succ.shape[1]))], axis = 0)

	if(trans_fail.shape[0] == 1):
		trans_fail = np.concatenate([trans_fail, np.zeros((2, trans_fail.shape[1]))])

	ax.scatter3D(trans_succ[0, :], trans_succ[1, :], trans_succ[2, :], color = "blue")
	ax.scatter3D(trans_fail[0, :], trans_fail[1, :], trans_fail[2, :], color = "red")


ax = plt.axes(projection = "3d")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
plotDataPoints(X, y)
# plotSurface()

X = X.T 
y = y.reshape(1, -1)

decx = np.linspace(-2, 2, 30)
decy = np.linspace(-2, 2, 30)
decX, decY = np.meshgrid(decx, decy)
decX = decX.reshape(1, -1)
decY = decY.reshape(1, -1)
decz = sigmoid(np.dot(params["w2"].T, relu(np.dot(params["w1"].T, np.concatenate([decX, decY])) + params["b1"])) + params["b2"])
ax.plot_surface(decX.reshape(30, 30), decY.reshape(30, 30), decz.reshape(30, 30), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
plt.show()