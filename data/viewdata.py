import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from matplotlib import pyplot as plt
from matplotlib import style
from mpl_toolkits import mplot3d


df = pd.read_csv("multiple-linear-regression-dataset.csv")
w1 = 0.47187332
w2 = 0.47187332
b = 0.04409846724322021

data = preprocessing.scale(df)

ax = plt.axes(projection = '3d')

ax.scatter3D(data[:, 0], data[:, 2], data[:, 1])
ax.plot3D([-1.5, 2], [-1.5, 2], [(w1 * -1.5) + (w2 * -1.5) + b, (w1 * 2) + (w2 * 2) + b], color = 'orange', linewidth = 2)

ax.xlabel = "deneyim"
ax.ylabel = "yas"
ax.zlabel = "mass"

plt.show()




