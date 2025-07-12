# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 22:27:50 2025

@author: Zeng Baiming
"""
from time import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from swissroll import swiss_roll_dataset
from utils.dimensionality_utils import run_isomap

fig = plt.figure(figsize=(15, 8))
#data_num = [10000, 100000, 1000000]
data_num = [1000, 10000, 50000]
num_neighbors = 10
                 
for i in range(3):
    phi,_,swiss_roll =swiss_roll_dataset(0.004*data_num[i]*np.pi, 0.001*data_num[i], 20, data_num[i])
    X3 = swiss_roll[:, :3]
    t0 = time()
    trans_data=run_isomap(X3,2,num_neighbors)
    t1 = time()
    print(f"{data_num[i]}points in 3D: %.2g sec" % (t1 - t0))
    
    ax = fig.add_subplot(231+i)
    plt.scatter(trans_data[:,0], trans_data[:,1],c = phi,cmap=plt.cm.rainbow,s=2)
    plt.title(f"{data_num[i]}points in 3D (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis("tight")
    
for i in range(3):
    phi,_,swiss_roll =swiss_roll_dataset(0.004*data_num[i]*np.pi, 0.001*data_num[i], 20, data_num[i])
    t0 = time()
    trans_data=run_isomap(swiss_roll,2,num_neighbors)
    t1 = time()
    print(f"{data_num[i]}points in 20D: %.2g sec" % (t1 - t0))
    
    ax = fig.add_subplot(234+i)
    plt.scatter(trans_data[:,0], trans_data[:,1], c = phi, cmap=plt.cm.rainbow,s=2)
    plt.title(f"{data_num[i]}points in 20D (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis("tight")

plt.show()