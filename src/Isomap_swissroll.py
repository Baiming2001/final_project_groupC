# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 22:27:50 2025

@author: Zeng Baiming
"""
from time import time
import matplotlib.pyplot as plt
import numpy as np
import psutil,os
from matplotlib.ticker import NullFormatter
from swissroll import swiss_roll_dataset
from utils.dimensionality_utils import run_isomap

#initial memory monitor
proc = psutil.Process(os.getpid()) 
mem_mb = lambda: proc.memory_info().rss / 1024 / 1024
mem_limit_gb = 10
def est_mem_gb(n_samples, bytes_per_elem=8):
    """estimate the memory cost to implement """
    return 2*bytes_per_elem * n_samples**2 / 1024**3

#parameters initialization
fig = plt.figure(figsize=(15, 8))
data_num = [10000, 20000, 30000, 50000, 100000, 200000, 500000, 1000000]
num_neighbors = 10

#process data in 3D space    
for i in range(len(data_num)):
    est_gb = est_mem_gb(data_num[i])
    if est_gb > mem_limit_gb:                               
        print(f"Skipping N={data_num[i]}: estimated memory {est_gb:.2f} GB > {mem_limit_gb} GB")
        continue
    
    phi,_,swiss_roll =swiss_roll_dataset(np.log10(data_num[i])*np.pi, 0.001*data_num[i], 3, data_num[i])
    #record memory and time(before execution)
    t0 = time()
    m0 = mem_mb()
    #perform isomap
    trans_data=run_isomap(swiss_roll,2,num_neighbors)
    #record memory and time(after execution)
    t1 = time()
    m1 = mem_mb() 
    print(f"{data_num[i]}points in 3D: %.2f sec" % (t1 - t0))
    print("memory cost is %.2f mb"%(m1-m0))
    
    ax = fig.add_subplot(281+i)
    plt.scatter(trans_data[:,0], trans_data[:,1],c = phi,cmap='Spectral',s=2)
    plt.title(f"{data_num[i]}points in {data_num[i]}D (%.2f sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis("tight")
plt.show()