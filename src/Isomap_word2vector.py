# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 22:27:50 2025

@author: Zeng Baiming
"""
from time import time
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import NullFormatter
from utils.dimensionality_utils import *

fig = plt.figure(figsize=(20, 20))
base_dir = os.path.dirname(os.path.abspath(__file__))
bin_path = os.path.join(base_dir, "..", "data", "word2vec", "text8_vectors.bin")
#num_neighbors = [5,15,25,35]
num_neighbors = [15,25,35,45]
num_components =2
vectors,_ = load_word_vectors(bin_path, top_n=10000)
title="2D Embedding"

for i in range(4):
    t0 = time()
    trans_data = run_isomap(vectors, num_components, num_neighbors[i])
    t1 = time()
    length = len(trans_data[:,0])
    print(f"{length} datasets,runing time: %.2f sec" % (t1 - t0))
    
    ax = fig.add_subplot(221+i)
    plt.scatter(trans_data[:, 0], trans_data[:, 1], s=1, alpha=0.6)
    plt.title(f"isomap neighbors are {num_neighbors[i]}" )
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())


plt.show()