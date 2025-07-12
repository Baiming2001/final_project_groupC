# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 21:26:08 2025

@author: Zeng Baiming
"""
import numpy as np
import matplotlib.pyplot as plt
from swissroll import swiss_roll_dataset

datapoints=1000000
length_phi=400*np.pi
length_Z=2000
phi, Z, data = swiss_roll_dataset(length_phi, length_Z, 20, datapoints)
                                     
                                      

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=1, c=phi, cmap='Spectral')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.suptitle("Swiss Roll in 3D", fontsize=12)
ax.set_title(f" -- {datapoints} hidden in 20D space", x=1, ha='right' ,fontsize=10)
fig.subplots_adjust(top=0.9)
plt.show()