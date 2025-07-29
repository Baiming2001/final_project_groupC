# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 14:11:17 2025

@author: Zeng Baiming
"""
import numpy as np

def swiss_roll_dataset(length_phi, length_Z, n_dimensions=20, datapoints=5000):
    length_phi = length_phi #length of swiss roll in angular direction
    length_Z = length_Z #length of swiss roll in z direction
    sigma = 0      #noise strength
    m = datapoints        #number of samples
    extend_dimensions = n_dimensions - 3 #number of dimensions to extend from original 3D_space

    ##generate dataset
    np.random.seed(101) #set random seed to be 101
    phi = length_phi*np.random.rand(m)
    xi = np.random.rand(m)
    Z = length_Z*np.random.rand(m)
    X = 1./6*(phi + min(length_Z,length_phi) + sigma*xi)*np.sin(phi)
    Y = 1./6*(phi + min(length_Z,length_phi) + sigma*xi)*np.cos(phi)

    swiss_roll = np.array([X, Y, Z]).T
    swiss_roll = np.append(swiss_roll,[[0]*extend_dimensions]*m,1)
    #rotate with X coordiante
    theta = np.radians(90)
    rotation = np.identity(n_dimensions)
    rotation[2,2]=np.cos(theta)
    rotation[1,2]=-np.sin(theta)
    rotation[2,1]=np.sin(theta)
    rotation[1,1]=np.cos(theta)
    swiss_roll=rotation.dot(swiss_roll.T).T
    
    return phi, Z, swiss_roll
