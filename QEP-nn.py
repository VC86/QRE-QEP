#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Quantised Eclipse Problem - Study by Effect on k-NN

Created on Tue Feb 21 08:42:57 2017

@author: Dr. Valerio Cambareri, Univ. catholique de Louvain, Belgium
"""

import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
#from sklearn.manifold import TSNE
#from sklearn.neighbors.nearest_centroid import NearestCentroid

def quant_randn_embedding(x, m, delta, rngseed):
    dims = np.shape(x)
    np.random.seed(rngseed)
    phi = np.random.standard_normal([m,dims[0]])
    xi = np.random.uniform(-delta*0.5,delta*0.5,m)
    w = np.dot(phi, x) + (np.tile(xi,[dims[1],1])).T
    return np.round(w/delta)*delta 
    # return np.int16(np.round(w/delta))

if __name__ == "__main__":

    # Initialise random seed for projector
    # rngseed = np.random.randint(1,2**10)
    
    # Dimensions
    n = 256
    m = 60
    ntrials = int(1e4)
    nclass = 3
    
    # Geometric parameters
    sigma = 0.0
    r = 1.0
    sep = 2*r + sigma 
    
    # Generates equidistant cluster centers (vertices of rotated and scaled simplex)
    cluster_centers = np.zeros([n,nclass])
    idxc = np.random.choice(range(n),nclass,False)
    cluster_centers[idxc,:] = sep*np.eye(nclass)/np.sqrt(2)
    phi,psi = np.linalg.qr(np.random.standard_normal([n,n])/np.sqrt(n))
    cluster_centers = np.dot(phi,cluster_centers)
    
    trials = np.zeros([nclass,n,ntrials])
    for i in range(nclass):
        # radii = np.tile(np.random.uniform(0,1,ntrials),[n,1])
        ballvecs = normalize(np.random.standard_normal([n,ntrials]).T).T # *radii
        trials[i,:,:] = r*ballvecs + np.tile(cluster_centers[:,i], [ntrials,1]).T

    trials = np.rollaxis(trials,1).reshape([n,ntrials*nclass])
    
    # Distance-based Classifier
    dist = np.zeros([ntrials*nclass,nclass])
    for i in range(nclass):
        dist[:,i] = np.sqrt(np.sum((trials-np.tile(cluster_centers[:,i], [nclass*ntrials,1]).T)**2,0))
    clf = np.argmin(dist,1)
    
    combis = combinations(range(nclass),2)
    coms = np.asarray([p for p in combis])
    tau = 0.0*np.ones(len(coms))
    rngseed = 77
    
    while np.any(tau < 1e-16):
        rngseed += 7
        np.random.seed(rngseed)
        test_projector = np.random.standard_normal([m,n])
    
        for i in range(len(coms)):
            c = np.subtract(cluster_centers[:,coms[i,0]],cluster_centers[:,coms[i,1]])
            z = Variable(n)
            obj = Minimize(norm(test_projector*z, "inf"))
            con = [norm(z-c) <= 2*r]
            prob = Problem(obj, con)
            
            try:
                prob.solve(solver=ECOS,verbose=False) 
            except SolverError:
                prob.solve(solver=SCS,verbose=False)
                
            tau_star = obj.value        
            tau[i] = tau_star 
               
    print "The current tau are: %s" % str(tau)
    
    # %% Perform the quantised embedding    
    delta = min(tau)
    if delta < 1e-3:
        print "Kernel not empty in worst case"
    
    delta = 2
    
    trials_proj = quant_randn_embedding(trials, m, delta, rngseed)
    centers_proj = quant_randn_embedding(cluster_centers, m, delta, rngseed)
    # Distance-based Classifier
    dist = np.zeros([ntrials*nclass,nclass])
    for i in range(nclass):
        dist[:,i] = np.sqrt(np.sum((1.0*trials_proj-np.tile(centers_proj[:,i], [nclass*ntrials,1]).T)**2,0))
    clf_proj = np.argmin(dist,1)    
    
    viz = PCA(2).fit(trials.T)
    trials_vis = viz.transform(trials.T)
    centers_vis = viz.transform(cluster_centers.T)
    
    viz = PCA(2).fit(trials_proj.T)
    trials_proj_vis = viz.transform(trials_proj.T)
    proj_centers_vis = viz.transform(centers_proj.T)
    
    plt.close()
    plt.figure(1)
    
    plt.subplot(121)
    for i in range(nclass):
        plt.scatter(trials_vis[i*ntrials:(i+1)*ntrials,0],trials_vis[i*ntrials:(i+1)*ntrials,1])
    
    plt.scatter(centers_vis[:,0],centers_vis[:,1])
    plt.axis('equal')
    
    plt.subplot(122)
    for i in range(nclass):
        plt.scatter(trials_proj_vis[i*ntrials:(i+1)*ntrials,0],trials_proj_vis[i*ntrials:(i+1)*ntrials,1])
        
    plt.scatter(proj_centers_vis[:,0],proj_centers_vis[:,1])
    plt.axis('equal')
    
    plt.rcParams["figure.figsize"] = [12,6]
    
    errs = sum(clf != clf_proj)
    error_rate = errs/float(ntrials*nclass)
    print "Error rate %4.2e" % error_rate
