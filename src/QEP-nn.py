# -*- coding: utf-8 -*-
"""
Quantised Eclipse Problem -- Quantifying the Effect on k-NN

Created on Tue Feb 21 08:42:57 2017

@author: Dr. Valerio Cambareri, Univ. catholique de Louvain, Belgium
"""
import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

plt.rcParams["figure.figsize"] = [12, 6]

DEFAULT_MIN_DELTA = 1e-3
DEFAULT_MIN_TAU = 1e-16


def quant_randn_embedding(x, m, delta, rngseed):
    """Quantized Random Gaussian Embeddings

    Applies random embeddings to the input, given target dimension and quantization bin width.
    This is a leaner code than that used in QEP-sim.py as it supports a subset of functions.

    Author: V. Cambareri

    Parameters
    ----------
    x
        input tensor
    m
        codomain dimension
    delta
        quantization bin width
    rngseed
        seed for random Gaussian basis generation

    Returns
    -------
    u
        quantized and dithered embeddings of x at given width delta and dimension m
    """
    dims = np.shape(x)
    np.random.seed(rngseed)
    phi = np.random.standard_normal([m, dims[0]])
    xi = np.random.uniform(-delta * 0.5, delta * 0.5, m)
    w = np.dot(phi, x) + (np.tile(xi, [dims[1], 1])).T
    u = np.round(w / delta) * delta
    return u


def run_qep_nn_trials(
    n=256, m=60, ntrials=10000, nclass=3, sigma=0.0, r=1.0, delta=10, rngseed=77
):
    # %% Setup
    # Generates equidistant cluster centers (vertices of rotated and scaled simplex)
    sep = 2 * r + sigma
    cluster_centers = np.zeros([n, nclass])
    idxc = np.random.choice(range(n), nclass, False)
    cluster_centers[idxc, :] = sep * np.eye(nclass) / np.sqrt(2)
    phi, _ = np.linalg.qr(np.random.standard_normal([n, n]) / np.sqrt(n))
    cluster_centers = np.dot(phi, cluster_centers)

    trials = np.zeros([nclass, n, ntrials])
    for i in range(nclass):
        ballvecs = normalize(np.random.standard_normal([n, ntrials]).T).T
        trials[i, :, :] = r * ballvecs + np.tile(cluster_centers[:, i], [ntrials, 1]).T

    trials = np.rollaxis(trials, 1).reshape([n, ntrials * nclass])

    # Distance-based Classifier
    dist = np.zeros([ntrials * nclass, nclass])
    for i in range(nclass):
        dist[:, i] = np.sqrt(
            np.sum(
                (trials - np.tile(cluster_centers[:, i], [nclass * ntrials, 1]).T) ** 2,
                0,
            )
        )
    clf = np.argmin(dist, 1)

    combis = combinations(range(nclass), 2)
    coms = np.asarray([p for p in combis])
    tau = np.zeros(len(coms))

    while np.any(tau < DEFAULT_MIN_TAU):
        rngseed += 7
        np.random.seed(rngseed)
        test_projector = np.random.standard_normal([m, n])

        for i in range(len(coms)):
            c = np.subtract(
                cluster_centers[:, coms[i, 0]], cluster_centers[:, coms[i, 1]]
            )
            z = Variable(n)
            obj = Minimize(norm(test_projector @ z, "inf"))
            con = [norm(z - c) <= 2 * r]
            prob = Problem(obj, con)

            try:
                prob.solve(solver=ECOS, verbose=False)
            except SolverError:
                prob.solve(solver=SCS, verbose=False)

            tau_star = obj.value
            tau[i] = tau_star

    print("The current tau are: {}".format(tau))

    # %% Perform Quantized Embeddings
    delta = tau.min()
    if delta < DEFAULT_MIN_DELTA:
        print("Kernel not empty in worst case")

    trials_proj = quant_randn_embedding(trials, m, delta, rngseed)
    centers_proj = quant_randn_embedding(cluster_centers, m, delta, rngseed)

    # Distance-based Classifier
    dist = np.zeros([ntrials * nclass, nclass])
    for i in range(nclass):
        dist[:, i] = np.sqrt(
            np.sum(
                (
                    1.0 * trials_proj
                    - np.tile(centers_proj[:, i], [nclass * ntrials, 1]).T
                )
                ** 2,
                0,
            )
        )
    clf_proj = np.argmin(dist, 1)

    viz = PCA(2).fit(trials.T)
    trials_vis = viz.transform(trials.T)
    centers_vis = viz.transform(cluster_centers.T)

    viz = PCA(2).fit(trials_proj.T)
    trials_proj_vis = viz.transform(trials_proj.T)
    proj_centers_vis = viz.transform(centers_proj.T)

    errs = sum(clf != clf_proj)
    error_rate = errs / float(ntrials * nclass)
    print("Error rate: {}".format(error_rate))

    # %% Plot Results in 2D
    plt.close("all")
    plt.figure(1)
    plt.subplot(121)
    for i in range(nclass):
        plt.scatter(
            trials_vis[i * ntrials : (i + 1) * ntrials, 0],
            trials_vis[i * ntrials : (i + 1) * ntrials, 1],
            label=f"Class {i}", s=1
        )

    plt.scatter(centers_vis[:, 0], centers_vis[:, 1])
    plt.axis("equal")
    plt.legend()

    plt.subplot(122)
    for i in range(nclass):
        plt.scatter(
            trials_proj_vis[i * ntrials : (i + 1) * ntrials, 0],
            trials_proj_vis[i * ntrials : (i + 1) * ntrials, 1],
            label=f"Class {i}", s=1
        )

    plt.scatter(proj_centers_vis[:, 0], proj_centers_vis[:, 1], c="k", marker="x")
    plt.axis("equal")
    plt.legend()
    plt.show(block=True)
    return


if __name__ == "__main__":
    run_qep_nn_trials()
