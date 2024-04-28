# -*- coding: utf-8 -*-
"""
Quantised Eclipse Problem - Full Scale Simulations

Created on Tue Feb 21 08:42:57 2017

@author: Dr. Valerio Cambareri, Univ. catholique de Louvain, Belgium
"""

# Let's first import the required libraries.
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# from cvxpy import *
# from itertools import combinations
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA

# from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits, make_blobs


###
def quant_rand_embed(x, m=2, delta=1, rngseed=42, mode="randn"):
    """
    Quantised Random Embedding using Random Projections
    and Uniform Scalar Quantisation with Dithering

    See also:
    Laurent Jacques and Valerio Cambareri.
    "Time for dithering: fast and quantized random embeddings via the restricted isometry property."
    arXiv preprint arXiv:1607.00816 (2016).
    In Information and Inference: A Journal of the IMA, accepted and in press, Mar. 2017.

    Inputs:
    x         input data             ndarray, shape (nvariables, nsamples)
    m         output feature size    int
    delta     quantiser bin width    float (>0)
    rngseed   generator state        int
    mode      type of random proj.   string

    Outputs:
    w         output features        float (quantised)
    q         output features        np.int16 (quantised)
    """

    # Initialise the quant. rand. proj. setup
    dims = np.shape(x)
    if rngseed > 0:
        np.random.seed(rngseed)
    else:
        raise Exception("Seed must be nonnegative!")

    if delta < 0:
        raise Exception("Quantiser resolution parameter must be nonnegative!")

    # Apply the sensing operator
    # The random projection is applied to axis 0 of the input data, which is
    # assumed as a matrix of size $n \times p$
    if mode == "randn":
        # Gaussian Random Projections
        phi = np.random.standard_normal([m, dims[0]])
        q = np.dot(phi, x)

    elif mode == "randc":
        # Random Convolution using Antipodal Kernel (via FFT)
        ker = 2 * (np.random.randint(0, 1, dims[0]) - 0.5)
        idx = np.random.choice(dims[0], size=m, replace=False)
        q = np.fft.irfft(
            np.dot(np.diag(np.fft.rfft(ker)), np.fft.rfft(x, axis=0)), axis=0
        )[idx]
        # # Alternative w. convolve
        # q = (np.apply_along_axis(lambda x: np.convolve(x,ker,mode='fft'), 0, x))[idx]

    else:
        raise Exception(
            'Mode not recognised. Only "randn" and "randc" are implemented so far.'
        )

    if delta == 0:
        return q, q

    # Generate dithering
    xi = np.random.uniform(-delta * 0.5, delta * 0.5, m)

    # Apply dithering and quantise with round operation
    w = np.round((q + (np.tile(xi, [dims[1], 1])).T) / delta)
    return w * delta, np.int16(w)


if __name__ == "__main__":

    plt.close()

    # Prepare train/test data
    nclass = 10  # Number of classes, $C$
    main_rseed = 77  # Seed of random number generator
    N = 256  # Dimensionality
    dset = 0  # Dataset choice
    nsamples = 128  # Must be less than available digits from dataset
    ntri = 16  # Repetitions per sensing matrix

    if dset == 1:
        # The Digits dataset (real)
        # This is the well-known handwritten digits dataset, loaded with nclass labels.
        # The classes will tend to be closer and overlap, thus departing from our
        # theoretical analysis that assumes linear separability.

        digits = load_digits(nclass)
        idx_train = np.random.choice(nsamples, nsamples // 2, replace=False)
        idx_test = np.setxor1d(np.arange(nsamples), idx_train)
        train_data = digits.data[idx_train]
        test_data = digits.data[idx_test]
        ctrain = digits.target[idx_train]
        ctest = digits.target[idx_test]

    if dset == 2:
        # The Blob dataset (synthetic)
        # This is a randomly generated dataset of blobs that tend to be sufficiently far apart.
        # The classes will tend to be separated, thus verifying our theoretical analysis.

        data, target = make_blobs(
            n_samples=nsamples, n_features=N, centers=nclass, cluster_std=2
        )
        idx_train = np.random.choice(nsamples, nsamples // 2, replace=False)
        idx_test = np.setxor1d(np.arange(nsamples), idx_train)
        train_data = data[idx_train]
        test_data = data[idx_test]
        ctrain = target[idx_train]
        ctest = target[idx_test]

    else:
        # The Blob dataset (synthetic)
        # This is a randomly generated dataset of blobs that tend to be sufficiently far apart.
        # The classes will tend to be separated, thus verifying our theoretical analysis.

        data, target = make_blobs(
            n_samples=nsamples, n_features=N, centers=nclass, cluster_std=0.5
        )
        idx_train = np.random.choice(nsamples, nsamples // 2, replace=False)
        idx_test = np.setxor1d(np.arange(nsamples), idx_train)
        train_data = data[idx_train]
        test_data = data[idx_test]
        ctrain = target[idx_train]
        ctest = target[idx_test]

    # Fit $\ell_2$-balls to each dataset
    centre = np.zeros([N, nclass])
    radius = np.zeros(nclass)
    for i in range(nclass):
        centre[i,] = np.mean(train_data[ctrain == i])
        nclassi = sum(ctrain == i)
        radius[i] = np.max(
            np.linalg.norm(
                train_data[ctrain == i] - np.tile(centre[:, i], (nclassi, 1)), axis=1
            )
        )

    print("The ball classes' radii are " + str(radius))

    # Minimum intra-class distance (closest point problem)
    mind = np.inf * np.ones([nclass, nclass])
    for i in range(nclass):
        for j in range(i + 1, nclass):
            dists = pairwise_distances(
                train_data[ctrain == i], train_data[ctrain == j], metric="euclidean"
            )
            mind[i, j] = np.min(dists)

    sigma = np.min(mind)
    print("The separation between classes is " + str(sigma))

    # With respect to $m$
    # M = np.unique(np.round(N*2**np.arange(-np.log2(N),0,0.5))).astype(int)
    # M = np.unique(np.round(np.logspace(0,np.log2(N),num=16,base=2.0,dtype=int))).astype(int)
    M = np.arange(1, N, 1, dtype=int)

    msteps = len(M)
    # delta = min(tau) # Safe estimate
    delta0 = 0  # No delta
    delta1 = sigma / 4  # Rough estimate (relies on isometry)
    delta2 = sigma  # Very rough estimate (relies on isometry)

    qrp_perr_0 = []
    qrp_perr_1 = []
    qrp_perr_2 = []
    pca_perr = []

    for tt in range(ntri):
        rseed = np.random.randint(1, high=32768)
        KNN = KNeighborsClassifier(p=2, metric="minkowski")

        for ss in range(msteps):
            # Apply QRP with dimension m and resolution delta
            print("QRP_0_" + str(M[ss]))
            red_train, _ = quant_rand_embed(
                train_data.T, m=M[ss], delta=delta0, rngseed=rseed
            )
            red_test, _ = quant_rand_embed(
                test_data.T, m=M[ss], delta=delta0, rngseed=rseed
            )
            red_train = red_train.T
            red_test = red_test.T
            KNN.fit(red_train, ctrain)
            qrp_cpred_0 = KNN.predict(red_test)
            qrp_perr_0.append(float(sum(qrp_cpred_0 != ctest)) / ctest.size)
            bitrate = np.floor(np.log2((np.unique(red_train[:])).size)) + 1

            # Apply QRP with dimension m and resolution delta
            print("QRP_1_" + str(M[ss]))
            red_train, _ = quant_rand_embed(
                train_data.T, m=M[ss], delta=delta1, rngseed=rseed
            )
            red_test, _ = quant_rand_embed(
                test_data.T, m=M[ss], delta=delta1, rngseed=rseed
            )
            red_train = red_train.T
            red_test = red_test.T
            KNN.fit(red_train, ctrain)
            qrp_cpred_1 = KNN.predict(red_test)
            qrp_perr_1.append(float(sum(qrp_cpred_1 != ctest)) / ctest.size)
            bitrate = np.floor(np.log2((np.unique(red_train[:])).size)) + 1

            # Apply QRP with dimension m and resolution delta
            print("QRP_2_" + str(M[ss]))
            red_train, _ = quant_rand_embed(
                train_data.T, m=M[ss], delta=delta2, rngseed=rseed
            )
            red_test, _ = quant_rand_embed(
                test_data.T, m=M[ss], delta=delta2, rngseed=rseed
            )
            red_train = red_train.T
            red_test = red_test.T
            KNN.fit(red_train, ctrain)
            qrp_cpred_2 = KNN.predict(red_test)
            qrp_perr_2.append(float(sum(qrp_cpred_2 != ctest)) / ctest.size)
            bitrate = np.floor(np.log2((np.unique(red_train[:])).size)) + 1

            # Apply PCA with the same dimension m, and train it
            print("PCA_" + str(M[ss]))
            PC = PCA(n_components=M[ss])
            PC.fit(train_data)
            pca_train = PC.transform(train_data)
            pca_test = PC.transform(test_data)
            KNN.fit(pca_train, ctrain)
            pca_cpred = KNN.predict(pca_test)
            pca_perr.append(float(sum(pca_cpred != ctest)) / ctest.size)

    # %% Plot results (M)
    mqrp_perr_0 = np.mean(np.reshape(np.array(qrp_perr_0), (ntri, msteps)), axis=0)
    mqrp_perr_1 = np.mean(np.reshape(np.array(qrp_perr_1), (ntri, msteps)), axis=0)
    mqrp_perr_2 = np.mean(np.reshape(np.array(qrp_perr_2), (ntri, msteps)), axis=0)
    mpca_perr = np.mean(np.reshape(np.array(pca_perr), (ntri, msteps)), axis=0)

    plt.figure(figsize=(8, 6))
    plt.plot(np.log2(M * 1.0 / N), mqrp_perr_0, label="$QRP_{m,\delta=0}$", marker="o")
    plt.plot(
        np.log2(M * 1.0 / N),
        mqrp_perr_1,
        label="$QRP_{m,\delta=0.25 \sigma}$",
        marker="o",
    )
    plt.plot(
        np.log2(M * 1.0 / N), mqrp_perr_2, label="$QRP_{m,\delta=\sigma}$", marker="o"
    )
    plt.plot(np.log2(M * 1.0 / N), mpca_perr, label="$PCA_{m}$", marker="o")
    plt.title("Probability of error")
    plt.xlabel("$\\log_2 m/n$")
    plt.ylabel("$P_e$")
    plt.legend()
    plt.show()
