# Auto-MPFT

This repository is the official implementation of 
[Fast Multidimensional Partial Fourier Transform with Automatic Hyperparameter Selection](https://openreview.net/forum?id=oAaN6LtHut) 
(submitted to KDD 2024).
The codes for [FFTW](http://www.fftw.org/index.html), [MKL](https://software.intel.com/mkl), [PFT](https://github.com/snudatalab/PFT/blob/main/src/PFT.cpp), and [Pruned FFT](http://www.fftw.org/pruned.html) are also included in `src/`.

## Abstract

Given a multidimensional array, how can we optimize the computation process for a part of Fourier coefficients? Discrete Fourier transform plays an overarching role in various data mining tasks. Recent interest has focused on efficiently calculating a small part of Fourier coefficients, exploiting the energy compaction property of real-world data. Current methods for partial Fourier transform frequently encounter efficiency issues, yet the adoption of pre-computation techniques within the PFT algorithm has shown promising performance. However, PFT still faces limitations in handling multidimensional data efficiently and requires manual hyperparameter tuning, leading to additional costs. 
In this paper, we propose Auto-MPFT (Automatic Multidimensional Partial Fourier Transform), which efficiently computes a subset of Fourier coefficients in multidimensional data without the need for manual hyperparameter search. Auto-MPFT leverages multivariate polynomial approximation for trigonometric functions,  generalizing its domain to multidimensional Euclidean space. Moreover, we present a convex optimization-based algorithm for automatically selecting the optimal hyperparameter of Auto-MPFT. We provide a rigorous proof for the explicit reformulation of the original optimization problem of Auto-MPFT, demonstrating the process that converts it into a well-established unconstrained convex optimization problem. Extensive experiments show that Auto-MPFT surpasses existing partial Fourier transform methods and optimized FFT libraries, achieving up to 7.6x increase in speed without sacrificing accuracy. In addition, our optimization algorithm accurately finds the optimal hyperparameter for Auto-MPFT, significantly reducing the cost associated with hyperparameter search.

## Prerequisites
The implementation requires the following libraries.
- mkl.h
- mkl_dfti.h
- ipp.h
- ipps.h
- fftw3.h

## Datasets
We provide the synthetic datasets used in our experiments at [here](https://drive.google.com/open?id=1gt_L-RK1cXc8w1OCbU7q5pkabpA6L9JD&usp=drive_copy).
The real-world datasets are available at [Cityscapes](https://www.cityscapes-dataset.com/), [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/), [DF2K](https://www.kaggle.com/datasets/thaihoa1476050/df2k-ost), [RiceLeaf](https://www.kaggle.com/datasets/shayanriyaz/riceleafs), and [Bird](https://www.kaggle.com/datasets/akash2907/bird-species-classification).

| **Dataset** | **Type** | **# of Images** | **Size** |
| :----------- | :----------- | -----------: | -----------: |
| $S_{n=8,\cdots,15}$ | Synthetic | 1K | $2^n \times 2^n$ |
| Cityscapes | Real-world | 5K | $2048 \times 1024$ |
| ADE20K | Real-world | 20K | $2048 \times 2048$ |
| DF2K | Real-world | 3K | $2040 \times 1536$ |
| RiceLeaf | Real-world | 3.3K | $3120 \times 3120$ |
| Bird | Real-world | 306 | $6000 \times 4000$ |

