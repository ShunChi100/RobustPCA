# Robust Principle Component Analysis

Implementation of robust principal component analysis and stable principal component pursuit based on the following references:

* Candes, Emmanuel J. et al. "Robust Principal Component Analysis?" Journal of the ACM, Vol. 58, No. 3, Article 11, 2011.
* Zhou, Zihan, et al. "Stable principal component pursuit." Information Theory Proceedings (ISIT), 2010 IEEE International Symposium on. IEEE, 2010.

### Description
The classical _Principal Component Analysis_ (PCA) is widely used for high-dimensional analysis and dimensionality reduction. Mathematically, if all the data points are stacked as column vectors of a (n, m)matrix $M$, PCA tries to decompose $M$ as

$$M = L + S,$$

where $L$ is a rank $k$ ($k<\min(n,m)$) matrix and $S$ is some perturbation/noise matrix. To obtain $L$, PCA solves the following optimization problem

$$\min_{L} ||M-L||_2,$$

given that rank($L$) <= $k$. However, the effectiveness of PCA relies on the assumption of the noise matrix $S$: $s_{i,j}$ is small and i.i.d. Gaussian. That means PCA is not robust to outliers in data $M$.

To resolve this issue, Candes, Emmanuel J. et al proposed _Robust Principal Component Analysis_ (Robust PCA or RPCA). The objective is still trying to decompose $M$ into $L$ and $S$, but instead optimizing the following problem

$$ \min_{L,S} ||L||_{*} + \lambda||S||_{1}$$

subject to $L+S = M$.

Minimizing the $l_1$-norm of $S$ is known to favour sparsity while minimizing the
nuclear norm of $L$ is known to favour low-rank matrices (sparsity of singular values). In this way, $M$ is decomposed to a low-rank matrix but not sparse $L$ and a sparse but not low rank $S$. Here $S$ can be viewed as a sparse noise matrix. Robust PCA allows the separation of sparse but outlying values from the original data.  

Also, Zhou et al. further proposed a "stable" version of Robust PCA, which is called _Stable Principal Component Pursuit_ (Stable PCP or SPCP), which allows a non-sparse Gaussian noise term $Z$ in addition to $L$ and $S$:

$$M = L+S+Z.$$

Stable PCP is intuitively more practical since it combines the strength of classical PCA and Robust PCA. However, depending on the exact problem, the proper method should be selected.

The drawback of Robust PCA and Stable PCP is their scalability. They are generally slow since the implementation do SVD (singular value decomposition) in the converging iterations. Recently, a new algorithm was proposed: "[Grassmann Averages](https://ieeexplore.ieee.org/document/6909882)" for Scalable Robust PCA.


### Examples

To install the package:
```
pip install git+https://github.com/ShunChi100/RobustPCA
```

To use
```
from RobustPCA.rpca import RobustPCA
from RobustPCA.spcp import StablePCP

rpca = RobustPCA()
spcp = StablePCP()

rpca.fit(M)
L = rpca.get_low_rank()
S = rpca.get_sparse()

spcp.fit(M)
L = spcp.get_low_rank()
S = spcp.get_sparse()
```
Here `L` and `S` are desired low rank matrix and sparse matrix.

For more options of these functions, please see the documentation and source codes.

### Contributions
Feel free to fork and develop this project. It is under MIT license.
