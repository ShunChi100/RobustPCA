# Authors: Shun Chi (shunchi100@gmail.com)

import numpy as np
try:
    from fbpca import pca
except ModuleNotFoundError:
    print('\n install fbpca first: `pip install fbpca` \n')

class StablePCP:
    """Stable principal component pursuit (stable version of Robust PCA)

    Dimensionality reduction using Accelerated Proximal Gradient (APG)
    to decompose the input 2D matrix M into a lower rank dense 2D matrix L and sparse
    but not low-rank 2D matrix S and a noise term Z. Here the noise matrix Z = M-L-S and
    satisfying Frobenius norm ||Z|| < detla. The algorithm is tested to be effective
    under the assumption that Z is Gaussian noise.

    Parameters
    ----------
    lamb : positive float
        Sparse component coefficient.
        if user doesn't set it:
            lamb = 1/sqrt(max(M.shape))
        A effective default value from the reference.

    mu0 : positive float
        Coefficient for the singular value thresholding of M
        if user doesn't set it:
            mu0 = min([mu0_init*np.sqrt(2*max(M.shape)), 0.99*||M||2])
        namely, mu0 is chosen between manual value mu0_init*np.sqrt(2*max(M.shape)) and emprical value 0.99*||M||2

    mu0_init : positive float/int
        Coefficient for initial mu0

    mu_fixed : bool
        Flag for whether or not use a fixed mu for iterations

    mu_min : positive float
        minimum mu for thresholding

    sigma : positive float
        The standard deviation of the Gaussian noise N(0,sigma) for generating E

    eta : float
        Decay coefficient for thresholding, 0 < eta < 1

    max_rank : positive int
        The maximum rank allowed in the low rank matrix
        default is None --> no limit to the rank of the low
        rank matrix.

    tol : positive float
        Convergence criterion

    max_iter : positive int
        Maximum iterations for alternating updates

    use_fbpca : bool
        Determine if use fbpca for SVD. fbpca use Fast Randomized SVDself.
        default is False

    fbpca_rank_ratio : float, between (0, 1]
        If max_rank is not given, this sets the rank for fbpca.pca()
        fbpca_rank = int(fbpca_rank_ratio * min(M.shape))

    Attributes:
    -----------
    L : 2D array
            Lower rank dense 2D matrix

    S : 2D array
        Sparse but not low-rank 2D matrix

    converged : bool
        Flag shows if the fit is converged or not

    error : list
        list of errors from iterations


    References
    ----------
    Zhou, Zihan, et al. "Stable principal component pursuit."
        Information Theory Proceedings (ISIT), 2010 IEEE International Symposium on. IEEE, 2010.

    Lin, Zhouchen, et al. "Fast convex optimization algorithms for exact
    recovery of a corrupted low-rank matrix."
        Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP) 61.6 (2009).

    Wei Xiao "onlineRPCA"
        https://github.com/wxiao0421/onlineRPCA/tree/master/rpca

    """

    def __init__(self, lamb=None, mu0=None, mu0_init=1000, mu_fixed=False, mu_min=None, sigma=1, eta = 0.9, max_rank=None, tol=1e-6, max_iter=100, use_fbpca=False, fbpca_rank_ratio=0.2, verbsome=False):
        self.lamb = lamb
        self.mu0 = mu0
        self.mu0_init = mu0_init
        self.mu_fixed = mu_fixed
        self.mu_min = mu_min
        self.sigma = sigma
        self.eta = eta
        self.max_rank = max_rank
        self.tol = tol
        self.max_iter = max_iter
        self.use_fbpca = use_fbpca
        self.fbpca_rank_ratio = fbpca_rank_ratio
        self.converged = None
        self.error = []

    def s_tau(self, X, tau):
        """Shrinkage operator
            Sτ [x] = sign(x) max(|x| − τ, 0)

        Parameters
        ----------
        X : 2D array
            Data for shrinking

        tau : positive float
            shrinkage threshold

        Returns
        -------
        shirnked 2D array
        """

        return np.sign(X)*np.maximum(np.abs(X)-tau,0)


    def fit(self, M):
        """Stable PCP fit.
        A Gaussian noise is assumed.

        Parameters
        ----------
        M : 2D array
            2D array for docomposing
        """

        size = M.shape

        # initialize L, S and t
        L0, L1 = np.zeros(size), np.zeros(size)
        S0, S1 = np.zeros(size), np.zeros(size)
        t0, t1 = 1, 1

        # if lamb and mu are not set, set with default values
        if self.mu_fixed:
            self.mu0 = np.sqrt(2*np.max(size))*self.sigma

        elif self.mu0==None:
            self.mu0 = np.min([self.mu0_init*np.sqrt(2*np.max(size)), 0.99*np.linalg.norm(M, 2)])
            if self.mu_min==None:
                self.mu_min = np.sqrt(2*np.max(size))*self.sigma

        mu = self.mu0 * 1

        if self.lamb==None:
            self.lamb = 1/np.sqrt(np.max(size))

        #
        for i in range(self.max_iter):
            YL = L1 + (t0-1)/t1*(L1-L0)
            YS = S1 + (t0-1)/t1*(S1-S0)

            # Thresdholding for updating L
            GL = YL - 0.5*(YL+YS-M)
            # singular value decomposition
            if self.use_fbpca:
                if self.max_rank:
                    (u, s, vh) = pca(GL, self.max_rank, True, n_iter = 5)
                else:
                    (u, s, vh) = pca(GL, int(np.min(X.shape)*self.fbpca_rank_ratio), True, n_iter = 5)
            else:
                u, s, vh = np.linalg.svd(GL, full_matrices=False)

            s = s[s>(mu/2)] - mu/2  # threshold by mu/2
            rank = len(s)

            # Max rank cut
            if self.max_rank:
                if rank > self.max_rank:
                    rank = self.max_rank*1
                    s = s[0:rank]

            # update L1, L0
            L0 = L1
            L1 = np.dot(u[:,0:rank]*s, vh[0:rank,:])

            # Thresdholding for updating S
            GS = YS - 0.5*(YL+YS-M)
            # update S0, SL
            S0 = S1
            S1 = self.s_tau(GS, self.lamb*mu/2) # threshold by lamb*mu/2

            # update t0, t1
            t0 = t1
            t1 = (1+np.sqrt(4*t1**2+1))/2

            if not self.mu_fixed:
                # update mu
                mu = np.max([self.eta*mu, self.mu_min])

            # Check Convergence
            EA = 2*(YL-L1)+(L1+S1-YL-YS)
            ES = 2*(YS-S1)+(L1+S1-YL-YS)
            Etot = np.sqrt(np.linalg.norm(EA)**2+np.linalg.norm(ES)**2)
            self.error.append(Etot)
            if Etot <= self.tol:
                break

        # Print if the fit is converged
        if Etot > self.tol:
            print('Not converged within %d iterations!'%self.max_iter)
            print('Total error: %f, allowed tolerance: %f'%(Etot, self.tol))
            self.converged = False
        else:
            print('Converged!')
            self.converged = True

        self.L, self.S, self.rank = L1, S1, rank

    def get_low_rank(self):
        '''Return the low rank matrix

        Returns:
        --------
        L : 2D array
            Lower rank dense 2D matrix
        '''
        return self.L

    def get_sparse(self):
        '''Return the sparse matrix

        Returns:
        --------
        S : 2D array
            Sparse but not low-rank 2D matrix
        '''
        return self.S

    def get_rank(self):
        '''Return the rank of low rank matrix

        Returns:
        rank : int
            The rank of low rank matrix
        '''
        return self.rank
