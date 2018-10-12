import numpy as np

class RobustPCA:
    """Robust principal component analysis (Robust PCA)
    
    Dimensionality reduction using alternating directions methods 
    to decompose the input 2D matrix M into a lower rank dense 2D matrix L and sparse
    but not low-rank 2D matrix S.
    
    Parameters
    ----------
    lamb : positive float
        Sparse component coefficient.
        if user doesn't set it:
            lamb = 1/sqrt(max(M.shape))
        A effective default value from the reference. 
    
    mu : positive float 
        Coefficient for augmented lagrange multiplier
        if user doesn't set it:
            n1, n2 = M.shape
            mu = n1*n2/4/norm1(M) # norm1(M) is M's l1-norm
        A effective default value from the reference.
        
    tol : positive float
        Convergence tolerance
        
    max_iter : positive int
        Maximum iterations for alternating updates
        
    Attributes:
    -----------
    L : 2D array
            Lower rank dense 2D matrix

    S : 2D array
        Sparse but not low-rank 2D matrix
        
    converged : bool
        Flag shows if the fit is converged or not
    
    
    Reference:
    ----------
    `Emmanuel J. Candes, Xiaodong Li, Yi Ma, and John Wright` 
    "Robust Principal Component Analysis?"
    https://statweb.stanford.edu/~candes/papers/RobustPCA.pdf
        
    """
    
    def __init__(self, lamb=None, mu=None, tol=1e-6, max_iter = 10000):
        self.lamb = lamb
        self.mu = mu
        self.tol = tol
        self.max_iter = max_iter
        self.converged = None
    
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
    
    
    def d_tau(self, X):
        """Singular value thresholding operator
            Dτ (X) = USτ(Σ)V∗, where X = UΣV∗
            
        Parameters
        ----------
        X : 2D array
            Data for shrinking
                        
        Returns
        -------
        thresholded 2D array
        """
        
        # singular value decomposition 
        u, s, vh = np.linalg.svd(X, full_matrices=False)
        
        # Shrinkage of singular values
        tau = 1.0/self.mu
        s = s[s>tau] - tau
        rank = len(s)
        
        # reconstruct thresholded 2D array
        return  np.dot(u[:, 0:rank] * s, vh[0:rank,:]), rank

    
    
    def fit(self, M):
        """Robust PCA fit
                        
        Parameters
        ----------
        M : 2D array
            2D array for docomposing
                        
        Returns
        -------
        L : 2D array
            Lower rank dense 2D matrix
        
        S : 2D array
            Sparse but not low-rank 2D matrix
        """
        
        size = M.shape
        
        # initialize S and Y (Lagrange multiplier)
        S = np.zeros(size)
        Y = np.zeros(size)
        
        # if lamb and mu are not set, set with default values
        if self.mu==None:
            self.mu = np.prod(size)/4.0/np.sum(np.abs(M))
        if self.lamb==None:
            self.lamb = 1/np.sqrt(np.max(size))
            
        # Alternating update
        for i in range(self.max_iter):
            L, rank = self.d_tau(M-S+1.0/self.mu*Y)
            S= self.s_tau(M-L+1.0/self.mu*Y, self.lamb/self.mu)
            
            # Calculate residuals
            residuals = M-L-S
            residuals_sum = np.sum(np.abs(residuals))
            
            # Check convergency
            if residuals_sum <= self.tol:
                break
                
            Y = Y + self.mu*residuals
            
        # Check if the fit is converged 
        if residuals_sum > self.tol:
            print('Not converged!')
            print('Total error: %f, allowed tolerance: %f'%(residuals_sum, self.tol))
            self.converged = False
        else:
            print('Converged!')
            self.converged = True
            
        self.L, self.S, self.rank = L, S, rank
        
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
      