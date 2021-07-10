import numpy as np
from numpy.linalg import norm
from scipy.stats import multivariate_normal


class GaussianUtils:

    def __init__(self, k, x, x_true):
        self.K = k
        self.X = x
        self.x_true = x_true
        D = self.X.shape[1]
        self.pks = np.array([[1 / self.K for i in range(self.K)]])
        self.mean_k = np.random.rand(self.K, D)
        self.cov_S = [np.cov(self.X.T)*np.identity(self.X.shape[1]) for z in range(k)]

    def log_likelihood_update(self):
         return np.sum(np.log(np.sum(self.normal_distribution(), axis=1)))
    
    

    def normal_distribution(self):
        p_x = np.zeros((self.X.shape[0], self.K))
        I = np.identity(self.X.shape[1])  # I vector
        self.cov_S = np.array([ I*Sk for Sk in self.cov_S])
        for k in range(self.K):
            p = multivariate_normal(self.mean_k[k], self.cov_S[k], allow_singular=True).pdf(self.X)
            p_x[:, k] = p
        
        p_x = self.pks * p_x
           
        return p_x

    def E_step(self):
        
        first = self.normal_distribution()
        return first / np.sum(first, axis=1).reshape(-1, 1)

    def M_step(self, e_step):
        self.m_update(e_step)
        self.covS_update(e_step)
        self.pks_update(e_step)

    def covS_update(self, e_step):

        parameterD = self.X.shape[1]  # pixels

        for k in range(self.K):
            gamma = e_step[:, k].reshape(-1, 1)
            sum_estep = np.sum(gamma, axis=0)
            X_m = self.X - self.mean_k[k]
            X_m = np.square(X_m)  # ( Xnd - μkd )^2

            numerator_k = np.sum(np.sum(X_m * gamma, axis=1), axis=0)  # apply the modify and change every vector sk
            self.cov_S[k] = numerator_k / (parameterD * sum_estep)

    def m_update(self, e_step):

        classes_k = e_step.shape[1]
        for k in range(classes_k):
            gamma = e_step[:, k].reshape(-1, 1)
            sum_estep = np.sum(gamma)
            numerator = np.sum(gamma.T.dot(self.X), axis=0)

            self.mean_k[k] = numerator / sum_estep

    # update Pks
    def pks_update(self, e_step):
        self.pks = np.sum(e_step, axis=0) / e_step.shape[0]
        self.pks = self.pks.reshape(1,-1)

    # check if we have convergence
    def checkConvergence(self, like1, like2, tol):
        return int(like1 - like2 < tol)

    def ExpectationMaximizationAlgorithm(self, tol, iterations):

        oldLikelihood = self.log_likelihood_update()
        iter = 0
        while iter < iterations:
            E_STEP = self.E_step()
            self.M_step(E_STEP)
            newLikelihood = self.log_likelihood_update()

            if newLikelihood-oldLikelihood<0:
                print("Error!")
                return 
            if self.checkConvergence(newLikelihood, oldLikelihood, tol):
                break
            oldLikelihood = newLikelihood
            iter += 1
        return E_STEP, self.mean_k
    
    def error_reconstruction(self, x_r):
        N = self.X.shape[0]
        self.X = self.X * 255
        result = self.X - x_r
        error = np.sqrt(np.sum(np.power(result, 2))) / N
        return error
    



