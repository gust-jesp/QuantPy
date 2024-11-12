import numpy as np
import pandas as pd
from typing import Union
from scipy.linalg import norm

def _rBar(sampleCov: np.ndarray) -> np.ndarray:

    """
    Helper function used in the linear shrinkage estimator.
    """

    n = sampleCov.shape[0]
    sampleVar = np.diag(sampleCov)
    sqrtVar = np.sqrt(sampleVar)
    rBar = (np.sum(sampleCov / np.outer(sqrtVar, sqrtVar))-n) / (n * (n-1))

    return rBar

def constCorrMatrix(returns: Union[np.ndarray, pd.Series]) -> np.ndarray:

    """
    Calculate the constant correlation matrix given a sample covariance 
    matrix of size NxN. The constant correlation matrix may be used as a 
    shrinkage target.

    Input
    ---
        returns:
           The return series of constituents with shape TxN, where T is number 
           of observations and N is number of constituents.

    Returns
    ---
        constCorrTarget:
            Constant correlation matrix of size NxN.
    """

    T = returns.shape[0]
    t = T-1
    
    sampleCov = (returns.T @ returns) / t
    sampleVar = np.diag(sampleCov)
    sqrtVar = np.sqrt(sampleVar)
    rBar = _rBar(sampleCov) 
    constCorrTarget = rBar * np.outer(sqrtVar, sqrtVar)
    np.fill_diagonal(constCorrTarget, sampleVar)

    return constCorrTarget

def shrinkageIntensity(returns: Union[np.ndarray, pd.Series],
                       target: np.ndarray):

    """
    Calculate shrinkage intensity given a series of returns and a shrinkage 
    target.

    Input
    ---
        returns:
            The return series of constituents with shape TxN, where T is number 
            of observations and N is number of constituents.

        target:
            Shrinkage target of size NxN.

    Returns:
    ---
        shrinkageIntensity:
            The shrinkage intensity used to shrink the sample covariance 
            toward the shrinkage target.
    """
    
    T = returns.shape[0]
    t = T-1
    
    sampleCov = (returns.T @ returns) / t
    sampleVar = np.diag(sampleCov)
    sqrtVar = np.sqrt(sampleVar)

    returns2 = returns**2
    sampleCov2 = (returns2.T @ returns2) / t
    piMat = sampleCov2 - sampleCov**2
    piHat = np.sum(piMat)
    gammaHat = norm(sampleCov - target, "fro")**2

    rhoDiag = np.sum(np.diag(piMat))
    termOne = (returns**3).T @ returns / t
    termTwo = np.outer(sampleVar, sampleVar) * sampleCov
    thetaMat = termOne - termTwo
    np.fill_diagonal(thetaMat, 0.0)
    rhoOff = _rBar(sampleCov) * np.sum(np.outer(1/sqrtVar, sqrtVar)*thetaMat)
    rhoHat = rhoDiag + rhoOff

    kappaHat = (piHat - rhoHat) / gammaHat
    kappaHat /= t

    if kappaHat >= 1.0: intensity = 1.0
    else: intensity = kappaHat

    return intensity

def linearShrinkageCovMatrix(sampleCov: Union[np.ndarray, pd.Series],
                            target: Union[np.ndarray, pd.Series],
                            intensity) -> np.ndarray:
    """
    Lienar shrinkage of sample covariance toward a given target matrix. The 
    intensity determines how much the sample covariance is shrunk toward the 
    target.

    Input
    ---
        sampleCov:
            Sample covariance of observations.

        target:
            The shrinkage target exogenously chosen.

        intensity:
            The intensity used to shrink sample toward target.

    Returns
    ---
        A covariance matrix estimate using the linear shrinkage estimator.

    """

    if sampleCov.shape != target.shape:
        raise ValueError("Sample covariance and target is not same shape.")

    return sampleCov * intensity + target * (1-intensity)

class Covariance:

    """
    This class allows for estimating the sample-covariance using different 
    approaches.

    1. sample covariance.
    2. linear shrinkage using constant-correlation matrix.

    Input
    ---
        s:
            Matrix of size NxT where N is number of constituents and T is 
            number of observations.
    """

    def __init__(self,
                 s: Union[np.ndarray, pd.Series]) -> None:

        self.s = s
        self.processed = False

        # process observations
        self.s = self.s - self.s.mean(axis=0) # demean returns

    def sampleCovariance(self) -> None:
        self.cov = np.cov(self.s, rowvar=False)

    def linearShrinkageConstCorr(self):
        sampleCov = np.cov(self.s, rowvar=False)
        target = constCorrMatrix(self.s)
        intensity = shrinkageIntensity(self.s, target)
        self.cov = linearShrinkageCovMatrix(sampleCov, target, intensity)

