# Author: Mikolaj Mizera <mikolajmizera@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.base import clone

class MUDRAEstimator():
    
    """Multi-Descriptor Read Across (MuDRA)
    
    Flexible implementation of MuDRA algorithm.
    
    Read more in the :ref:`Alves, V. M., Golbraikh, A., Capuzzi, S. J., Liu, K., Lam, W. I., Korn, D. R., ... & Tropsha, A. (2018). Multi-Descriptor Read Across (MuDRA): A Simple and Transparent Approach for Developing Accurate Quantitative Structure–Activity Relationship Models. Journal of chemical information and modeling, 58(6), 1214-1223.`.
    
    Parameters
    ----------
    
    estimator_type : {'regressor', 'classifier'}
        Type of estimation task: `regressor` for continous values prediction,
        `classifier` for classification.
    
    kwargs : dict, optional
        Optional keyword arguments passed to the base estimator.
        
    Attributes
    ----------
    
    base_estimator : scikit-learn estimator
        Base estimator used to create an ensemble.
        
    estimators: list of scikit-learn estimators
        A list of estimators in an ensemble.
        
    """
    def __init__(self, estimator_type,  **kwargs):
        
        if estimator_type =='regressor':
            self.base_estimator = KNeighborsRegressor(**kwargs)
        elif estimator_type =='classifier':
            self.base_estimator = KNeighborsClassifier(**kwargs)
        else:
            raise RuntimeError('Estimator type must be "regressor" or "classifier"!')
    
    def fit(self, X, y):
        
        """Build an esnemble of N estimators from training set 
        [(X1, y1) ... (XN, yN)].
                
        Parameters
        ----------
        
        X : list of N ndarrays of shape = [n_samples, n_features]
            The training input samples. Each of N ndarrrays in a list contains
            sample description of molecule in given descriptor space.
            
        y : ndarray of shape = [n_samples]
            The target values (continous values for regression, class labels
            for classification).
            
        Returns
        -------
        
        estimators : list of N estimators
            List of N estimators, each built with correspoding samples description.
            
        """
    
        ndims = np.unique([x.ndim for x in X])
        if len(ndims)>1:
            raise RuntimeError('All desc. spaces must be of the same ndim, provided: %r'%ndims)
                
        lens = np.unique([len(x) for x in X])
        if len(lens)>1:
            raise RuntimeError('All desc. spaces must be of the same length, provided: %r'%lens)
        
        self.estimators = [clone(self.base_estimator).fit(x, y) for x in X]
        
        return self.estimators
        
    def predict(self, X):
        
        """Predict target for X.
        
        The predicted target of an input sample is computed as the
        value of nearest neighbor among all estimators in ensemble.
        
        Parameters
        ----------
        
        X : list of ndarrays of shape = [n_samples, n_features]
            The training input samples. Each ndarray in a list contains
            sample description in different (descriptor) space.
            
        Returns
        -------
        
        y : array of shape = [n_samples]
            The predicted values.
            
        min_distances : array of shape = [ensemble_size, n_samples]
            The minimal distances reported by each of estimators in ensemble.
            
        min_indices : array of shape = [ensemble_size, n_samples]
            The indices of target variable associated with nearest neighbor.

        """
        
        preds = np.array([est.predict(x) for x, est in zip(X, self.estimators)])
        dists = np.array([est.kneighbors(x)[0].mean(axis=1) for x, est in zip(X, self.estimators)])
        
        min_indices = dists.argmin(axis=0)
        min_distances = dists.min(axis=0)
        y_pred = preds[min_indices, np.arange(preds.shape[1])]
        
        return y_pred, min_distances, min_indices




    