import numpy as np
from typing import List
from operator import add
from toolz import reduce, partial 
from scipy.optimize import fmin_slsqp 
from rank import spectral_rank, universal_rank

"""
Convex (Simplex) Regression  

credit to 'Matheus Facure Alves': 
https://matheusfacure.github.io/python-causality-handbook/15-Synthetic-Control.html
"""
def loss_w(W, X, y) -> float: 
	return np.sqrt(np.mean((y - X.dot(W))**2))

lambda x: np.sum(x) - 1 

def cvx_regr(X, y):    
	w_start = [1/X.shape[1]]*X.shape[1]
	weights = fmin_slsqp(partial(loss_w, X=X, y=y),
						 np.array(w_start),
						 f_eqcons=lambda x: np.sum(x) - 1,
						 bounds=[(0.0, 1.0)]*len(w_start),
						 disp=False)
	return weights 

""" 
Principal Component Regression (PCR) 
"""
def pcr(X, y, max_rank=None, t=None): 
	(u, s, v) = np.linalg.svd(X, full_matrices=False)
	if max_rank is not None: 
		rank = max_rank 
	elif t is not None: 
		rank = spectral_rank(s, t=t)
	else: 
		(n1, n2) = X.shape 
		ratio = min(n1, n2) / max(n1, n2)
		rank = universal_rank(s, ratio=ratio)
	s_rank = s[:rank]
	u_rank = u[:, :rank]
	v_rank = v[:rank, :] 
	weights = ((v_rank.T/s_rank) @ u_rank.T) @ y
	return weights

