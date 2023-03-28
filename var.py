import numpy as np

"""
Threshold errors (for numerical stability)
"""
def err_threshold(err, thresh=1e-10): 
	err[np.abs(err)<=1e-10] = 0
	return err 

"""
Homoskedastic variance estimator
""" 
def var_homo(y_n, y_t, Y0, w_hz, w_vt, Hu_perp, Hv_perp): 
	# rank 
	R = int(np.trace(Y0@np.linalg.pinv(Y0)))

	# hz 
	if R==len(y_t): 
		var_hz = 0
		sigma2_hz = 0  
	else: 
		scale = len(y_t) - R
		err_hz = Hu_perp @ y_t
		err_hz = err_threshold(err_hz)
		sigma2_hz = (1/scale) * (np.linalg.norm(err_hz)**2)
		var_hz = sigma2_hz * np.dot(w_vt, w_vt)

	# vt 
	if R==len(y_n):
		var_vt = 0
		sigma2_vt = 0  
	else:
		scale = len(y_n) - R 
		err_vt = Hv_perp @ y_n
		err_vt = err_threshold(err_vt)
		sigma2_vt = (1/scale) * (np.linalg.norm(err_vt)**2)
		var_vt = sigma2_vt * np.dot(w_hz, w_hz)

	# interaction  
	A = (sigma2_hz * sigma2_vt) * (np.linalg.pinv(Y0)@np.linalg.pinv(Y0.T)) 
	trA = np.trace(A)
	return (var_hz, var_vt, trA)

"""
Jackknife variance estimator
""" 
def var_jack(y_n, y_t, Y0, w_hz, w_vt, Hu_perp, Hv_perp):  
	# rank 
	R = int(np.trace(Y0@np.linalg.pinv(Y0)))

	# hz 
	if R==len(y_t): 
		var_hz = 0 
		sigma_hz = np.zeros((len(y_t), len(y_t)))
	else: 
		H_inv_hz = np.linalg.pinv((Hu_perp*Hu_perp*np.eye(Hu_perp.shape[0])))
		err_hz = (Hu_perp@y_t) * (Hu_perp@y_t)
		err_hz = err_threshold(err_hz)
		sigma_hz = np.diag(H_inv_hz@err_hz) 
		var_hz = np.dot(w_vt, sigma_hz@w_vt)

	# vt 
	if R==len(y_n): 
		var_vt = 0 
		sigma_vt = np.zeros((len(y_n), len(y_n)))
	else: 
		H_inv_vt = np.linalg.pinv((Hv_perp*Hv_perp*np.eye(Hv_perp.shape[0])))
		err_vt = (Hv_perp@y_n) * (Hv_perp@y_n)
		err_vt = err_threshold(err_vt)
		sigma_vt = np.diag(H_inv_vt@err_vt) 
		var_vt = np.dot(w_hz, sigma_vt@w_hz)

	# interaction 
	A = np.linalg.pinv(Y0) @ sigma_hz @ np.linalg.pinv(Y0.T) @ sigma_vt
	trA = np.trace(A)
	return (var_hz, var_vt, trA)

"""
HRK variance estimator 
"""
def var_hrk(y_n, y_t, Y0, w_hz, w_vt, Hu_perp, Hv_perp):  
	# hz 
	H_inv_hz = np.linalg.pinv((Hu_perp*Hu_perp))
	err_hz = (Hu_perp@y_t) * (Hu_perp@y_t)
	err_hz = err_threshold(err_hz)
	sigma_hz = np.diag(H_inv_hz@err_hz) 
	var_hz = np.dot(w_vt, sigma_hz@w_vt)

	# vt 
	H_inv_vt = np.linalg.pinv((Hv_perp*Hv_perp))
	err_vt = (Hv_perp@y_n) * (Hv_perp@y_n)
	err_vt = err_threshold(err_vt)
	sigma_vt = np.diag(H_inv_vt@err_vt) 
	var_vt = np.dot(w_hz, sigma_vt@w_hz)

	# interaction 
	A = np.linalg.pinv(Y0) @ sigma_hz @ np.linalg.pinv(Y0.T) @ sigma_vt 
	trA = np.trace(A)
	return (var_hz, var_vt, trA)

"""
Compute variance estimate 
"""
def var_est(y_n, y_t, Y0, w_hz, w_vt, Hu_perp, Hv_perp, v_alg): 
	if v_alg=='homoskedastic': 
		(var_hz, var_vt, trA) = var_homo(y_n, y_t, Y0, w_hz, w_vt, Hu_perp, Hv_perp)
	if v_alg=='jackknife':
		(var_hz, var_vt, trA) = var_jack(y_n, y_t, Y0, w_hz, w_vt, Hu_perp, Hv_perp)
	elif v_alg=='HRK': 
		(var_hz, var_vt, trA) = var_hrk(y_n, y_t, Y0, w_hz, w_vt, Hu_perp, Hv_perp)
	return (var_hz, var_vt, trA)
