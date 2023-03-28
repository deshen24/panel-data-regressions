import sys, os
import warnings 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
import seaborn as sns 
from sklearn.linear_model import LinearRegression 
from regr import pcr 
from var import var_est
from rank import svt, spectral_rank

warnings.filterwarnings("ignore")

if len(sys.argv)!=3:
	print('Usage: python {} <dataset name> <# iterations>'.format(sys.argv[0]))
	sys.exit()
data = str(sys.argv[1])
n_iters = int(sys.argv[2])
print() 
print("*** {} ***".format(data))
print("---------------------")

# data info 
info = {'prop99': {'treated_unit': 'California', 
				   'ymin': 0, 'ymax': 140, 
				   'outcome_var': 'cigarette sales', 'iv': 'Prop 99', 'interval': 5},
		'basque': {'treated_unit': 'Basque', 
				   'ymin': 0, 'ymax': 14,  
				   'outcome_var': 'gdp', 'iv': 'terrorism', 'interval': 10},
		'germany': {'treated_unit': 'West Germany', 
					'ymin': 0, 'ymax': 36000, 
				    'outcome_var': 'gdp', 'iv': 'reunification', 'interval': 10}
		}
treated_unit = info[data]['treated_unit']
(ymin, ymax) = (info[data]['ymin'], info[data]['ymax'])
(outcome_var, iv, interval) = (info[data]['outcome_var'], info[data]['iv'], info[data]['interval'])
output_dir = 'output/simulation'
os.makedirs(output_dir, exist_ok=True)

""" 
Study Data 
"""  
fname_pre = 'data/{}/pre_outcomes.csv'.format(data)
fname_post = 'data/{}/post_outcomes.csv'.format(data)
df_pre = pd.read_csv(fname_pre)
df_post = pd.read_csv(fname_post) 

# unit indices 
units = df_post.unit.unique() 
donors = units[units!=treated_unit]

# time indices   
pre_cols = list(df_pre.drop(columns=['unit']).columns)
post_cols = list(df_post.drop(columns=['unit']).columns) 

# get observed data   
Y0_obs = df_pre.loc[df_pre['unit'].isin(donors)].drop(columns=['unit']).values
y_n_obs = df_pre.loc[df_pre['unit']==treated_unit].drop(columns=['unit']).values.flatten()
y_t_obs = df_post.loc[df_post['unit'].isin(donors), post_cols[0]].values.flatten()

"""
Calibrate to case study data 
""" 
# take low rank approximation of Y0
(u, s, v) = np.linalg.svd(Y0_obs, full_matrices=False) 
k = spectral_rank(s, t=0.999) # retain 99.9% of spectral energy 
print("spectral rank: {}".format(k))
(Y0, Hu, Hv, Hu_perp, Hv_perp) = svt(Y0_obs, max_rank=k) 

# get underlying regression models 
regr = LinearRegression(fit_intercept=False) 
alpha = regr.fit(Y0_obs, y_t_obs).coef_ 
beta = regr.fit(Y0_obs.T, y_n_obs).coef_ 

# compute underlying responses 
y_n = Y0.T @ beta 
y_t = Y0 @ alpha 

# errors 
(N0, T0) = Y0.shape
sigma_t = np.linalg.norm(Hu_perp@y_t_obs)**2 / (N0-k)
sigma_t = np.diag(sigma_t*np.ones(N0))
sigma_n = np.linalg.norm(Hv_perp@y_n_obs)**2 / (T0-k)
sigma_n = np.diag(sigma_n*np.ones(T0))
print("data configuration: ({}, {})".format(N0, T0))

""" 
Estimation 
""" 
var_ests = ['var_hz', 'var_vt', 'var_dr', 'var_mod'] # var_mod: conservative version of var_dr 
estimands = ['mu_hz', 'mu_vt', 'mu_dr'] 

# confidence interval parameter (95%)
z = 1.96 

"""
coverage methods 
"""
# update coverage probability 
def update_count(z, pred, var, estimand, var_type, estimand_type, cp_dict): 
	se = z * np.sqrt(var) 
	lb = pred - se 
	ub = pred + se 
	if (estimand >= lb) and (estimand <= ub): 
		cp_dict[var_type][estimand_type] += 1 
	return cp_dict

# update coverage length
def iv_len(z, var, y): 
	return (2*z*np.sqrt(var)) / np.abs(y)

# initialize
cp_dict = {var_est: {estimand: 0 for estimand in estimands} for var_est in var_ests} 
al_dict = {var_est: np.zeros(n_iters) for var_est in var_ests}

# iterate through all samples 
print("Computing...")
for i in range(n_iters): 

	# sample new data 
	y_n_iter = np.random.multivariate_normal(y_n, sigma_n)
	y_t_iter = np.random.multivariate_normal(y_t, sigma_t)

	# define estimands 
	mu_hz = np.dot(y_n_iter, Hv@alpha)
	mu_vt = np.dot(y_t_iter, Hu@beta) 
	mu_dr = np.dot(alpha, Y0.T@beta) 

	# point estimation 
	regr = LinearRegression(fit_intercept=False) 
	alpha_hat = regr.fit(Y0, y_t_iter).coef_ 
	beta_hat = regr.fit(Y0.T, y_n_iter).coef_ 
	pred = np.dot(y_n_iter, alpha_hat) 

	# uncertainty estimation 
	(var_hz, var_vt, trA) = var_est(y_n_iter, y_t_iter, Y0, 
									alpha_hat, beta_hat, 
									Hu_perp, Hv_perp, 
								  	v_alg='homoskedastic')
	var_dr = var_hz + var_vt - trA 
	var_mod = var_hz + var_vt if (trA>np.max([var_hz, var_vt])) else var_dr.copy()

	# update hz variance estimator wrt all 3 estimands 
	cp_dict = update_count(z, pred, var_hz, mu_hz, 'var_hz', 'mu_hz', cp_dict)
	cp_dict = update_count(z, pred, var_hz, mu_vt, 'var_hz', 'mu_vt', cp_dict)
	cp_dict = update_count(z, pred, var_hz, mu_dr, 'var_hz', 'mu_dr', cp_dict)
	al_dict['var_hz'][i] = iv_len(z, var_hz, pred) 

	# update vt variance estimator wrt all 3 estimands 
	cp_dict = update_count(z, pred, var_vt, mu_hz, 'var_vt', 'mu_hz', cp_dict)
	cp_dict = update_count(z, pred, var_vt, mu_vt, 'var_vt', 'mu_vt', cp_dict)
	cp_dict = update_count(z, pred, var_vt, mu_dr, 'var_vt', 'mu_dr', cp_dict)
	al_dict['var_vt'][i] = iv_len(z, var_vt, pred)

	# update dr variance estimator wrt all 3 estimands 
	cp_dict = update_count(z, pred, var_dr, mu_hz, 'var_dr', 'mu_hz', cp_dict)
	cp_dict = update_count(z, pred, var_dr, mu_vt, 'var_dr', 'mu_vt', cp_dict)
	cp_dict = update_count(z, pred, var_dr, mu_dr, 'var_dr', 'mu_dr', cp_dict)
	al_dict['var_dr'][i] = iv_len(z, var_dr, pred)

	# update modified variance estimator wrt all 3 estimands 
	cp_dict = update_count(z, pred, var_mod, mu_hz, 'var_mod', 'mu_hz', cp_dict)
	cp_dict = update_count(z, pred, var_mod, mu_vt, 'var_mod', 'mu_vt', cp_dict)
	cp_dict = update_count(z, pred, var_mod, mu_dr, 'var_mod', 'mu_dr', cp_dict)
	al_dict['var_mod'][i] = iv_len(z, var_mod, pred)

print("Computing completed!")
print()

""" 
Coverage results
""" 
print("Printing coverage results...")
with open(os.path.join(output_dir, '{}_coverage.txt'.format(data)), 'w') as f: 
	lines = [] 
	lines.append("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	lines.append("=== {}: Coverage ===".format(data))
	lines.append("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~") 

	# iterate through variance estimator types  
	for var_est in var_ests: 
		lines.append("*** {} ***".format(var_est))

		# iterate through estimand types 
		for estimand in estimands: 

			# compute CP + AL 
			CP = cp_dict[var_est][estimand] / n_iters 
			AL = np.mean(al_dict[var_est])
			lines.append("{} (CP) = {:.2f}".format(estimand, CP))
			lines.append("{} (AL) = {:.3f}".format(estimand, AL))
			lines.append("--------------")
		lines.append("\n")
	f.write("\n".join(lines))
print("Done!")
