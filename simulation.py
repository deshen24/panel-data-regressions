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
from rank import svt

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
				   'ymin': 0, 'ymax': 140, 'k': 4, 
				   'outcome_var': 'cigarette sales', 'iv': 'Prop 99', 'interval': 5},
		'basque': {'treated_unit': 'Basque', 
				   'ymin': 0, 'ymax': 14, 'k': 2, 
				   'outcome_var': 'gdp', 'iv': 'terrorism', 'interval': 10},
		'germany': {'treated_unit': 'West Germany', 
					'ymin': 0, 'ymax': 36000, 'k': 4,
				    'outcome_var': 'gdp', 'iv': 'reunification', 'interval': 10}
		}
treated_unit = info[data]['treated_unit']
(ymin, ymax) = (info[data]['ymin'], info[data]['ymax'])
k = info[data]['k']
(outcome_var, iv, interval) = (info[data]['outcome_var'], info[data]['iv'], info[data]['interval'])
output_dir = 'output/simulation/{}'.format(data) 
os.makedirs(output_dir, exist_ok=True)

""" 
Study Data 
"""  
fname_pre = 'data/{}/pre_outcomes.csv'.format(data)
fname_post = 'data/{}/post_outcomes.csv'.format(data)
df_pre = pd.read_csv(fname_pre)
df_post = pd.read_csv(fname_post) 

# unit data 
units = df_post.unit.unique() 
donors = units[units!=treated_unit]

# time data  
pre_cols = list(df_pre.drop(columns=['unit']).columns)
post_cols = list(df_post.drop(columns=['unit']).columns) 

# pretreatment data   
Y0_obs = df_pre.loc[df_pre['unit'].isin(donors)].drop(columns=['unit']).values
y_n_obs = df_pre.loc[df_pre['unit']==treated_unit].drop(columns=['unit']).values.flatten()

"""
Simulation data
""" 
# underlying values (artificially designate final pretreatment period as pseudo treated period)
Y0 = Y0_obs[:, :-1]
y_n = y_n_obs[:-1]
y_t = Y0_obs[:, -1]
y_nt = y_n_obs[-1]

# errors 
(N0, T0) = Y0.shape
sigma_n = np.diag(np.std(y_n)*np.ones(T0)) 
sigma_t = np.diag(np.std(y_t)*np.ones(N0))
print("data configuration: ({}, {})".format(N0, T0))

""" 
Estimation 
""" 
directions = ['hz', 'vt', 'mixed', 'mod']
algs = ['ols', 'pcr'] 

# parameters
z = 1.96 
params = {'ols': 
				{'max_rank': min(N0, T0),
				 't': 1.0},
		  'pcr':
		  		{'max_rank': None,
		  		 't': None}
		 }

# pre-computations  
alg_data_dict = {alg: {'Y0': np.zeros((N0, T0)), 
					   'Hu': np.zeros((N0, N0)),
					   'Hv': np.zeros((T0, T0)), 
					   'Hu_perp': np.zeros((N0, N0)),
					   'Hv_perp': np.zeros((T0, T0))} for alg in algs}
for alg in algs: 
	max_rank = k if alg=='pcr' else min(N0, T0)
	(Y0_alg, Hu_alg, Hv_alg, Hu_perp_alg, Hv_perp_alg) = svt(Y0, max_rank=max_rank)
	alg_data_dict[alg]['Y0'] = Y0_alg 
	alg_data_dict[alg]['Hu'] = Hu_alg 
	alg_data_dict[alg]['Hv'] = Hv_alg 
	alg_data_dict[alg]['Hu_perp'] = Hu_perp_alg 
	alg_data_dict[alg]['Hv_perp'] = Hv_perp_alg 
	rank = int(np.round(np.trace(Y0_alg @ np.linalg.pinv(Y0_alg))))
	print("{} rank: {}".format(alg, rank))
print() 

# check if HRK estimator is valid 
var_algs = {alg: ['homoskedastic', 'jackknife'] for alg in algs}
for alg in algs: 
	u_max = np.max(1-np.diagonal(alg_data_dict[alg]['Hu_perp']))
	v_max = np.max(1-np.diagonal(alg_data_dict[alg]['Hv_perp']))
	hrk_count = 0 
	if u_max<1/2: 
		hrk_count += 1 
	if v_max<1/2: 
		hrk_count += 1 
	if hrk_count==2: 
		var_algs[alg].append('HRK')

"""
coverage methods 
"""
# update coverage probability 
def update_count(z, var, y_hat, y_obs, alg, v_alg, d, count_dict): 
	se = z * np.sqrt(var) 
	lb = y_hat - se 
	ub = y_hat + se 
	if (y_obs >= lb) and (y_obs <= ub): 
		count_dict[alg][v_alg][d] += 1
	return count_dict

# update coverage length
def iv_len(z, var, y): 
	return (2*z*np.sqrt(var)) / np.abs(y)

# initialize
est_dict = {alg: np.zeros(n_iters) for alg in algs}
res_dict = {alg: {d: np.zeros(n_iters) for d in ['hz', 'vt']} for alg in algs}
len_dict = {alg: {v_alg: {d: np.zeros(n_iters) for d in directions} for v_alg in var_algs[alg]} for alg in algs}
count_dict = {alg: {v_alg: {d: 0 for d in directions} for v_alg in var_algs[alg]} for alg in algs}

# iterate through all samples 
print("Computing...")
for i in range(n_iters): 

	# sample new data 
	y_n_iter = np.random.multivariate_normal(y_n, sigma_n)
	y_t_iter = np.random.multivariate_normal(y_t, sigma_t)

	# OLS 
	if 'ols' in algs: 
		regr = LinearRegression(fit_intercept=False)

		# hz 
		w_hz = regr.fit(Y0, y_t_iter).coef_ 
		hz_err = alg_data_dict['ols']['Hu_perp'] @ y_t_iter
		res_dict['ols']['hz'][i] = np.linalg.norm(hz_err) / np.linalg.norm(y_t_iter)

		# vt 
		w_vt = regr.fit(Y0.T, y_n_iter).coef_ 
		vt_err = alg_data_dict['ols']['Hv_perp'] @ y_n_iter
		res_dict['ols']['vt'][i] = np.linalg.norm(vt_err) / np.linalg.norm(y_n_iter)

		# prediction 
		y_nt_ols = np.dot(y_n_iter, w_hz) 
		est_dict['ols'][i] = y_nt_ols

		# iterate through valid confidence intervals 
		for v_alg in var_algs['ols']: 
			(var_hz, var_vt, trA) = var_est(y_n_iter, y_t_iter, alg_data_dict['ols']['Y0'], 
											w_hz, w_vt, 
											alg_data_dict['ols']['Hu_perp'], 
											alg_data_dict['ols']['Hv_perp'], 
										  	v_alg=v_alg)
			var_mixed = max(0, var_hz + var_vt - trA)
			var_mod = var_hz + var_vt if (trA>np.max([var_hz, var_vt])) else var_mixed.copy()

			# update hz  
			count_dict = update_count(z, var_hz, y_nt_ols, y_nt, 'ols', v_alg, 'hz', count_dict)
			len_dict['ols'][v_alg]['hz'][i] = iv_len(z, var_hz, y_nt_ols)

			# update vt   
			count_dict = update_count(z, var_vt, y_nt_ols, y_nt, 'ols', v_alg, 'vt', count_dict)
			len_dict['ols'][v_alg]['vt'][i] = iv_len(z, var_vt, y_nt_ols)

			# update mixed  
			count_dict = update_count(z, var_mixed, y_nt_ols, y_nt, 'ols', v_alg, 'mixed', count_dict)
			len_dict['ols'][v_alg]['mixed'][i] = iv_len(z, var_mixed, y_nt_ols)

			# update mod 
			count_dict = update_count(z, var_mod, y_nt_ols, y_nt, 'ols', v_alg, 'mod', count_dict)
			len_dict['ols'][v_alg]['mod'][i] = iv_len(z, var_mod, y_nt_ols)  

	# PCR
	if 'pcr' in algs: 

		# hz 
		w_hz = pcr(Y0, y_t_iter, max_rank=k)
		hz_err = alg_data_dict['pcr']['Hu_perp'] @ y_t_iter
		res_dict['pcr']['hz'][i] = np.linalg.norm(hz_err) / np.linalg.norm(y_t_iter)

		# vt 
		w_vt = pcr(Y0.T, y_n_iter, max_rank=k) 
		vt_err = alg_data_dict['pcr']['Hv_perp'] @ y_n_iter
		res_dict['pcr']['vt'][i] = np.linalg.norm(vt_err) / np.linalg.norm(y_n_iter)

		# prediction 
		y_nt_pcr = np.dot(y_n_iter, w_hz) 
		est_dict['pcr'][i] = y_nt_pcr

		# iterate through valid confidence intervals 
		for v_alg in var_algs['pcr']: 
			(var_hz, var_vt, trA) = var_est(y_n_iter, y_t_iter, alg_data_dict['pcr']['Y0'], 
											w_hz, w_vt, 
											alg_data_dict['pcr']['Hu_perp'], 
											alg_data_dict['pcr']['Hv_perp'], 
										  	v_alg=v_alg)
			var_mixed = max(0, var_hz + var_vt - trA)
			var_mod = var_hz + var_vt if (trA>np.max([var_hz, var_vt])) else var_mixed.copy()

			# update hz  
			count_dict = update_count(z, var_hz, y_nt_pcr, y_nt, 'pcr', v_alg, 'hz', count_dict)
			len_dict['pcr'][v_alg]['hz'][i] = iv_len(z, var_hz, y_nt_pcr)

			# update vt   
			count_dict = update_count(z, var_vt, y_nt_pcr, y_nt, 'pcr', v_alg, 'vt', count_dict)
			len_dict['pcr'][v_alg]['vt'][i] = iv_len(z, var_vt, y_nt_pcr)

			# update mixed  
			count_dict = update_count(z, var_mixed, y_nt_pcr, y_nt, 'pcr', v_alg, 'mixed', count_dict)
			len_dict['pcr'][v_alg]['mixed'][i] = iv_len(z, var_mixed, y_nt_pcr)

			# update mod 
			count_dict = update_count(z, var_mod, y_nt_pcr, y_nt, 'pcr', v_alg, 'mod', count_dict)
			len_dict['pcr'][v_alg]['mod'][i] = iv_len(z, var_mod, y_nt_pcr)  

print("Computing completed!")
print()

"""
Prediction errors
""" 
print("Printing and plotting prediction errors...")
with open(os.path.join(output_dir, 'prediction_error.txt'), 'w') as f: 
	lines = [] 
	lines.append("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	lines.append("=== {}: Prediction Errors ===".format(data))
	lines.append("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~") 

	# get true values 
	true_vals = y_nt * np.ones(n_iters)

	# iterate through algorithms
	for alg in algs:
		y_alg = est_dict[alg]
		lines.append("*** {} ***".format(alg))

		# avg. hz in-sample error
		hz_err = np.mean(res_dict[alg]['hz'])
		lines.append("HZ err: {:.2f}".format(hz_err))

		# avg. vt in-sample error
		vt_err = np.mean(res_dict[alg]['vt'])
		lines.append("VT err: {:.2f}".format(vt_err))

		# bias 
		bias = np.mean(y_alg-y_nt) / np.abs(y_nt)
		lines.append("Bias: {:.2f}".format(bias))

		# rmse
		rmse = np.linalg.norm(true_vals-y_alg) / np.linalg.norm(true_vals)
		lines.append("RMSE: {:.2f}".format(rmse)) 
		lines.append("\n")
	f.write("\n".join(lines))

# plot distribution of errors 
fname = os.path.join(output_dir, data)
colors = {
    'ols': 'royalblue',
    'pcr': 'yellowgreen'
}
if data=='germany':
	data_title = 'West Germany Reunification'
elif data=='basque':
	data_title = 'Terrorism in Basque Country'
else:
	data_title = 'CA Proposition 99'
(fig, axes) = plt.subplots(dpi=100)
for alg in algs:
	bias = (est_dict[alg] - y_nt) / np.abs(y_nt)
	sns.distplot(bias, label=alg, color=colors[alg])
plt.axvline(0, linestyle='--', color='black')
plt.legend(loc='best')
plt.title(data_title)
plt.xlabel('Bias')
axes.yaxis.set_major_locator(MaxNLocator(5)) 
axes.xaxis.set_major_locator(MaxNLocator(5)) 
plt.savefig(fname, dpi=100, bbox_inches="tight")
plt.close() 

""" 
Coverage results
""" 
print("Printing coverage results...")
with open(os.path.join(output_dir, 'coverage.txt'), 'w') as f: 
	lines = [] 
	lines.append("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	lines.append("=== {}: Coverage ===".format(data))
	lines.append("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~") 

	# iterate through algorithms
	for alg in algs:
		lines.append("*** {} ***".format(alg))

		# iterate through valid intervals 
		for v_alg in var_algs[alg]: 
			lines.append("== {} ==".format(v_alg)) 

			# iterate through directions 
			for (d, direction) in enumerate(directions): 
				# coverage rate 
				coverage_rate = count_dict[alg][v_alg][direction] / n_iters 
				lines.append("{} rate = {:.2f}".format(direction, coverage_rate))
 
				# coverage lenth 
				coverage_len = np.mean(len_dict[alg][v_alg][direction])
				lines.append("{} len = {:.2f}".format(direction, coverage_len))
				lines.append("--------------")

			lines.append("\n")
	f.write("\n".join(lines))
print("Done!")

