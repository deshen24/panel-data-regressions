import sys, os
import warnings 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet 
from regr import cvx_regr, pcr 
from var import var_est
from rank import svt, spectral_rank

warnings.filterwarnings("ignore")

if len(sys.argv)!=2:
	print('Usage: python {} <dataset name>'.format(sys.argv[0]))
	sys.exit()
data = str(sys.argv[1])
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
output_dir = 'output/case_study/{}'.format(data) 
os.makedirs(output_dir, exist_ok=True)

""" 
Data 
"""  
fname_pre = 'data/{}/pre_outcomes.csv'.format(data)
fname_post = 'data/{}/post_outcomes.csv'.format(data)
df_pre = pd.read_csv(fname_pre)
df_post = pd.read_csv(fname_post) 

# unit data
units = df_post.unit.unique() 
donors = units[units!=treated_unit]
N0 = len(donors)
N = N0 + 1 

# time data 
pre_cols = list(df_pre.drop(columns=['unit']).columns)
post_cols = list(df_post.drop(columns=['unit']).columns) 
T0 = len(pre_cols)
T1 = len(post_cols)
T = T0 + T1 
print("data configuration: ({}, {})".format(N0, T0))

""" 
Estimation 
""" 
directions = ['hz', 'vt', 'dr'] # sources of randomness 
sym_class = ['ols', 'pcr', 'ridge'] # symmetric class estimators 
asym_class = ['lasso', 'simplex', 'elastic net'] # asymmetric class estimators 
ci_algs = ['ols', 'pcr'] # symmetric estimators with valid confidence intervals
algs = sym_class + asym_class 

# parameters
z = 1.96 # 95% confidence interval parameter 
t = 0.999 # retain 99.9% of spectral energy  
fit_intercept = False 

# pre-computations  
alg_data_dict = {alg: {'Y0': np.zeros((N0, T0)), 
					   'Hu': np.zeros((N0, N0)),
					   'Hv': np.zeros((T0, T0)), 
					   'Hu_perp': np.zeros((N0, N0)),
					   'Hv_perp': np.zeros((T0, T0))} for alg in ci_algs}
y_n = df_pre.loc[df_pre['unit']==treated_unit].drop(columns=['unit']).values.flatten()
Y0 = df_pre.loc[df_pre['unit'].isin(donors)].drop(columns=['unit']).values 
for alg in ci_algs: 
	(u, s, v) = np.linalg.svd(Y0, full_matrices=False)
	k = spectral_rank(s, t=t) if alg=='pcr' else min(N0, T0)
	(Y0_alg, Hu_alg, Hv_alg, Hu_perp_alg, Hv_perp_alg) = svt(Y0, max_rank=k)
	alg_data_dict[alg]['Y0'] = Y0_alg 
	alg_data_dict[alg]['Hu'] = Hu_alg 
	alg_data_dict[alg]['Hv'] = Hv_alg 
	alg_data_dict[alg]['Hu_perp'] = Hu_perp_alg 
	alg_data_dict[alg]['Hv_perp'] = Hv_perp_alg 
	rank = int(np.round(np.trace(Y0_alg @ np.linalg.pinv(Y0_alg))))
	print("{} rank: {}".format(alg, rank))
print() 

# check if HRK estimator is valid 
var_algs = {alg: ['homoskedastic', 'jackknife'] for alg in ci_algs}
for alg in ci_algs: 
	u_max = np.max(1-np.diagonal(alg_data_dict[alg]['Hu_perp']))
	v_max = np.max(1-np.diagonal(alg_data_dict[alg]['Hv_perp']))
	hrk_count = 0 
	if u_max<1/2: 
		hrk_count += 1 
	if v_max<1/2: 
		hrk_count += 1 
	if hrk_count==2: 
		var_algs[alg].append('HRK')

# plot spectra of singular values 
(_, s, _) = np.linalg.svd(Y0, full_matrices=False)
fname = os.path.join(output_dir, 'spectra_{}'.format(data))
x_idxs = np.linspace(0, len(s)-1, num=len(s))
x_labels = [int(i) for i in np.linspace(1, len(s), num=len(s))]
if data=='germany':
    data_title = 'West Germany Reunification'
elif data=='basque':
    data_title = 'Terrorism in Basque Country'
else:
    data_title = 'CA Proposition 99'
(fig, axes) = plt.subplots(dpi=100)
plt.stem(s, markerfmt='+')
plt.title(data_title)
plt.ylabel('magnitude')
plt.xlabel('ordered singular values')
plt.xticks(x_idxs[::3], x_labels[::3])
plt.savefig(fname, dpi=100, bbox_inches="tight")
plt.close()

# initialize 
est_dict = {alg: {d: np.zeros(T1) for d in ['hz', 'vt']} for alg in algs} 
res_dict = {alg: {d: np.zeros(T) if d=='vt' else np.zeros((N, T1)) 
				  for d in ['hz', 'vt']}
			for alg in ci_algs}
se_dict = {alg: {v_alg: {d: np.zeros(T1) for d in directions} for v_alg in var_algs[alg]} for alg in ci_algs}

# iterate through post-treatment period 
print("Computing...")
for (t, post_t) in enumerate(post_cols): 

	# get post-treatment outcome data 
	y_t = df_post.loc[df_post['unit'].isin(donors), post_t].values.flatten()

	# ols 
	if 'ols' in algs: 
		regr = LinearRegression(fit_intercept=fit_intercept)

		# hz 
		w_hz = regr.fit(Y0, y_t).coef_ 
		est_dict['ols']['hz'][t] = np.dot(y_n, w_hz)

		# vt 
		w_vt = regr.fit(Y0.T, y_n).coef_
		est_dict['ols']['vt'][t] = np.dot(y_t, w_vt)
		
		# confidence intervals
		for v_alg in var_algs['ols']: 
			(var_hz, var_vt, trA) = var_est(y_n, y_t, alg_data_dict['ols']['Y0'], 
											w_hz, w_vt, 
											alg_data_dict['ols']['Hu_perp'], 
											alg_data_dict['ols']['Hv_perp'], 
										  	v_alg=v_alg)
			var_dr = max(0, var_hz + var_vt - trA)
			se_dict['ols'][v_alg]['hz'][t] = z * np.sqrt(var_hz)
			se_dict['ols'][v_alg]['vt'][t] = z * np.sqrt(var_vt) 
			se_dict['ols'][v_alg]['dr'][t] = z * np.sqrt(var_dr)

		# store in-sample residuals
		res_dict['ols']['hz'][:-1, t] = alg_data_dict['ols']['Hu_perp'] @ y_t
		res_dict['ols']['vt'][:T0] = alg_data_dict['ols']['Hv_perp'] @ y_n # this is repeated... 

	# pcr 
	if 'pcr' in algs: 
		# hz 
		w_hz = pcr(Y0, y_t, max_rank=k)
		est_dict['pcr']['hz'][t] = np.dot(y_n, w_hz)

		# vt 
		w_vt = pcr(Y0.T, y_n, max_rank=k) 
		est_dict['pcr']['vt'][t] = np.dot(y_t, w_vt)

		# confidence intervals 
		for v_alg in var_algs['pcr']: 
			(var_hz, var_vt, trA) = var_est(y_n, y_t, alg_data_dict['pcr']['Y0'], 
											w_hz, w_vt, 
											alg_data_dict['pcr']['Hu_perp'], 
											alg_data_dict['pcr']['Hv_perp'], 
										  	v_alg=v_alg)
			var_dr = max(0, var_hz + var_vt - trA)
			se_dict['pcr'][v_alg]['hz'][t] = z * np.sqrt(var_hz)
			se_dict['pcr'][v_alg]['vt'][t] = z * np.sqrt(var_vt) 
			se_dict['pcr'][v_alg]['dr'][t]= z * np.sqrt(var_dr)

		# store in-sample residuals
		res_dict['pcr']['hz'][:-1, t] = alg_data_dict['pcr']['Hu_perp'] @ y_t
		res_dict['pcr']['vt'][:T0] = alg_data_dict['pcr']['Hv_perp'] @ y_n # this is repeated...

	# ridge
	if 'ridge' in algs: 
		regr = Ridge(fit_intercept=fit_intercept)

		# hz 
		w_hz = regr.fit(Y0, y_t).coef_ 
		est_dict['ridge']['hz'][t] = np.dot(y_n, w_hz)

		# vt 
		w_vt = regr.fit(Y0.T, y_n).coef_
		est_dict['ridge']['vt'][t] = np.dot(y_t, w_vt)

	# lasso 
	if 'lasso' in algs: 
		regr = Lasso(fit_intercept=fit_intercept)

		# hz 
		w_hz = regr.fit(Y0, y_t).coef_ 
		est_dict['lasso']['hz'][t] = np.dot(y_n, w_hz)

		# vt 
		w_vt = regr.fit(Y0.T, y_n).coef_
		est_dict['lasso']['vt'][t] = np.dot(y_t, w_vt)

	# elastic net 
	if 'elastic net' in algs:
		regr = ElasticNet(fit_intercept=fit_intercept)

		# hz 
		w_hz = regr.fit(Y0, y_t).coef_ 
		est_dict['elastic net']['hz'][t] = np.dot(y_n, w_hz)

		# vt 
		w_vt = regr.fit(Y0.T, y_n).coef_
		est_dict['elastic net']['vt'][t] = np.dot(y_t, w_vt)

	# simplex 
	if 'simplex' in algs: 
		# hz 
		w_hz = cvx_regr(Y0, y_t) 
		est_dict['simplex']['hz'][t] = np.dot(y_n, w_hz)

		# vt 
		w_vt = cvx_regr(Y0.T, y_n) 
		est_dict['simplex']['vt'][t] = np.dot(y_t, w_vt)

print("Computing completed!")
print()

""" 
PARAMETERS FOR PLOTTING
""" 
x_idxs = np.linspace(0, T-1, num=T)
x_labels = pre_cols + post_cols
y_pre = df_pre.loc[df_pre['unit']==treated_unit, pre_cols].values.flatten() 
y_post = df_post.loc[df_post['unit']==treated_unit, post_cols].values.flatten() 
y_obs = np.concatenate([y_pre, y_post])

# colors 
colors = {
    'ols': 'royalblue',
    'pcr': 'yellowgreen',
    'ridge': 'darkorange', 
    'lasso': 'red',
    'elastic net': 'gold',
    'simplex': 'violet'
}
palette = sns.color_palette("Pastel1", 10).as_hex()
box_colors = {
	'ols': palette[1], 
	'pcr': palette[2]
}

if data=='germany':
    data_title = 'West Germany Reunification'
elif data=='basque':
    data_title = 'Terrorism in Basque Country'
else:
    data_title = 'CA Proposition 99'

""" 
PLOT ESTIMATION RESULTS 
"""  
print("Plotting estimation results...")
fname = os.path.join(output_dir, "symmetric")
(fig, ax) = plt.subplots(dpi=100)
for alg in sym_class: 
	y_hz = est_dict[alg]['hz']
	y_vt = est_dict[alg]['vt']
	plt.plot(np.concatenate([y_pre, y_hz]), color=colors[alg], label='{}'.format(alg))
	plt.plot(np.concatenate([y_pre, y_vt]), color=colors[alg], linestyle='-.')
plt.plot(y_obs, color='black', lw=2.5, label='observed')
plt.axvline(x=T0, linestyle=':', color='gray', label=iv)
plt.xticks(x_idxs[::interval], x_labels[::interval])
plt.legend(loc='best')
plt.xlabel('year')
plt.ylabel(outcome_var)
plt.title(treated_unit)
plt.ylim([ymin, ymax])
plt.savefig(fname, dpi=100, bbox_inches="tight")
plt.close()

# asymmetric estimators  
fname = os.path.join(output_dir, "asymmetric")
(fig, ax) = plt.subplots(dpi=100) 
for alg in asym_class: 
	y_hz = est_dict[alg]['hz']
	y_vt = est_dict[alg]['vt']
	plt.plot(np.concatenate([y_pre, y_hz]), color=colors[alg], label='hz-{}'.format(alg))
	plt.plot(np.concatenate([y_pre, y_vt]), color=colors[alg], linestyle='-.', label='vt-{}'.format(alg))
plt.plot(y_obs, color='black', lw=2.5, label='observed')
plt.axvline(x=T0, linestyle=':', color='gray', label=iv)
plt.xticks(x_idxs[::interval], x_labels[::interval])
plt.legend(loc='best')
plt.xlabel('year')
plt.ylabel(outcome_var)
plt.title(treated_unit)
plt.ylim([ymin, ymax])
plt.savefig(fname, dpi=100, bbox_inches="tight")
plt.close()

""" 
PLOT RESIDUALS
"""
# store out-of-sample residuals
for alg in ci_algs: 
	y_alg = est_dict[alg]['hz']
	err = y_post - y_alg 
	res_dict[alg]['hz'][-1, :] = err 
	res_dict[alg]['vt'][T0:] = err

# find largest error value for plotting
y_lim = 0  
for alg in ci_algs: 
	y_lim_hz = np.abs(res_dict[alg]['hz'][-1]).max() 
	y_lim_vt = np.abs(res_dict[alg]['vt'][T0:]).max() 
	y_lim_alg = np.max([y_lim_hz, y_lim_vt])
	y_lim = y_lim_alg if y_lim_alg>y_lim else y_lim 
y_lim = int(np.round(y_lim)) * 1.01

# VT residuals
fname = os.path.join(output_dir, "vt_residuals_in_out")
res_dict[alg]['vt'][np.abs(res_dict[alg]['vt'])<=1e-10] = 0
(fig, ax) = plt.subplots(dpi=100)
plt.axvline(x=T0, linestyle=':', color='gray', label=iv)
plt.axhline(y=0, color='black')
for alg in ci_algs: 
	plt.plot(res_dict[alg]['vt'], '*', color=colors[alg], label='vt-{}'.format(alg))
plt.xticks(x_idxs[::3], x_labels[::3])
plt.xticks(rotation=90)
plt.legend(loc='best')
plt.xlabel('year')
plt.ylabel('gap in {} in {}'.format(outcome_var, treated_unit))
plt.title(data_title)
plt.ylim([-y_lim, y_lim])
bbox_props = dict(boxstyle="larrow", fc=box_colors[alg], ec=colors[alg], lw=1)
t = ax.text(T0-1, y_lim//2, "in-sample errors", ha="right", va="center", bbox=bbox_props)
n_xticks = len(ax.get_xticklabels())
x_idx = int(np.round(T0/T * n_xticks))
for j in range(x_idx, n_xticks): 
	plt.gca().get_xticklabels()[j].set_color("red")
plt.savefig(fname, dpi=100, bbox_inches="tight")
plt.close()

# HZ residuals
units = list(donors) + [treated_unit]
x_unit_idxs = np.linspace(0, N-1, num=N)
x_unit_labels = [unit[:4] + '.' for unit in units]
df_res = pd.DataFrame(columns=['alg', 'unit', 'error'])

fname = os.path.join(output_dir, "hz_residuals_in_out") 
(fig, ax) = plt.subplots(dpi=100)
plt.axhline(y=0, color='black')
for alg in ci_algs:
	hz_res = res_dict[alg]['hz'] 
	for (i, unit) in enumerate(units):  
		series = pd.DataFrame({'alg': ['hz-{}'.format(alg)] * T1,
							   'unit': [unit]*T1,
							   'error': hz_res[i]})
		df_res = pd.concat([df_res, series])

# swarmplot from dataframe of residuals
for alg in ci_algs:
	ax = sns.swarmplot(x='unit', y='error', data=df_res.loc[df_res['alg']=='hz-{}'.format(alg)], 
					   marker='*', hue='alg', palette=[colors[alg]])
plt.legend(loc='best')
plt.xticks(rotation=90)
plt.xticks(x_unit_idxs, x_unit_labels)
plt.gca().get_xticklabels()[-1].set_color("red")
plt.ylabel('gap in {} from {}-{}'.format(outcome_var, post_cols[0], post_cols[-1]))
plt.xlabel('units')
plt.title(data_title)
plt.axvline(x=N-1.5, linestyle=':', color='gray')
ax.set(ylim=(-y_lim, y_lim))
bbox_props = dict(boxstyle="larrow", fc=box_colors[alg], ec=colors[alg], lw=1)
t = ax.text(N-2, y_lim//2, "in-sample errors", ha="right", va="center", bbox=bbox_props)
plt.savefig(fname, dpi=100, bbox_inches="tight")
plt.close() 

""" 
PLOT CONFIDENCE INTERVALS
"""  
print("Plotting confidence intervals...")
for alg in ci_algs:
	y_alg = est_dict[alg]['hz']

	# iterate through valid variance estimators 
	for v_alg in var_algs[alg]: 

		# iterate through directions
		for d in directions: 
			fname = os.path.join(output_dir, "{}_{}_{}".format(alg, d, v_alg))
			(fig, ax) = plt.subplots(dpi=100)
			plt.plot(np.concatenate([y_pre, y_alg]), color=colors[alg], label=alg)
			se = se_dict[alg][v_alg][d]
			plt.fill_between(x_idxs[T0:], y_alg-se, y_alg+se, 
							 alpha=0.4, color=colors[alg], label='{}-{}'.format(d, v_alg))
			plt.plot(y_obs, color='black', lw=2.5, label='observed')
			plt.axvline(x=T0, linestyle=':', color='gray', label=iv)
			plt.xticks(x_idxs[::interval], x_labels[::interval])
			plt.legend(loc='best')
			plt.xlabel('year')
			plt.ylabel(outcome_var)
			plt.title(treated_unit)
			plt.ylim([ymin, ymax])
			plt.savefig(fname, dpi=100, bbox_inches="tight")
			plt.close()
print("Done!")
