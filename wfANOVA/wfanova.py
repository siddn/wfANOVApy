import numpy as np
import scipy as sp
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


sns.set_palette('flare') # change palette to 'crest' for a cool version

data = sp.io.loadmat("./data/wfANOVAdata.mat")
time = data['time'].flatten()
acceleration = data['A']-2
velocity = data['V']-1
subject = data['S']
factors = np.hstack([velocity, acceleration, subject])
ta_amplitude = data['TA'].T
velocity_labels = [25, 30, 35, 40]
acceleration_labels = [0.2, 0.3, 0.4]

def wavelet_decomposition(signals, level=4):
    if signals.ndim==1:
        n_trials = 1
        signals = signals.reshape((-1,1))
    else: n_trials = signals.shape[1]
    results = []
    for i in range(n_trials):
        coeffs = pywt.wavedec(data=signals[:,i], wavelet='coif3', mode='per', level=level)
        results.append(coeffs)
    if len(results) == 1:
        return results[0]
    return results

def wavelet_reconstruction(coefficients):
    reconstructed = pywt.waverec(coeffs=coefficients, wavelet='coif3' ,mode='per')
    return reconstructed

def plot_grid():
    subjects = np.unique(subject)
    accelerations = np.unique(acceleration)
    velocities = np.unique(velocity)
    fig, axs = plt.subplots(4,3, sharex=True, sharey=True)
    fig.tight_layout()
    for i, vel in enumerate(np.flip(velocities)):
        for j, acc in enumerate(accelerations):
            current_curves = ta_amplitude[:,(velocity==vel).flatten() & (acceleration==acc).flatten()]
            axs[i,j].autoscale(enable=True, tight=True)
            axs[i,j].plot(time, current_curves, 'lightgrey', linewidth=0.3)
            axs[i,j].set_title(f'{velocity_labels[vel]}cm/s, {acceleration_labels[acc]}g', fontsize=9)
            axs[i,j].plot(time, np.mean(current_curves, axis=1), 'k', linewidth=0.5)
            axs[i,j].set_xlim([0,1])
            for k, subj in enumerate(subjects):
                subject_specific_curves = ta_amplitude[:,(velocity==vel).flatten() & (acceleration==acc).flatten() & (subject==subj).flatten()]
                subject_mean = np.mean(subject_specific_curves, axis=1)
                axs[i,j].plot(time, subject_mean, linewidth=0.3)
    sns.despine()

def plot_contrasts():
    subjects = np.unique(subject)
    accelerations = np.unique(acceleration)
    velocities = np.unique(velocity)
    fig, axs = plt.subplots(1,2, figsize=(4,1))
    fig.tight_layout()

    # Mean Difference
    baseline_velocity = np.mean(ta_amplitude[:,(velocity==0).flatten()], axis=1) 
    baseline_acceleration = np.mean(ta_amplitude[:,(acceleration==0).flatten()], axis=1)
    # Acceleration Contrasts
    for i, acc in enumerate(accelerations):
        if i == 0:
            continue
        current_mean_curve = np.mean(ta_amplitude[:,(acceleration==acc).flatten()], axis=1)
        mean_difference = current_mean_curve - baseline_acceleration



        axs[i-1].plot(time, mean_difference, 'k', linewidth=0.5)
        axs[i-1].set_title(f'{acceleration_labels[acc]}g vs {acceleration_labels[0]}g', fontsize=9)
        axs[i-1].set_xlim([0,1])

    sns.despine()

    fig, axs = plt.subplots(3,1, figsize=(2,3))
    fig.tight_layout()
    # Velocity Contrasts
    for i, vel in enumerate(np.flip(velocities)):
        if i == 3:
            continue
        current_mean_curve = np.mean(ta_amplitude[:,(velocity==vel).flatten()], axis=1)
        mean_difference = current_mean_curve - baseline_velocity
        axs[i].plot(time, mean_difference, 'k', linewidth=0.5)
        axs[i].set_title(f'{velocity_labels[vel]}cm/s vs {velocity_labels[0]}cm/s', fontsize=9)
        axs[i-1].set_xlim([0,1])

    sns.despine()
            
def wfANOVAtest(data, factors, posthoc_factors, factor_labels=None, alpha=0.05):
    indexing_array = [len(component) for component in data[0]]
    wavedata = np.array([np.concatenate(wavedatapt) for wavedatapt in data]).T
    n_datapoints = wavedata.shape[0]
    n_repetitions = wavedata.shape[1]
    full_results = []
    significant = []
    sources = None
    # Initial N-Way ANOVA test
    for i in range(1):
        current_datapt = wavedata[i,:]
        currentdf = pd.DataFrame(np.row_stack([current_datapt, factors.T]).T, columns=['coef', 'vel', 'acc', 'subj'])
        model = ols('coef ~ C(vel) + C(acc) + C(subj)',data=currentdf).fit()
        anova_results = sm.stats.anova_lm(model, typ=3).round(3)
        anova_res_2 = pg.anova(data=currentdf, dv='coef', between=['vel', 'acc', 'subj']).round(3)
        print(anova_res_2)
        print(anova_results)
        pvals = anova_results['PR(>F)'].to_list()[1:-1]
        significant.append([True if pval<=alpha else False for pval in pvals])
        full_results.append(pvals)

    all_pvals = np.array(full_results)

    print(f'{all_pvals.shape=}')
    posthoc_results = pg.pairwise_tukey(data=currentdf, dv='coef', between='subj').round(3)
    print(posthoc_results)
    # for i in range(len(posthoc_factors)):
    #     waveletcoeffs = np.zeros(n_datapoints, 1)
    #     if posthoc_factors[i] is True:
    #         current_pvals = all_pvals[:,i]
    #         for pval in current_pvals:
    #             if pval <= alpha:
    #                 posthoc_results = pg.pairwise_tukey(data=)
    # #                 posthoce_results = pairwise_tukeyhsd


            
    
    return full_results, significant, sources


test_data = ta_amplitude
wavedata = wavelet_decomposition(test_data)

results, significances, sources = wfANOVAtest(wavedata, factors, [True, True, False])
print(results[0])
print(significances[0])
print(factors[0,:])
print(sources)



# fig = plt.figure(figsize=(3,1))
# markerline, stemlines, baseline = plt.stem(np.concatenate(results))
# plt.setp(stemlines, linewidth=0.2, color='k', alpha=1, linestyle='-')
# plt.setp(markerline, color='k', fillstyle='none', markersize=4)
# plt.setp(baseline, color='k', alpha=0.4)
# plt.show()
