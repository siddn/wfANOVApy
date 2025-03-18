import numpy as np
import scipy as sp
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pywt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import tukey_hsd
# import scikit_posthocs as sp


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

def wfANOVAoneway(data, factors, alpha=0.05, orientation='v', use_posthoc_alpha=True):
    wavedata = np.array([pywt.coeffs_to_array(data[i])[0] for i in range(len(data))]).T
    coeff_slices = [pywt.coeffs_to_array(data[i])[1] for i in range(len(data))]
    n_datapoints = wavedata.shape[0]
    all_pvals = []
    significant = []
    for i in range(n_datapoints):
        current_datapt = wavedata[i,:]
        currentdf = pd.DataFrame(np.row_stack([current_datapt, factors.T]).T)
        anova_res = pg.anova(data=currentdf, dv=0, between=1, detailed=True)
        pvals = anova_res['p-unc'][0]
        all_pvals.append(pvals)
        if pvals < alpha: significant.append(True)
        else: significant.append(False)

    all_pvals = np.array(all_pvals)

    n_contrasts = anova_res['DF'][0]
    wave_contrasts = np.zeros((n_datapoints, anova_res['DF'][0]))
    if use_posthoc_alpha:posthoc_alpha = alpha/np.sum(all_pvals < alpha) # Why do this posthoc_alpha thing?
    else: posthoc_alpha = alpha
    for i in range(n_datapoints):
        if significant[i] is True:
            current_datapt = wavedata[i,:]
            currentdf = pd.DataFrame(np.row_stack([current_datapt, factors.T]).T)
            tukey_posthoc_res = pg.pairwise_tukey(data=currentdf, dv=0, between=1)
            posthoc_res = pg.pairwise_tests(data=currentdf, dv=0, between=1, padjust='bonf', return_desc=True).round(5)
            for j in range(n_contrasts):
                if posthoc_res['p-unc'][j] < posthoc_alpha:
                    wave_contrasts[i,j] = -tukey_posthoc_res['diff'][j]
                    # wave_contrasts[i,j] = -posthoc_res['mean(B)'][j] + posthoc_res['mean(A)'][j]

    print(posthoc_res)
    print(currentdf)

    contrasts = np.zeros((n_datapoints, anova_res['DF'][0]))
    for i in range(n_contrasts):
        contrasts[:,i] = wavelet_reconstruction(pywt.array_to_coeffs(wave_contrasts[:,i], coeff_slices=coeff_slices[0], output_format='wavedec'))

    if orientation == 'h':  fig, axs = plt.subplots(1, n_contrasts, figsize=(5,1))
    else: fig, axs = plt.subplots(n_contrasts,1, figsize=(1,3))
    for i in range(n_contrasts):
        if orientation == 'h': ax = axs[i]
        else: ax = axs[n_contrasts - i - 1]
        ax.plot(time, contrasts[:,i], 'r', linewidth=0.3)
        ax.set_xlim([0,1])
        ax.set_ylim([-0.5,0.5])
    
def wfANOVAtest(data, factors, posthoc_factors, factor_labels=None, alpha=0.05):
    wavedata = np.array([pywt.coeffs_to_array(data[i])[0] for i in range(len(data))]).T
    coeff_slices = [pywt.coeffs_to_array(data[i])[1] for i in range(len(data))]
    n_datapoints = wavedata.shape[0]
    full_results = []
    # Initial N-Way ANOVA test
    for i in range(n_datapoints):
        current_datapt = wavedata[i,:]
        currentdf = pd.DataFrame(np.row_stack([current_datapt, factors.T]).T, columns=['coef', 'vel', 'acc', 'subj'])
        model = ols('coef ~ C(vel) + C(acc) + C(subj)',data=currentdf,).fit()
        anova_results = sm.stats.anova_lm(model, typ=3).round(3)
        anova_res_2 = pg.anova(data=currentdf, dv='coef', between=['vel', 'acc', 'subj'], detailed=True, ss_type=3).round(6)
        print(anova_res_2)
        print(anova_results)
        pvals = anova_results['PR(>F)'].to_list()[1:-1]
        full_results.append(pvals)

    all_pvals = np.array(full_results)
    print(f'{all_pvals.shape=}')


    vel_pvals = all_pvals[:,0]
    posthoc_alpha = alpha/np.sum(vel_pvals<alpha)
    contrast1 = []
    contrast2 = []
    contrast3 = []
    for i in range(n_datapoints):
        current_datapt = wavedata[i,:]
        currentdf = pd.DataFrame(np.row_stack([current_datapt, factors.T]).T, columns=['coef', 'vel', 'acc', 'subj'])

        if  vel_pvals[i] < alpha:
            # posthoc_results = pg.pairwise_tests(data=currentdf, dv='coef', between='vel', padjust='bonf', return_desc=True).round(5)
            posthoc_results = pg.pairwise_tukey(data=currentdf, dv='coef', between='vel').round(5)
            # posthoc_2 = tukey_hsd(currentdf[currentdf['vel']==0]['coef'], currentdf[currentdf['vel']==1]['coef'], currentdf[currentdf['vel']==2]['coef'], currentdf[currentdf['vel']==3]['coef'])
            if posthoc_results['p-tukey'][0] < posthoc_alpha:
                # contrast1.append(posthoc_results['mean(B)'][0]-posthoc_results['mean(A)'][0])
                contrast1.append(-posthoc_results['diff'][0])
            else:
                contrast1.append(0)

            if posthoc_results['p-tukey'][1] < posthoc_alpha:
                # contrast2.append(posthoc_results['mean(B)'][1]-posthoc_results['mean(A)'][1])
                contrast2.append(-posthoc_results['diff'][1])
            else:
                contrast2.append(0)

            if posthoc_results['p-tukey'][2] < posthoc_alpha:
                # contrast3.append(posthoc_results['mean(B)'][2]-posthoc_results['mean(A)'][2])
                contrast3.append(-posthoc_results['diff'][2])
            else:
                contrast3.append(0)
        else:
            contrast1.append(0)
            contrast2.append(0)
            contrast3.append(0)

    # print(posthoc_2)
    # print(posthoc_2.confidence_interval())
    print(np.nonzero(np.array(contrast1)))
    # Realign Wavelet Coefficients
    new_wavelet_coef1 = pywt.array_to_coeffs(contrast1, coeff_slices=coeff_slices[0], output_format='wavedec')
    new_wavelet_coef2 = pywt.array_to_coeffs(contrast2, coeff_slices=coeff_slices[0], output_format='wavedec')
    new_wavelet_coef3 = pywt.array_to_coeffs(contrast3, coeff_slices=coeff_slices[0], output_format='wavedec')

    reconstructed1 = wavelet_reconstruction(coefficients=new_wavelet_coef1)
    reconstructed2 = wavelet_reconstruction(coefficients=new_wavelet_coef2)
    reconstructed3 = wavelet_reconstruction(coefficients=new_wavelet_coef3)

    fig, axs = plt.subplots(3,1, figsize=(5,2))
    axs[0].plot(time, reconstructed1)
    axs[1].plot(time, reconstructed2)
    axs[2].plot(time, reconstructed3)

    axs[0].set_xlim([0,1])
    axs[1].set_xlim([0,1])
    axs[2].set_xlim([0,1])

    return full_results

# plot_contrasts()
plot_grid()
test_data = ta_amplitude
wavedata = wavelet_decomposition(test_data)

# # results = wfANOVAtest(wavedata, factors, [True, True, False])

wfANOVAoneway(wavedata, velocity, orientation='v', use_posthoc_alpha=True)
wfANOVAoneway(wavedata, acceleration, orientation='h', use_posthoc_alpha=True)
plt.show()


# fig = plt.figure(figsize=(3,1))
# markerline, stemlines, baseline = plt.stem(np.concatenate(results))
# plt.setp(stemlines, linewidth=0.2, color='k', alpha=1, linestyle='-')
# plt.setp(markerline, color='k', fillstyle='none', markersize=4)
# plt.setp(baseline, color='k', alpha=0.4)
# plt.show()
