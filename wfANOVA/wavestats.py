import numpy as np
import scipy as sp
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import _wavetools

sns.set_palette('flare') # change palette to 'crest' for a cool version

def wttest(x:np.ndarray, y:np.ndarray, alpha=0.05, draw=False, correction=None):
    """Performs pointwise two sample t-test on wavelet coefficients of two time series signals.
    corrects the alpha value for multiple comparisons.

    Args:
        data_a (np.ndarray): _description_
        data_b (np.ndarray): _description_
        alpha (float, optional): _description_. Defaults to 0.05.
    """
    signal_length_x = x.shape[0]
    signal_length_y = y.shape[0]

    assert signal_length_x==signal_length_y, "Length of the input data (time series) must be the same"

    signal_length = signal_length_x

    wavelets_x = _wavetools.wavelet_decomposition(x, draw=draw)
    wavelets_y = _wavetools.wavelet_decomposition(y, draw=draw)

    _, coef_slices, coef_shapes = pywt.ravel_coeffs(wavelets_x[0])

    wavelets_x = np.vstack([np.concatenate(coeffs) for coeffs in wavelets_x]).T
    wavelets_y = np.vstack([np.concatenate(coeffs) for coeffs in wavelets_y]).T

    print(wavelets_x.shape, wavelets_y.shape)


    mean_x = wavelets_x.mean(axis=1)
    mean_y = wavelets_y.mean(axis=1)

    print(mean_x.shape, mean_y.shape)
    # Pass 1
    pvals = np.array([pg.ttest(wavelets_x[i,:], wavelets_y[i,:], correction='auto')['p-val'].values[0] for i in range(signal_length)])
    num_significant_points = np.sum(pvals < alpha)

    # Pass 2
    if num_significant_points > 0:
        adjusted_alpha = alpha / num_significant_points
    else:
        adjusted_alpha = alpha
    
    if correction is None: adjusted_alpha = alpha

    significance = pvals < adjusted_alpha
    significant_difference = (mean_y - mean_x)*significance
    restructured_mean_coeffs = pywt.unravel_coeffs(significant_difference, coef_slices, coef_shapes, output_format='wavedec')
    significant_waveform = _wavetools.wavelet_reconstruction(restructured_mean_coeffs)

    if draw:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        _wavetools.plot_wavelet_coeffs(np.concatenate(restructured_mean_coeffs), color='r', ax=ax)
        ax.set_title("Wavelet Coefficients")

    return significant_waveform



if __name__ == "__main__":
    data = sp.io.loadmat("./data/wfANOVAdata.mat")
    emg = data['TA'].T # Data should have the shape (time, trials)
    acceleration = data['A'].flatten()-2
    velocity = data['V'].flatten()-1
    subject = data['S'].flatten()
    time = data['time'].flatten()


    # Split data by velocity
    velocity1_data = emg[:,(velocity==0)]
    velocity2_data = emg[:,(velocity==1)]
    velocity3_data = emg[:,(velocity==2)]
    velocity4_data = emg[:,(velocity==3)]

    # print(velocity1_data.shape)
    # print(velocity2_data.shape)
    result1 = wttest(velocity1_data, velocity2_data, alpha=0.01, draw=False, correction='auto')
    result2 = wttest(velocity1_data, velocity3_data, alpha=0.01, draw=False, correction='auto')
    plt.plot(time, velocity1_data.mean(axis=1))
    plt.plot(time, velocity2_data.mean(axis=1))
    plt.plot(time, result1)
    plt.xlim([0,1])
    plt.ylim([-.5,.5])
    plt.figure()
    plt.plot(time, result2)
    plt.xlim([0,1])
    plt.ylim([-.5,.5])
    plt.show()