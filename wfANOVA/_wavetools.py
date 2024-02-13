import numpy as np
import scipy as sp
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import pywt

def plot_wavelet_coeffs(coeffs, ax:plt.Axes=None, color='k'):
    """_summary_

    Args:
        coeffs (list): a list of coefficients, list willl be flattened if not one dimensional
        ax (plt.Axes, optional): axes to plot on. Defaults to None.
    """
    markerline, stemlines, baseline = ax.stem(coeffs)
    plt.setp(stemlines, linewidth=0.2, color=color, alpha=1, linestyle='-')
    plt.setp(markerline, color=color, fillstyle='none', markersize=4)
    plt.setp(baseline, color=color, alpha=0.4)


def wavelet_decomposition(signals, level=4, draw=False, as_mean=True) -> list:
    """Get the wavelet decomposition of a signal.

    Args:
        signals (ndarray): the signals to decompose in shape (time, trials)
        level (int, optional): wavelet decomposition level. Defaults to 4.
        draw (bool, optional): show a stem plot of the wavelet coefficients. Still need to invoke plt.plot() . Defaults to False.

    Returns:
        list: list of coefficients, with a list of list for each trial.
    """

    # Shape the data correctly
    if signals.ndim==1:
        n_trials = 1
        signals = signals.reshape((-1,1))
    else: n_trials = signals.shape[1]

    # Extract the coefficients for each input trial
    decomp = []
    for i in range(n_trials):
        coeffs = pywt.wavedec(data=signals[:,i], wavelet='coif3', mode='per', level=level)
        decomp.append(coeffs)

    # Optional Plotting
    if draw is True:
        fig = plt.figure()
        if as_mean is True:
            mean_coefficients = np.vstack([np.concatenate(coeffs) for coeffs in decomp]).mean(axis=0)
            data_to_plot = [mean_coefficients]
            trials_to_plot = 1
            plot_title = "Mean Wavelet Coefficients"
        else:
            trials_to_plot = n_trials
            data_to_plot = [np.concatenate(coeffs) for coeffs in decomp]
            plot_title = "Wavelet Coefficients"

        plt.title(plot_title)
        for i in range(trials_to_plot):
            ax = fig.add_subplot(trials_to_plot, 1, i+1)
            plot_wavelet_coeffs(data_to_plot[i], ax=ax)
    
    if len(decomp) == 1: decomp = decomp[0]
    return decomp

def wavelet_reconstruction(coefficients: list, draw=False) -> list:
    """

    Args:
        coefficients (list): list of coefficients, with a list of list for each trial.

    Returns:
        list: list of reconstructed signals, with a list of list for each trial. 
    """
    reconstructed = pywt.waverec(coeffs=coefficients, wavelet='coif3' ,mode='per')
    if draw:
        fig = plt.figure()
        plt.plot(reconstructed)

    return reconstructed


if __name__ == "__main__":

    data = sp.io.loadmat("./data/wfANOVAdata.mat")
    emg = data['TA'].T # Data should have the shape (time, trials)
    acceleration = data['A'].flatten()-2
    velocity = data['V'].flatten()-1
    subject = data['S'].flatten()
    time = data['time'].flatten()


    # Get velocity data for a single subject
    velocity1_data = emg[:,(velocity==0)&(subject==1)]
    velocity2_data = emg[:,(velocity==1)&(subject==1)]
    velocity3_data = emg[:,(velocity==2)&(subject==1)]

    decomp = wavelet_decomposition(velocity1_data, draw=False, as_mean=False)
    trial1 = decomp[0]
    print(type(trial1))
    print(len(trial1))
    recomp = wavelet_reconstruction(trial1, draw=True)

    plt.show()

