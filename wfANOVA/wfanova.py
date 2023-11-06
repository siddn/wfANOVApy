import numpy as np
import scipy as sp
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt

data = sp.io.loadmat("./data/wfANOVAdata.mat")
print(data.keys())
time = data['time']
acceleration = data['A']
velocity = data['V']
subject = data['S']
ta_amplitude = data['TA']

print(type(time))
print(type(acceleration))
print(type(velocity))
print(type(subject))
print(type(ta_amplitude))

print(subject[subject==7].shape)
print((subject==7).flatten().shape)
# current_curve = ta_amplitude[(velocity==1).flatten() & (acceleration==2).flatten(),:]
# plt.plot(current_curve.T)
# plt.show()

def plot_grid():
    subjects = np.unique(subject)
    accelerations = np.unique(acceleration)
    velocities = np.unique(velocity)
    fig, axs = plt.subplots(4,3)
    print(f'{velocities=}')
    print(f'{accelerations=}')
    print(f'{subjects=}')
    for i, vel in enumerate(np.flip(velocities)):
        for j, acc in enumerate(accelerations):
            current_curves = ta_amplitude[(velocity==vel).flatten() & (acceleration==acc).flatten(),:].T
            axs[i,j].plot(current_curves, 'grey', alpha=0.5)
            axs[i,j].title.set_text(f'V={vel}, A={acc}')
    plt.show()

plot_grid()

