#!/usr/bin/env ipython
# aggregates accross all subjects to analyse erp P300 hopefully
import glob
import matplotlib as plt
import mne
import numpy as np
from scipy import stats

plt.use('Qt5Agg')

# timing of P3 window
TMIN=0.3
TMAX=.5
CHANNELS=['CP3','CP1','CPz','CP2','CP4','P3','P1','Pz','P2','P4','PO3','POz','PO4']
CHANNELS=['CP1','CPz','CP2','P1','Pz','P2']
#CHANNELS=['Pz']

part_files = glob.glob("out_evoked/evoked*ave.fif")
tvns_evokeds = []
sham_evokeds = []
tvns_means= []
sham_means= []
for file in part_files:
    print(f'LOADING {file}')
    evoked = mne.read_evokeds(file)
    print(f'datalen: {len(evoked[0]._data[0])} size: {evoked[0]._size}')

    tvns_evokeds.append(evoked[0].pick_channels(CHANNELS).detrend(1))
    sham_evokeds.append(evoked[1].pick_channels(CHANNELS).detrend(1))
    tvns_means.append(evoked[0].copy().crop(TMIN,TMAX).data.mean())
    sham_means.append(evoked[1].copy().crop(TMIN,TMAX).data.mean())


tvns_grand=mne.grand_average(tvns_evokeds[1:])
sham_grand=mne.grand_average(sham_evokeds[1:])

mne.viz.plot_compare_evokeds([tvns_grand, sham_grand], picks='eeg', combine='mean',show_sensors=True)
#PO sites only
t_test = stats.ttest_rel(tvns_means,sham_means,alternative='greater')
print("dependent t-test, tvns > sham:", t_test)
