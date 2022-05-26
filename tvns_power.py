#!/usr/bin/env ipython
# see https://mne.discourse.group/t/eeg-processing-pipeline-with-autoreject/3443
# https://autoreject.github.io/stable/auto_examples/plot_auto_repair.html

import matplotlib as plt
import mne
import numpy as np
import os
import sys
import getopt
import glob
from yasa import bandpower
from scipy import stats

from clean_events import Participant,_CHECKED

SUBJ=1
CHARTS=False
T_MIN = -3
T_MAX = 1
T_INT = .05
L_PASS = 15
H_PASS = .5
TOPOMAPS = False
DETREND = 1 # linear detrend
# used for charting, evoked contains all channels
CHANNELS = ['CP1', 'CPz', 'CP2', 'P1', 'Pz', 'P2']

# For interactive plotting, load the following backend:
plt.use('Qt5Agg')

def format_fig(figure, window_tit='', canvas_tit=''):
    if window_tit:
        figure.canvas.manager.set_window_title(window_tit)
    if canvas_tit:
        figure.suptitle(canvas_tit)
    return figure

if __name__ == '__main__':
    # get options, default is no charts -p6 -c also switches on charts
    argv = sys.argv[1:]
    ALL=False
    help_mess="evoke_tvns -- -p <participant number> -c [--charts=n] [--tmin=0] [--tmax=1] [--tint=.05] [--detrend=none/0/1"
    p_charts = False
    if (len(argv)):
        CHARTS = False
        try:
            opts, args = getopt.getopt(argv, "p:cf:", ["charts=","all",'tmin=','tmax=','tint=','pcharts','detrend='])
        except:
            print("invalid command line arguments:")
            print(help_mess)
            sys.exit()
        for opt, arg in opts:
            if opt == '-h':
                print(help_mess)
            elif opt == '-f':
                L_PASS = int(arg)
            elif opt == '-p':
                SUBJ = int(arg)
            elif opt == '--tint':
                T_INT = float(arg)
            elif opt == '--tmin':
                T_MIN = float(arg)
            elif opt == '--tmax':
                T_MAX = float(arg)
            elif opt == '--pcharts':
                p_charts = True
            elif opt in ["-c", "--charts"]:
                if arg == "n":
                    CHARTS = False
                else:
                    CHARTS = True
            elif opt == '--topomaps':
                TOPOMAPS = True
            elif opt == '--all':
                ALL = True
    if ALL:
        participants = _CHECKED
    else:
        participants = [ SUBJ ]

    sham_power = []
    tvns_power = []
    for subj in participants:
        if subj in participants:
            participant = Participant(subj)
            part_file = f"out_epochs/cleaned_stimoff_epoch_sub-{participant.part_str}-epo.fif"
            print(f"P{subj} loading {part_file}")
            epochs = mne.read_epochs(part_file)
            epochs.crop(T_MIN,T_MAX)
            # ignore the stim channel so ch_names matches data
            epochs.pick('eeg')
            if CHARTS:
                freqs = np.logspace(*np.log10([2,30]), num=20)
                n_cycles = freqs / 2
                #inspect power
                power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                               return_itc=True, decim=3, n_jobs=1)
                power.crop(-.1,.7)
                itc.crop(-.1,.7)
                baseline_mode = 'logratio'
                baseline = (None, 0)

                (power.copy()
                 .pick_types(eeg=True, meg=False)
                 .plot_topo())
                # single channel
                power.plot(picks='Pz',baseline=baseline)
                # frequency ranges
                fig, axis = plt.pyplot.subplots(1, 4, figsize=(7, 5))
                power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=1.5, fmin=1, fmax=3,
                                   baseline=baseline, mode=baseline_mode, axes=axis[0],
                                   title='Delta', show=False, contours=1)
                power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=1.5, fmin=4, fmax=7,
                                   baseline=baseline, mode=baseline_mode, axes=axis[1],
                                   title='Theta', show=False, contours=1)
                power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=1.5, fmin=8, fmax=12,
                                   baseline=baseline, mode=baseline_mode, axes=axis[2],
                                   title='Alpha', show=False, contours=1)
                power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=1.5, fmin=13, fmax=30,
                                   baseline=baseline, mode=baseline_mode, axes=axis[3],
                                   title='Beta', show=False, contours=1)
                mne.viz.tight_layout()
                plt.pyplot.show()
                #joint plot
                power.plot_joint(baseline=baseline, mode='mean', tmin=None, tmax=None,
                                 timefreqs=[(0.05, 2.), (0.1, 11.)])
                plt.pyplot.show()
                #inspect itc
                itc.plot_topo(title='Inter-Trial coherence', vmin=0., vmax=0.5, cmap='Reds')

            #https: // raphaelvallat.com / bandpower.html

            tvns_data = np.array(epochs['tvns/off'].average().get_data())*10e6
            sham_data = np.array(epochs['sham/off'].average().get_data())*10e6
            df_tvns = bandpower(tvns_data, sf=512, ch_names=epochs.ch_names, relative=True)
            df_sham = bandpower(sham_data, sf=512, ch_names=epochs.ch_names, relative=True)
            tvns_power.append(df_tvns.loc[CHANNELS].mean())
            sham_power.append(df_sham.loc[CHANNELS].mean())
            #ratio = db_all.loc[:,'Delta']/db_all.loc[:,'Alpha']
            # global average
            #av_ratio=db_all.loc[:,'Theta'] / db_all.loc[CHANNELS, 'Alpha'].mean()
    #t-tests
    tvns_power=np.array(tvns_power)
    sham_power=np.array(sham_power)
    alpha_col = list(df_tvns.columns).index('Alpha')
    theta_col = list(df_tvns.columns).index('Theta')
    delta_col = list(df_tvns.columns).index('Delta')

    t_test = stats.ttest_rel(tvns_power[:,alpha_col], sham_power[:,alpha_col],alternative="less")
    print("dependent Alpha power t-test, tvns < sham:")
    print(tvns_power[:,alpha_col].mean(),'<',sham_power[:,alpha_col].mean(), t_test)
    t_test = stats.ttest_rel(tvns_power[:,theta_col], sham_power[:,theta_col])
    print("dependent Theta power t-test, <> sham:")
    print(tvns_power[:,theta_col].mean(),'<>',sham_power[:,theta_col].mean(), t_test)
    t_test = stats.ttest_rel(tvns_power[:,delta_col], sham_power[:,delta_col])
    print("dependent Delta power t-test, tvns <> sham:")
    print(tvns_power[:,delta_col].mean(),'<>',sham_power[:,delta_col].mean(), t_test)
    t_test = stats.ttest_rel(tvns_power[:,delta_col]/tvns_power[:,alpha_col], sham_power[:,delta_col]/sham_power[:,alpha_col])
    print(f"Delta to Alpha ratio: {t_test}")
    print((tvns_power[:,delta_col]/tvns_power[:,alpha_col]).mean(),'<>',(sham_power[:,delta_col]/sham_power[:,alpha_col]).mean())





