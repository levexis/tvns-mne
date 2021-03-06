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

from clean_events import Participant,_CHECKED

SUBJ=1
CHARTS=False
T_MIN = 0
T_MAX = 1
T_INT = .05
L_PASS = None
TSHIFT = 0
H_PASS = None
TOPOMAPS = False
DETREND = None # linear detrend
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

def write_evoked(participant_number,epochs):

    # lowpass to 10 for ERP analysis, the lower the filter the longer the data window needs to be
    shorter_epochs = epochs.copy().filter(H_PASS, L_PASS, fir_design='firwin')

    ## crop to time period of interest for ERP identification
    shorter_epochs.crop(tmin=T_MIN, tmax=T_MAX, include_tmax=True)


    ## COMPARE ERPS
    evoked_tvns = shorter_epochs['tvns'].average()
    evoked_sham = shorter_epochs['sham'].average()
    if DETREND is not None:
        print(f'detrending ({DETREND})')
        evoked_tvns.detrend(DETREND)
        evoked_sham.detrend(DETREND)

    # save evoked
    mne.write_evokeds(f'out_evoked/evoked_tvns_sham_P{participant.part_str}-ave.fif', [evoked_tvns, evoked_sham])

    return evoked_tvns,evoked_sham


if __name__ == '__main__':
    # get options, default is no charts -p6 -c also switches on charts
    argv = sys.argv[1:]
    ALL=False
    help_mess="evoke_tvns -- -p <participant number> -c -h0.1 -l15 [--all] [--charts=n] [--topomaps] [--tmin=0] [--tmax=1] [--tint=.05] [--detrend=none/0/1] [--hpass=0.1] [--lpass=15] [--tshift=0]"
    p_charts = False
    if (len(argv)):
        CHARTS = False
        try:
            opts, args = getopt.getopt(argv, "p:c", ["charts=","all",'tmin=','tmax=','tint=','pcharts','detrend=','topomaps','hpass=','lpass=','tshift='])
        except:
            print("invalid command line arguments:")
            print(help_mess)
            sys.exit()
        print(opts)
        for opt, arg in opts:
            if opt == '-h':
                print(help_mess)
            elif opt == '--lpass':
                L_PASS = int(arg)
            elif opt == '--hpass':
                H_PASS = float(arg)
            elif opt == '--tint':
                T_INT = float(arg)
            elif opt == '--tmin':
                T_MIN = float(arg)
            elif opt == '--tmax':
                T_MAX = float(arg)
            elif opt == '--detrend':
                if arg=='none':
                    DETREND = None
                else:
                    DETREND = int(arg)
            elif opt == '--pcharts':
                p_charts = True
            elif opt in ["-c", "--charts"]:
                if arg == "n":
                    CHARTS = False
                else:
                    CHARTS = True
            elif opt == '--topomaps':
                TOPOMAPS = True
            elif opt == '--tshift':
                TSHIFT = float(arg)
                print (TSHIFT)
            elif opt == '--all':
                ALL = True
    if ALL:
        participants = _CHECKED
    else:
        participants = [ SUBJ ]

    tvns_evokeds = []
    sham_evokeds = []
    total_epochs = 0
    stim_epochs = 0
    for subj in participants:
        participant = Participant(subj)
        part_file = f"out_epochs/cleaned_stimoff_epoch_sub-{participant.part_str}-epo.fif"
        print(f"P{subj} loading {part_file}")
        epochs = mne.read_epochs(part_file)
        print(TSHIFT)
        if TSHIFT!=0:
            print(f"shifting time by {TSHIFT} offset")
            epochs.shift_time(TSHIFT)

            # enable to save epochs with new offset
            #epochs.save(f'out_epochs/cleaned_stimoff_epoch_sub-{participant.part_str}-epo.fif', overwrite=True)

        print (f"Epochs from {round(epochs.times[0],2)} to {round(epochs.times[-1],2)}")
        total_epochs += len(epochs)
        stim_epochs += len(epochs['tvns'])
        evoked_tvns, evoked_sham = write_evoked(subj,epochs)
        tvns_evokeds.append(evoked_tvns)
        sham_evokeds.append(evoked_sham)
        if p_charts:
            fig = mne.viz.plot_compare_evokeds([evoked_tvns, evoked_sham], picks='eeg', combine='mean')
            format_fig(fig[0], f'Paricipant {subj} Evoked Comparison',f'Evoked Comparison Participant: {subj}')

    print(f"Processed {total_epochs} epochs from {len(participants)} participants ({(total_epochs*100)/(len(participants)*88)}%) Stim: {stim_epochs}, Sham: {total_epochs-stim_epochs}")
    print(CHARTS)
    if CHARTS:
        tvns_grand = mne.grand_average(tvns_evokeds)
        sham_grand = mne.grand_average(sham_evokeds)
        fig = mne.viz.plot_compare_evokeds({'tvns':tvns_grand, 'sham': sham_grand},
                                           picks='eeg',
                                           combine='mean',
                                           colors=dict(tvns='orange',sham='green'),
                                           linestyles=dict(tvns='solid',sham='dotted'))
        format_fig(fig[0],'Grand Evoked Comparison')
        # difference wave
        grand_diff = mne.combine_evoked([tvns_grand,sham_grand],weights=[1, -1])
        fig = mne.viz.plot_compare_evokeds({'tVNS - Sham':grand_diff}, picks='eeg', combine='mean')
        format_fig(fig[0],'Difference Wave, Stim-Sham')

        if TOPOMAPS:
            # plot 0 to 1 with 50ms steps (inclusive)
            timepoints = np.arange(T_MIN, T_MAX + T_INT, T_INT)
            format_fig(sham_grand.plot_topomap(timepoints),
                       'SHAM Topomaps',
                       'SHAM Grand Topomaps')
            format_fig(tvns_grand.plot_topomap(timepoints),
                       'tVNS Topomaps',
                       'tVNS Grand Topomaps')
        input("Press Enter to exit (closes charts)...")