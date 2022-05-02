#!/usr/bin/env ipython
# see https://mne.discourse.group/t/eeg-processing-pipeline-with-autoreject/3443
# https://autoreject.github.io/stable/auto_examples/plot_auto_repair.html

import matplotlib as plt
import mne
import numpy as np
import os
import sys
import getopt
from autoreject import AutoReject, get_rejection_threshold

from clean_events import Participant, check_stim_artifacts, _CHECKED

from mne.preprocessing import (ICA, create_eog_epochs)

# For interactive plotting, load the following backend:
plt.use('Qt5Agg')

# can set the subject number here if running from an IDE
SUBJ = 20
DATA_DIR = os.path.expanduser('~') + '/win-vr/eegdata'
CHARTS = True
ALL = False
RANDOM_STATE = 101  # Random seed
DETREND_EPOCHS = 1

def format_fig(figure, window_tit='', canvas_tit=''):
    if window_tit:
        figure.canvas.manager.set_window_title(window_tit)
    if canvas_tit:
        figure.suptitle(canvas_tit)
    return figure

def pre_process(participant_number, show_charts=False):

    participant = Participant(participant_number)

    ### IMPORT AND RE-REFERENCE EEG FILE
    raw_eeg = mne.io.read_raw_bdf(f"{DATA_DIR}/{participant.filename}",
                                  preload=True)  # Load bdf file to enable re-referencing"
    raw_eeg.set_eeg_reference(ref_channels=['EXG5', 'EXG6'])  # Take average of mastoids as reference
    # drop the mastoid reference channels so they don't interfere EOG analysis
    raw_eeg.drop_channels(['EXG5', 'EXG6'])
    # interpolate bad channels (note this is also done by autoreject so can maybe remove this)
    raw_eeg.info['bads'].extend(participant.bad_channels)
    # these require ipython to run or they just crash. Plot accepts (block=True) which fixes that

    if show_charts:
        format_fig(raw_eeg.plot_psd(fmax=100),'Raw EEG Spectrum')
        format_fig(raw_eeg.plot(block=True, scalings=dict(eeg=100e-6, eog=100e-6)),
                   'Raw EEG unprocessed')

    ### DEFINE ELECTRODE MAP
    biosemi_montage = mne.channels.make_standard_montage("biosemi64")
    raw_eeg.set_montage(biosemi_montage, on_missing='ignore')
    # set all external electrodes to type EOG for ICA.
    raw_eeg.set_channel_types(
        {'EXG1': 'eog', 'EXG2': 'eog', 'EXG3': 'eog', 'EXG4': 'eog'})  # , 'EXG7': 'eog', 'EXG8': 'eog'})
    raw_eeg.drop_channels(participant.exclude_channels)  # Define at top if extra channels need to be excluded

    ### FIND STIM/OFF EVENTS
    event_id = {
        'tvnsblock': 31,
        'shamblock': 32,
        'stim/on': 33,
        'stim/off': 34,
        'break/start': 45,
        'break/stop': 46,
    }

    events = participant.load_clean_events(raw_eeg)
    off_events = participant.get_offset_events()

    off_event_id = {
        'tvns/off': 31 + 34,
        'sham/off': 32 + 34}

    stim_epochs = mne.Epochs(raw_eeg, off_events, off_event_id, tmin=-4, tmax=1, baseline=(None, -3.6), preload=True)
    # check for stim artifact in all epochs, problems should be resolved by running clean_events.py and adding BAD_ annotations
    # this should be done running clean_events,
    check_stim_artifacts(stim_epochs, participant)

    # now drop stim channel
    raw_eeg.drop_channels(['EXG7', 'EXG8'])  # drop the stimulation electrodes, remainder are used for EOG artifacts

    ### PREPROCESS RAW EEG ###
    # only now I can interpolate the raw data without an error (bug?)
    raw_eeg.interpolate_bads()  # Auto reject can fix channels are noisy but channels that are bad all the time are better done here.

    ###BAND AND NOTCH FILTERS ###
    # eog filtering is false by default so set picks to include
    picks = mne.pick_types(raw_eeg.info, eeg=True, eog=True,
                           stim=False)

    # now bandpass filter todo: changed to 20 to see if artifacts go
    filtered_eeg = raw_eeg.copy().filter(0.1, 40, fir_design='firwin', picks=picks)  # Filter between 0.1Hz and 40Hz

    # remove 25Hz stim artifact and 50Hz mains artifact, iir = butterworth filter
    filtered_eeg = filtered_eeg.notch_filter([25], method='iir', notch_widths=1, picks=picks)
    filtered_eeg = filtered_eeg.notch_filter([50], method='iir', notch_widths=.5, picks=picks)

    ### ICA ####
    method = 'picard'  # infomax' #'picard' #'fastica'
    n_components = .99  # should be .99 % of variance
    max_iter = 500  # normally 500
    ica = ICA(n_components=n_components, method=method, random_state=RANDOM_STATE, max_iter=max_iter)

    # https://erpinfo.org/blog/2018/6/18/hints-for-using-ica-for-artifact-correction#:~:text=Similarly%2C%20it%27s%20important%20to%20eliminate,lot%20of%20high%2Dfrequency%20noise.
    # A general heuristic is that the  # of time points must be greater than 20 x (# channels)^2.  The key is that the number of channels is squared.  So, with 64 channels, you would need 81,920 points (which would be about 5.5 minutes of data with a 250 Hz sampling rate).

    # having issues (participant4) where the eog channels seem clear but only get 5 epochs (unless you just pass exg4)
    # to fix this I have hard coded a threshold of 200uV which produces more epochs
    # reject_threshold = get_rejection_threshold(stim_epochs, random_state=RANDOM_STATE, ch_types='eog')
    eog_epochs = mne.preprocessing.create_eog_epochs(filtered_eeg, tmin=-1, tmax=1, thresh=200e-6)

    front_electrodes = ['Fp1', 'Fpz', 'Fp2', 'AF8', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8']

    # just look for artifacts away for eyes
    ar_epochs = eog_epochs.copy().pick_channels([ch for ch in eog_epochs.ch_names if ch not in front_electrodes])
    #reject bad on selected channels based on extreme values, this provides cleaner data for ICA than autoreject
    ar_epochs.drop_bad(reject={'eeg': 200e-6})
    # need to keep eog_epochs same size
    #dropped = [idx-1 for idx in range(len(eog_epochs)+1) if len(ar_epochs.drop_log[idx]) and idx>0]
    drop_log = ar_epochs.drop_log
    crash_test=0
    # seem to get an additional NO_DATA epoch at the start of the log (bug?)
    while len(drop_log)>len(eog_epochs) and len(drop_log[0])==1:
        # only seen these at the start and the end, extra positions in array.
        if drop_log[0][0]!='NO_DATA' and drop_log[0][0]!='TOO_SHORT':
            #exit loop if not one of these
            break
        drop_log=drop_log[1:]
        crash_test=crash_test+1
        if crash_test>100000:
            raise Exception(f'Stuck in drop_log loop {drop_log}')

    dropped = [idx for idx in range(len(drop_log)) if len(drop_log[idx]) and drop_log[idx][0]!='TOO_SHORT' and drop_log[idx][0]!='NO_DATA' ]
    eog_epochs.drop(dropped, 'pre-AR non-EOG threshold')

    auto_reject_pre_ica = AutoReject(random_state=RANDOM_STATE, n_interpolate=[1, 2, 3, 4], n_jobs=1).fit(
        ar_epochs[:200])  # just use first 20 epochs to save time
    reject_log = auto_reject_pre_ica.get_reject_log(ar_epochs)

    if show_charts:
        # epochs_ar[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))
        reject_log.plot('horizontal')


    # Fit ICA

    # picks is including the eog channels in the ICA solution, will cause an error if applied to just EEG
    # looking at first 200 epochs (should be enough - need to check the number of timesamples)
    ica.fit(eog_epochs[~reject_log.bad_epochs][:200])  # ,picks=picks)
    # ica.fit(ica_eeg, picks=picks, reject=reject_threshold)

    ica.exclude = []

    eog_indices, eog_scores = ica.find_bads_eog(eog_epochs, measure='zscore')
    ica.exclude = eog_indices

    # check for saccades as these don't always get spotted
    saccade_scores = (np.absolute(eog_scores[0]) + np.absolute(eog_scores[1])) / 2
    blink_scores = (np.absolute(eog_scores[2]) + np.absolute(eog_scores[3])) / 2

    # arbitrary cut off at abs score of .5, do blinks and saccade channels. Need to check this is not too sensitive
    eog_indices.extend(np.where(saccade_scores > 0.5)[0])
    eog_indices.extend(np.where(blink_scores > 0.5)[0])
    # remove duplicates
    eog_indices = list(set(eog_indices))

    #    print(np.argwhere(abs(eog_scores) > 0.5).ravel().tolist())

    print(f"Removing {len(eog_indices)} ICA components based on EOG artifact scores")

    if show_charts:
        format_fig(ica.plot_overlay(eog_epochs.average()),
                   'EOG epochs avg after ICA')
        format_fig(ica.plot_components()[0],'Component Topomaps')
        format_fig(ica.plot_sources(eog_epochs),'Component Sources','EOG Epochs')
        format_fig(ica.plot_scores(eog_scores),
                   'Component Scores',
                   '1/2 horizontal, 3/4 vertical')
        print(f"blink scores: {blink_scores}")
        print(f"saccade scores: {saccade_scores}")


    # EPOCHS around off_events
    tmin, tmax = (-4, 1.5)
    baseline = (-4, -3.6)  # off event


    # linear detrended epochs
    epochs = mne.Epochs(filtered_eeg, off_events, event_id=off_event_id, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, detrend=DETREND_EPOCHS)

    # Finally, apply the ICA to the epoched data
    ica.apply(epochs)

    epochs.apply_baseline(baseline)

    epochs.shift_time(participant.stim_offset)  # offset
    # epochs.shift_time(3.6-.5) #comment in for onset

    # dirty epochs, allows charting of what was removed
    dirty_epochs = epochs.copy()

    # drop the ones without stimulation
    epochs = epochs.drop(participant.bad_epochs, 'no stim')
    # now drop the ones with a wild signal as autoreject doesn't always do this
    epochs.drop_bad(reject={'eeg': 200e-6})

    ### AUTOREJECT ####
    ar_post_ica = AutoReject(random_state=RANDOM_STATE).fit(epochs)
    epochs_clean, reject_log = ar_post_ica.fit_transform(epochs, return_log=True)

    epochs_clean.save(f'out_epochs/cleaned_stimoff_epoch_sub-{participant.part_str}-epo.fif', overwrite=True)

    # lowpass to 10 for ERP analysis, the lower the filter the longer the data window needs to be
    shorter_epochs = epochs_clean.copy().filter(None, 10, fir_design='firwin')

    ## crop to time period of interest for ERP identification
    shorter_epochs.crop(tmin=0, tmax=1, include_tmax=True)

    ## limit to parietal channels for P3 eboked chart
    parietal_channels = ['CP1', 'CPz', 'CP2', 'P1', 'Pz', 'P2']

    ## COMPARE ERPS
    evoked_tvns = shorter_epochs['tvns'].average().detrend(1)
    evoked_sham = shorter_epochs['sham'].average().detrend(1)

    # mne.viz.plot_compare_evokeds([evoked_tvns, evoked_sham], picks='eeg', combine='mean')
    # with confidence intervals
    evoked = dict(tvns=list(shorter_epochs['tvns'].pick_channels(parietal_channels).iter_evoked()),
                  sham=list(shorter_epochs['sham'].pick_channels(parietal_channels).iter_evoked()))
    if CHARTS:
        # for interactive functionality to work save needs to come after this
        format_fig(evoked_sham.plot(),'SHAM Evoked Offset')
        # plot 0 to 1 with 50ms steps (inclusive)
        timepoints = np.arange(0, 1.05, .05)
        format_fig(evoked_sham.plot_topomap(timepoints),
                   'SHAM Topomaps',
                   'SHAM Topomaps from Offset Event')
        format_fig(shorter_epochs['sham'].plot_psd(fmax=50),'SHAM Power Spectrum')
        format_fig(evoked_tvns.plot(),'tVNS Evoked Offset')
        format_fig(evoked_tvns.plot_topomap(timepoints),
                   'tVNS Topomaps',
                   'tVNS Topomaps from Offset Event')
        format_fig(shorter_epochs['tvns'].plot_psd(fmax=50),'tVNS Power Spectrum')
        # show spectrum power
        # shorter_epochs['tvns'].plot_psd_topomap()
        format_fig(mne.viz.plot_compare_evokeds(evoked, combine='mean', picks='eeg')[0],'Evoked Comparison Parietal Electrodes with CIs')
        # show underlying epochs
        format_fig(epochs_clean.plot(picks=picks, events=epochs_clean.events, block=True, scalings=dict(eeg=100e-6)),
                   'Final Raw Epochs after Preprocessing')
        # now work out the drop epochs list
        if False and (len(participant.bad_epochs) or True in reject_log.bad_epochs): #disabled need to match up dimensions of masks to add them
            bad_mask = np.full((1, len(dirty_epochs)), False)[0]
            if True in reject_log.bad_epochs:
                bad_mask = bad_mask + reject_log.bad_epochs

            for baddie in participant.bad_epochs:
                bad_mask[baddie - 1] = True

            dirty_epochs[bad_mask].plot(events=dirty_epochs.events, scalings=dict(eeg=100e-6))
            dirty_epochs[bad_mask].plot_psd()

    return epochs_clean


if __name__ == '__main__':
    # get options, default is no charts -p6 -c also switches on charts
    argv = sys.argv[1:]
    ALL = False
    if (len(argv)):
        CHARTS = False
        try:
            opts, args = getopt.getopt(argv, "p:c", ["charts=","all","detrend="])
        except:
            print("invalid command line arguments:")
            print("eeg_tvns -- -p <participant number> [--charts=<y/n>] [--detrend=none/0/1")
            sys.exit()
        for opt, arg in opts:
            if opt == '-h':
                print("eeg-tvns -- -p <participant number> [--charts=<y/n>] [--detrend=none/0/1")
            elif opt == '-p':
                SUBJ = int(arg)
            elif opt == '--detrend':
                if arg == 'none':
                    DETREND = None
                else:
                    DETREND = int(arg)
            elif opt in ["-c", "--charts"]:
                if arg == "n":
                    CHARTS = False
                else:
                    CHARTS = True
            elif opt == '--all':
                ALL = True
    if ALL:
        participants = _CHECKED
    else:
        participants = [ SUBJ ]
    for subj in participants:
        print(f"partipant: {subj}, charts: {CHARTS}")
        pre_process(subj, show_charts=CHARTS)

    print (f"Preprocessed {len(participants)} participants")