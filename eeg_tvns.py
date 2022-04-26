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
SUBJ = 9
DATA_DIR = os.path.expanduser('~') + '/win-vr/eegdata'
CHARTS = True
ALL = False
RANDOM_STATE = 101  # Random seed


def format_fig(figure, window_tit='', canvas_tit=''):
    if window_tit:
        figure.canvas.manager.set_window_title(window_tit)
    if canvas_tit:
        figure.suptitle(canvas_tit)
    return figure

def pre_process(participant_number, show_charts=False):

    participant = Participant(participant_number)

    ### IMPORT AND RE-REFERENCE EEG FILE
    # Read raw EEG file and visually inspect for bad channels
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

    # Specify specific montage used
    biosemi_montage = mne.channels.make_standard_montage("biosemi64")
    raw_eeg.set_montage(biosemi_montage, on_missing='ignore')
    # set all external electrodes to type EOG for ICA.
    raw_eeg.set_channel_types(
        {'EXG1': 'eog', 'EXG2': 'eog', 'EXG3': 'eog', 'EXG4': 'eog'})  # , 'EXG7': 'eog', 'EXG8': 'eog'})
    raw_eeg.drop_channels(participant.exclude_channels)  # Define at top if extra channels need to be excluded
    # raw_eeg.plot_sensors(ch_type='eeg')

    # # get an error here, apply later - bug? ValueError: array must not contain infs or NaNs
    # raw_eeg.interpolate_bads()

    ### FIND STIM/OFF EVENTS

    # event div
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
    check_stim_artifacts(stim_epochs, participant)

    # now drop stim channel
    raw_eeg.drop_channels(['EXG7', 'EXG8'])  # drop the stimulation electrodes, remainder are used for EOG artifacts

    ### PREPROCESS RAW EEG ###
    # only now I can interpolate the raw data without an error (bug?)
    # todo should this be done after ICA
    raw_eeg.interpolate_bads()  # Auto reject can fix channels are noisy but channels that are bad all the time are better done here.

    ###BAND AND NOTCH FILTERS ###
    # by default eog is not filtered, which means the stim artifact is there :(
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

    # ica_eeg = filtered_eeg.copy().filter(1,40, fir_design='firwin')
    # find which ICs match the EOG pattern
    # eog_epochs=create_eog_epochs(filtered_eeg,tmin=-3,tmax=3,l_freq=1,h_freq=30)
    # Autoreject (local) epochs to benefit ICA

    # having issues (participant4) where the eog channels seem clear but only get 5 epochs (unless you just pass exg4)
    # so hard coding a threshold of 200uV
    # reject_threshold = get_rejection_threshold(stim_epochs, random_state=RANDOM_STATE, ch_types='eog')

    eog_epochs = mne.preprocessing.create_eog_epochs(filtered_eeg, tmin=-1, tmax=1, thresh=200e-6,reject={'eeg': 200e-6})

    front_electrodes = ['Fp1', 'Fpz', 'Fp2', 'AF8', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8']

    # just look for artifacts away for eyes
    ar_epochs = eog_epochs.copy().pick_channels([ch for ch in eog_epochs.ch_names if ch not in front_electrodes])
    auto_reject_pre_ica = AutoReject(random_state=RANDOM_STATE, n_interpolate=[1, 2, 3, 4], n_jobs=1).fit(
        ar_epochs[:200])  # just use first 20 epochs to save time
    #    epochs_ar, reject_log = auto_reject_pre_ica.transform(ar_epochs, n_interpolate=[1,2,3,4], return_log=True)
    reject_log = auto_reject_pre_ica.get_reject_log(ar_epochs)

    if show_charts:
        # epochs_ar[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))
        reject_log.plot('horizontal')

    # reject_threshold = get_rejection_threshold(ar_epochs, random_state=RANDOM_STATE) #if using
    # we want to keep the eog artifacts for ICA (could actually pass ch_types='eeg'
    # del reject_threshold['eog']

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
        #        ica.plot_sources(ica_eeg)
        format_fig(ica.plot_sources(eog_epochs),'Component Sources','EOG Epochs')
        format_fig(ica.plot_scores(eog_scores),
                   'Component Scores',
                   '1/2 horizontal, 3/4 vertical')
        print(f"blink scores: {blink_scores}")
        print(f"saccade scores: {saccade_scores}")
        # ica.plot_scores(blink_scores)
        # ica.plot_scores(saccade_scores)
        # plot_properties throwing errors with reject criteria specified...
    #        if eog_indices:
    #            ica.plot_properties(filtered_eeg, picks=eog_indices)

    # EPOCHS around off_events
    tmin, tmax = (-4, 1.5)
    baseline = (-4, -3.6)  # off event
    #    baseline = (-4, .5 - 3.6) #onset event

    # linear detrended epochs
    epochs = mne.Epochs(filtered_eeg, off_events, event_id=off_event_id, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, detrend=1, reject={'eeg': 200e-6})

    # Finally, apply the ICA to the epoched data
    #    ica.apply(filtered_eeg)
    ica.apply(epochs)
    # if not using epochs could apply ICA to filtered raw and create epochs here

    epochs.apply_baseline(baseline)

    epochs.shift_time(participant.stim_offset)  # offset
    #    epochs.shift_time(3.6-.5) #onset

    # epochs_clean.apply_baseline(baseline)

    # dirty epochs, allows charting of what was removed
    dirty_epochs = epochs.copy()

    # drop the ones without stimulation
    epochs = epochs.drop(participant.bad_epochs, 'no stim')

    ### AUTOREJECT ####
    ar_post_ica = AutoReject(random_state=RANDOM_STATE).fit(epochs)
    epochs_clean, reject_log = ar_post_ica.fit_transform(epochs, return_log=True)
    # reject_log = { 'bad_epochs' : []}
    # reject_threshold = get_rejection_threshold(epochs, random_state=RANDOM_STATE)
    # del reject_threshold['eog']
    # print(f'post ICA rejection threshold {reject_threshold}')

    #    epochs.drop_bad(reject=reject_threshold)
    #    epochs_clean = epochs

    # if show_charts:
    #    if len(reject_log.bad_epochs):
    #        epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))
    #    reject_log.plot('horizontal')

    epochs_clean.save(f'out_epochs/cleaned_stimoff_epoch_sub-{participant.part_str}-epo.fif', overwrite=True)

    # lowpass to 10 for ERP analysis, the lower the filter the longer the data window needs to be
    shorter_epochs = epochs_clean.copy().filter(None, 10, fir_design='firwin')

    ## crop to time period of interest for ERP identification
    shorter_epochs.crop(tmin=0, tmax=1, include_tmax=True)


    ## limit to parietal channels for P3
    #    parietal_channels = [channel for channel in epochs.ch_names if 'P' in channel]
    parietal_channels = ['CP1', 'CPz', 'CP2', 'P1', 'Pz', 'P2']

    ## COMPARE ERPS
    evoked_tvns = shorter_epochs['tvns'].average().detrend(1)
    evoked_sham = shorter_epochs['sham'].average().detrend(1)

    # save evoked
    mne.write_evokeds(f'out_evoked/evoked_tvns_sham_P{participant.part_str}-ave.fif', [evoked_tvns, evoked_sham])

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
        # show underlying epochs, need to have participant.bad_epochs to remove a manual ist
        # filtered_eeg.plot_psd()
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
    if (len(argv)):
        CHARTS = False
        try:
            opts, args = getopt.getopt(argv, "p:c", ["charts=","all"])
        except:
            print("invalid command line arguments:")
            print("eeg_tvns -- -p <participant number> [--charts=<y/n>]")
            sys.exit()
        for opt, arg in opts:
            if opt == '-h':
                print("eeg-tvns -- -p <participant number> [--charts=<y/n>]")
            elif opt == '-p':
                SUBJ = int(arg)
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