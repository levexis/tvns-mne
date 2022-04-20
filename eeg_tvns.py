#!/usr/bin/env ipython
# see https://mne.discourse.group/t/eeg-processing-pipeline-with-autoreject/3443
# https://autoreject.github.io/stable/auto_examples/plot_auto_repair.html

import matplotlib as plt
import mne
import numpy as np
import os
import sys
import getopt
from autoreject import AutoReject
from autoreject import get_rejection_threshold
from clean_events import Participant, check_stim_artifacts

from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)

# For interactive plotting, load the following backend:
plt.use('Qt5Agg')


#can set the subject number here if running from an IDE
SUBJ = 17
DATA_DIR = os.path.expanduser('~') + '/win-vr/eegdata'
CHARTS = False

def pre_process (participant_number, show_charts = False):
    participant = Participant(participant_number)

    ### IMPORT AND RE-REFERENCE EEG FILE
    # Read raw EEG file and visually inspect for bad channels
    raw_eeg = mne.io.read_raw_bdf(f"{DATA_DIR}/{participant.filename}", preload=True)  # Load bdf file to enable re-referencing"
    raw_eeg.set_eeg_reference(ref_channels=['EXG5', 'EXG6'])  # Take average of mastoids as reference
    # drop the mastoid reference channels so they don't interfere EOG analysis
    raw_eeg.drop_channels(['EXG5','EXG6'])
    # interpolate bad channels (note this is also done by autoreject so can maybe remove this)
    raw_eeg.info['bads'].extend(participant.bad_channels)
    # these require ipython to run or they just crash. Plot accepts (block=True) which fixes that

    if show_charts:
        raw_eeg.plot_psd()
        raw_eeg.plot(block=True)

    ### DEFINE ELECTRODE MAP

    # Specify specific montage used
    biosemi_montage = mne.channels.make_standard_montage("biosemi64")
    raw_eeg.set_montage(biosemi_montage, on_missing='ignore')
    # set all external electrodes to type EOG for ICA.
    raw_eeg.set_channel_types({'EXG1': 'eog', 'EXG2': 'eog', 'EXG3': 'eog', 'EXG4': 'eog'}) # 'EXG5': 'eog', 'EXG6': 'eog','EXG7': 'eog', 'EXG8': 'eog'})
    raw_eeg.drop_channels(participant.exclude_channels)  # Define at top if extra channels need to be excluded
    #raw_eeg.plot_sensors(ch_type='eeg')

    # # get an error here, apply later - bug? ValueError: array must not contain infs or NaNs
    # raw_eeg.interpolate_bads()

    ### FIND STIM/OFF EVENTS

    #event div
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

    off_event_id={
        'tvns/off': 31+34,
        'sham/off': 32+34}

    stim_epochs = mne.Epochs(raw_eeg, off_events, off_event_id, tmin=-4, tmax=1, baseline=(None,-3.6), preload=True)

    #check for stim artifact in all epochs, problems should be resolved by running clean_events.py and adding BAD_ annotations
    check_stim_artifacts(stim_epochs,participant)

    # now drop stim channel
    raw_eeg.drop_channels(['EXG7', 'EXG8']) # drop the stimulation electrodes, remainder are used for EOG artifacts


    ### PREPROCESS RAW EEG ###
    #only now I can interpolate the raw data without an error (bug?)
    raw_eeg.interpolate_bads() #THIS SHOULD BE DONE BY AUTOREJECT but not convinced as getting ERP at bad electrode


    ###BAND AND NOTCH FILTERS ###

    #now bandpass filter
    filtered_eeg = raw_eeg.copy().filter(0.1, 40., fir_design='firwin')  # Filter between 0.1Hz and 40Hz

    # remove 25Hz stim artifact and 50Hz mains artifact, iir = butterworth filter
    filtered_eeg.notch_filter([25],method='iir')
    filtered_eeg.notch_filter([50],method='iir') #possibly superflous given the 40Hz lowpass but 50Hz artifact is large

    #filtered_eeg.plot_psd(area_mode='range',average=False)
    #filtered_eeg.plot()


    ### ICA ####

    # Set up ICA
    method = 'picard' #'fastica'

    n_components = .99 # should be .99 % of variance
    max_iter = 500
    random_state = 101  # Random seed
    ica = ICA(n_components=n_components, method=method, random_state=random_state,max_iter=max_iter)

    # ica works better with a 1Hz high pass filter,
    # note: I think this is done by ica.apply so superfluous
    # can make ica_eeg = filtered_eeg to reverse this
    ica_eeg = raw_eeg.filter(1, 40, fir_design='firwin')

    picks = mne.pick_types(ica_eeg.info, eeg=True, eog=True,
                           stim=False)

    # Fit ICA
    ica.fit(ica_eeg, picks=picks, decim=None, reject=None)

    # Check ICA components
    #ica.plot_components()

    # Check EOG events
    # todo: will this miss saccades as blink artifacts are much higher amplitude. ICA components look same if only using EXG2
    eog_epochs = create_eog_epochs(ica_eeg, baseline=(-0.5, 0))
    #eog_epochs.plot_image(combine='mean')
    #eog_epochs.average().plot_joint()

    # Use EOG epochs to reject ICA components ------------------------------------------------------------------------------
    ica.exclude = []
    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(ica_eeg)
    ica.exclude = eog_indices

    # check for saccades as these don't always get spotted
    saccade_scores = (np.absolute(eog_scores[0])+np.absolute(eog_scores[1]))/2
    blink_scores = (np.absolute(eog_scores[0])+np.absolute(eog_scores[1]))/2

    #arbitrary cut off at abs score of .5, do blinks and saccade channels. Need to check this is not too sensitive
    eog_indices.extend(np.where(saccade_scores > 0.5)[0])
    eog_indices.extend(np.where(blink_scores > 0.5)[0])
    #remove duplicates
    eog_indices = list(set(eog_indices))

    print (f"Removing {len(eog_indices)} ICA components based on EOG artifact scores")

    # plot all the topomaps
    # ica.plot_components()

    if show_charts:
        ica.plot_components()
        ica.plot_sources(filtered_eeg, show_scrollbars=False)
        ica.plot_scores(eog_scores)
        print(f"blink scores: {blink_scores}")
        print(f"saccade scores: {saccade_scores}")
        # ica.plot_scores(blink_scores)
        # ica.plot_scores(saccade_scores)
        if eog_indices:
            ica.plot_properties(filtered_eeg, picks=eog_indices)

    # Finally, apply the ICA exclude to the actual data
    ica.apply(filtered_eeg)

    #EPOCHS around off_events

    tmin, tmax = (-4, 1.5)
    baseline = (-4, -3.6)

    # linear detrended epochs

    epochs = mne.Epochs(filtered_eeg, off_events, event_id=off_event_id, tmin=tmin, tmax=tmax, baseline=baseline,
                        detrend=1, preload=True)

    # drop the ones without stimulation
    epochs.drop(participant.bad_epochs)

    # adjust for stimulation robot etc latency
    epochs.shift_time(participant.stim_offset)

    ### AUTOREJECT epochs with OUTLIERS ####

    ar = AutoReject(random_state=101)

    ar = AutoReject(random_state=101).fit(epochs)
    epochs_clean, reject_log = ar.transform(epochs, return_log=True)
    if show_charts and any(reject_log.bad_epochs):
        reject_log.plot('horizontal')
        epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))

#    epochs_clean = ar.fit_transform(epochs)
#    epochs_clean = epochs

    #epochs_clean.plot()


    ### SAVE CLEANED EPOCHS

    epochs_clean.save(f'out_epochs/cleaned_stimoff_epoch_sub-{participant.part_str}-epo.fif', overwrite=True)

    ## crop to time period of interest for ERP identification
    shorter_epochs = epochs.copy().crop(tmin=0, tmax=1, include_tmax=True)

    ## COMPARE ERPS
    evoked_tvns = shorter_epochs['tvns'].average()
    evoked_sham = shorter_epochs['sham'].average()

    #save evoked
    mne.write_evokeds(f'out_evoked/evoked_tvns_sham_P{participant.part_str}-ave.fif', [evoked_tvns,evoked_sham])

    #mne.viz.plot_compare_evokeds([evoked_tvns, evoked_sham], picks='eeg', combine='mean')
    # with confidence intervals
    evoked = dict(tvns=list(shorter_epochs['tvns'].iter_evoked()),
                  sham=list(shorter_epochs['sham'].iter_evoked()))
    if CHARTS:
        evoked_sham.plot_topomap()
        evoked_tvns.plot_topomap()
        mne.viz.plot_compare_evokeds(evoked, combine='mean', picks='eeg')
        # show underlying epochs, need to have participant.bad_epochs to remove a manual ist
        filtered_eeg.plot_psd()
        epochs_clean.plot(block=True)

    return epochs_clean

if __name__ == '__main__':
    # get options, default is no charts -p6 -c also switches on charts
    argv=sys.argv[1:]
    if (len(argv)):
        CHARTS=False
        try:
            opts,args= getopt.getopt(argv,"p:c",["charts="])
        except:
            print ("invalid command line arguments:")
            print ("eeg_tvns -- -p <participant number> [--charts=<y/n>]")
            sys.exit()
        for opt, arg in opts:
            if opt == '-h':
                print("eeg-tvns -- -p <participant number> [--charts=<y/n>]")
            elif opt == '-p':
                SUBJ = int(arg)
            elif opt in ["-c","--charts"]:
                if arg=="n":
                    CHARTS = False
                else:
                    CHARTS = True
    pre_process(SUBJ, show_charts=CHARTS)
    