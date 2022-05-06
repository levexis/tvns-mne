#!/usr/bin/env ipython
# coding: utf-8

# # Load modules
# run as script then walk through with charts for single participant for sorting out events
# if run as a module then it is being used to generate epochs.
#getopt.getopt(args, options, [long_options])

import matplotlib as plt
import mne
import numpy as np
import os
import sys
import getopt

#31 tvns - 65 off
#32 sham - 66 off

# For interactive plotting, load the following backend:
plt.use('Qt5Agg')

DATA_DIR = os.path.expanduser('~') + '/win-vr/eegdata'
# these can also be passed in as command line options
CHARTS = False
SUBJ = 17

#default offset for offevent, visual inspection of evoked stim channels shows zero at 320ms but this must be the upper range
#set to 300 as this is where there is a clear synchronisation of topomaps between conditions.
_OFFSET=-300/1000

#### definition of participant exceptions etc ###
# list of checked participants, allows validation that each subject has been visually inspected
_CHECKED = [1,2,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]# 3 & 5 dropped due to insufficient data

# participants where the order was reversed
_REVERSE_ORDER = [9,19,22]

# invalid will be ignored (bad trial), if block start events are correct then
# simpler to use annotations to exclude a bad trial and this is then shown on plots
# otherwise array length must match define all blocks that are extracted. use 'invalid' to exclude a block.
_BAD_EVENTS_BLOCKS = {8: ['shamblock','shamblock','shamblock','shamblock','shamblock','tvnsblock','tvnsblock','tvnsblock', 'tvnsblock', 'shamblock', 'shamblock'],
                      }
# can also exclude epochs using annotations
_ANNOTATIONS = {1 : [[2236,2281-2236,'noisy_exclude?']], #is this just high frequency noise?
                3: [['25.412109375','551.828125','BAD_no_stim_artifact']],
                9: [[265,605+254-265,'first_run_might_be_the_bad_one'],
                    [994,1489+77-994,'BAD_tvns_loc_incorrect']], #incorrect tvns electrode postion but was the order reversed?
                11: [['21.6796875','251.994140625','BAD_block_1_no_stim_artifact'],
                     ['701.998046875',577,'BAD_run2_participant_no_stim']],  # spans 2 blocks
                12: [[2300,554, 'BAD_run4_no_stim_artifact']],#last 2 blocks, experimenters mention fixing the tVNS unable to identify valid trials
                16: [['852.98828125', '51.279296875', 'BAD_no_stim']]}  # [ startime, duration, 'BAD_reason']
# bad channels, these will be interpolated
_BAD_CHANNELS = { 1: ['FT7'],
                  2: ['PO8'], #breaks ICA - 1 component
                  6: ['C6'],
                  7: ['P2'],
                  8: ['C1','Fp2'],
                  9: ['C6'],
                  10: ['P2'],
                  11: ['Pz'],
                  12: ['P2','POz','P7','FT7','F7'], #P7,FT7,F7 causing all EOG epochs to be rejected
                  13: ['Iz','Oz','POz','O2'], #is this an issue, they look to be three in a row?
                  15: ['P3','P9'],
                  16: ['T7','P2','PO4'],
                  #17: ['FT8'], # outlier in evoked, FT8 often rejected
                  19: ['C6'],#intermitent spikes
                  20: ['FC2','PO4','C6','T7'], #also C6 & T7 also look weird, in fact all the data is very noisy
                  21: ['P2'],
                  22: ['FC6'],
                  23: ['PO4','P2','P4','O2']} #FC6 looks weird on spectrum plot
# list of epochs to drop because stim missing
# note: if using with annotations then do annotations first to be sure indexes are correct
_BAD_EPOCHS = { 1: [0,1],
                2: [50,51],
                6: [42,43,46,47,48],
                8: [45, 47, 49, 50, 52, 55, 57, 59, 60, 62, 64, 65, 66, 67, 68, 69, 70, 71],
                12: [44,45,46,66],
                13: [55, 56, 57, 58, 59, 80],
                15: [65],
                16: [36,37,38,39,40,41,42,43,44,45,46],
                18: [10, 11, 35, 36, 37],
                20: [66, 67, 68, 69, 70, 71, 72, 73, 74, 75],
                22: [0]}

# fine adjustment for offset event (if needed)
_OFFSETS ={}


class Participant:

    def __init__(self, part_num):
        data_dir = os.path.expanduser('~') + '/win-vr'
        self.number = part_num
        self.checked = (part_num in _CHECKED)
        self.part_str = str(part_num).zfill(3)
        self.filename = f"P{self.part_str}.bdf"
        self.exclude_channels = []
        if part_num in _BAD_CHANNELS:
            self.bad_channels =_BAD_CHANNELS[part_num]
        else:
            self.bad_channels = []
        if part_num in _BAD_EPOCHS:
            self.bad_epochs = _BAD_EPOCHS[part_num]
        else:
            self.bad_epochs = []

        # order of stimulation
        if part_num % 2 and part_num not in _REVERSE_ORDER \
                or part_num in _REVERSE_ORDER and not part_num % 2:
            #odd participants
            self.first_condition = 'tvnsblock'
            self.last_condition = 'shamblock'
        else:
            self.first_condition = 'shamblock'
            self.last_condition = 'tvnsblock'

        # if events are missing and we have more than 8 pseudo blocks, or bad trials
        self.bad_blocks = part_num in _BAD_EVENTS_BLOCKS
        if part_num in _OFFSETS:
            self.stim_offset = _OFFSETS[part_num]
        else:
            self.stim_offset = _OFFSET

#True should return normal conditon, false should return the reverse.
#But what is the normal condition as we have more than 8 blocks

    def get_block_type(self, block_number, current_type = None):
        # false will return the normal condition, use just to exclude an invalid block
        if self.bad_blocks:
            if type(_BAD_EVENTS_BLOCKS[self.number][block_number]) is str:
                return _BAD_EVENTS_BLOCKS[self.number][block_number]
            # else must be boolean True so pass
            elif type(_BAD_EVENTS_BLOCKS[self.number][block_number]) is bool:
                #first conditon is to switch default if False specified
                if not current_type:
                    if not _BAD_EVENTS_BLOCKS[self.number][block_number]:
                        # wrong block was used reverse default if False, keep same if true
                        block_number += 1
                elif _BAD_EVENTS_BLOCKS[self.number][block_number]:
                    #other conditions work on the last block_start events
                    return current_type
                else:
                    #switch the block type (experimenter error)
                    if current_type==self.first_condition:
                        return self.last_condition
                    else:
                        return self.first_condition

        # default behaviour
        if block_number % 2:
            return self.first_condition
        else:
            return self.last_condition

    def get_annotations(self):
        if self.number in _ANNOTATIONS:
            annotations = np.array(_ANNOTATIONS[self.number])
            return mne.Annotations(annotations[:, 0],
                                   annotations[:, 1],
                                   annotations[:, 2])


    def load_clean_events(self,raw_eeg):
        events = mne.find_events(raw_eeg, initial_event=True, shortest_event=1)
        # remove noise (spurious events) in events data
        self.events = clean_events(events)
        return self.events

    def get_offset_events(self):
        return get_block_events('stim/off', self)[0]


# process block signals and add them to subsequent stim events so you can find which trial atarted
def get_block_events(event_label, participant):
    if len(participant.events)==0:
        raise Exception('Need to call load_clean_events to load data before running get_block_events')

    # events we are expecting dict
    event_id = {
        'tvnsblock': 31,
        'shamblock': 32,
        'stim/on': 33,
        'stim/off': 34,
        'break/start': 45,
        'break/stop': 46,
    }
    off_events = []
    # odd sham first, even stim first - should be set at the start or particpant level
    # todo, why are we missing the first block start event?
    block = participant.first_condition
    blocks = []
    sample_rate = 512
    # not quite working on p7 or 8, needs to be shorter?
    block_timeout = 50 * sample_rate
    last_event = None
    # because of problems with missing block start codes we try to identify blocks. these timings are printed useful for creating annotations.
    for event in participant.events:
        new_block = False
        # use block to work out condition of stim_off event
        # could use my own blocks event to define type including ignore.
        if event[2] == event_id['tvnsblock']:
            # there was a bug in the code so tvns is always sent first, this means even participants (or experimenter error) the codes need reversing
            block = participant.first_condition #odd runs
        elif event[2] == event_id['shamblock']:
            block = participant.last_condition #even runs
        if event[2] == event_id[event_label]:
            # check if we are overiding the codes due to missing / experiment issues
            if (participant.bad_blocks):
                block_num = len(blocks)
                if not any(blocks) or event[0] > last_event[0] + block_timeout:
                    block_num +=1
                    new_block=True

                this_block =participant.get_block_type(block_num,block)
                # can use invalid or nostim or whatever you like to drop events (not a defined event type)
                if block in event_id:
                    event[2] += event_id[this_block]
                    off_events.append(event)
            else:
                event[2] += event_id[block]
                off_events.append(event)

            # find block boundaries for annotations etc
            if len(blocks) == 0:
                blocks.append([event[0] / sample_rate, 0, 'block_1'])
            elif new_block:
                # last block timed out, start new block
                blocks[-1][1] = (last_event[0] / sample_rate) - blocks[-1][0]  # duration
                blocks.append([event[0] / sample_rate, 0, f'block_{len(blocks) + 1}'])
            last_event = event
    # create last block
    blocks[-1][1] = (last_event[0] / sample_rate) - blocks[-1][0]  # duration
    return np.array(off_events), np.array(blocks)

def clean_events(events):
    # remove events divisible by 8192
    # events_dedup = np.unique(events[:,2]%8192, return_counts=True)
    # set event id column to remainder of div by 8192
    events[:, 2] = events[:, 2] % 8192
    # all of the 8192 events will be zero
    events_clean = events[events[:, 2] > 0]
    # 253 event is between blocks so can be useful if missing events
    #    events_clean = events_clean[events_clean[:,2]!=253]
    # now remove events that have been duplicated because they spanned a spurious one
    events_deduped = []
    EVENT_LATCH = 50  # min time between events
    for event in events_clean:
        # event id should not match previous, unless more than EVENT latch ms ago
        if 'last_event' not in locals() \
                or (event[2] != last_event[2]
                    or event[0] > last_event[0] + EVENT_LATCH):
            # append to valid events
            events_deduped.append(event)
        last_event = event

    events_deduped = np.array(events_deduped)
    # print("deduped events" + np.unique(events_deduped[:,2], return_counts='true')
    return events_deduped

def check_stim_artifacts(epochs,participant,show_charts = False):
    # dont alter the original
    epochs_no_stim = epochs.copy()
    epochs_no_stim.set_channel_types({'EXG7':'eog','EXG8':'eog'})
    # test for stimulation by rejecting all epochs with stim artifact
    reject_criteria = dict(
        eog=15e-6,  # 150 µV
    )
    #if len(participant.bad_epochs):
    #    print(f"dropping {len(participant.bad_epochs)} epochs marked bad previously")
    #    epochs_no_stim=epochs_no_stim.drop(participant.bad_epochs)

    # this should drop all the epochs
    epochs_no_stim = epochs_no_stim.drop_bad(reject=reject_criteria) #flat={'eog':10e-6})
    if len(epochs_no_stim):
        # probably need to warn here
        print(f"warning - stimulation artifact not in {len(epochs_no_stim)} epochs, check and add BAD annotations or bad_epochs:")
        epoch_index = 0
        for log_line in epochs_no_stim.drop_log:
            if not log_line:
                print (f"see epoch {epoch_index}")
            epoch_index += 1
        if show_charts:
            epochs_no_stim.set_channel_types({'EXG7': 'eeg', 'EXG8': 'eeg'})
            epochs_no_stim.pick_channels(['EXG7', 'EXG8']).plot()
    else:
            print(f"Stim sized artifact found all in all {len(epochs)} epochs")
    return epochs_no_stim

def check_participant(part_number,show_charts,epoch_event = 'stim/off'):
    participant = Participant(part_number)

    print(f"loading participant {participant.number} data {participant.filename}")
    # Read raw EEG file and visually inspect for bad channels

    raw_eeg = mne.io.read_raw_bdf(f"{DATA_DIR}/{participant.filename}",preload=True)  # Load bdf file to enable re-referencing

    #exclude bad channels from stim artificat detection, which is accross all channels
    raw_eeg.info['bads']=participant.bad_channels

    # add annotations for participant due to problems with trials
    annotations = participant.get_annotations()
    if annotations:
        raw_eeg.set_annotations(annotations)

    # band pass filter for stim artifact on artifact channels
    # raw_eeg.filter(24.5, 25.5, fir_design='firwin', picks=mne.pick_channels(raw_eeg.ch_names, ['EXG7', 'EXG8']))

    participant.load_clean_events(raw_eeg)

    if show_charts:
        mne.viz.plot_events(participant.events)

    # block annotations are useful for marking data to be ignored using BAD_
    off_events, block_annotations = get_block_events(epoch_event, participant)

    print(f"blocks extracted: \n{block_annotations}")

    off_event_id = {
        'tvns/off': 31 + 34,
        'sham/off': 32 + 34}

    print(f"pseudo off event codes {off_event_id}")

    print(f"identified {len(off_events)} off events")
    if show_charts:
        mne.viz.plot_events(off_events)

    raw_eeg.annotations.append(block_annotations[:, 0],
                               block_annotations[:, 1],
                               block_annotations[:, 2])
    if show_charts:
    # visual check stim channels (below in epochs)
        # show annotations
        # todo: this needs to be cleaned up so we can spot problems, at least a bandpass filter on it.
        raw_eeg.plot(events=off_events, event_color={65:'r',66:'y'})

    # annotations preceded by BAD epochs are dropped
    epochs = mne.Epochs(raw_eeg, off_events, event_id=off_event_id, tmin=-4, tmax=1, baseline=(None, -3.6),
               detrend=1, preload=True)

    # check stim artifacts will drop _BAD_EPOCHS
    check_stim_artifacts(epochs, participant, show_charts)

    if participant.bad_epochs:
        epochs.drop(participant.bad_epochs)
        print(f'dropped {participant.bad_epochs} epochs from configuration')

    # shift event timing to allow for latency from arduino button press
#    epochs.shift_time(participant.stim_offset)
    # having a look at onset
    epochs.shift_time(3.6-.5)

    if show_charts:
        # these alter the epoch object need to use copy() if reusing
        epochs.pick_channels(['EXG7','EXG8']).plot(events=off_events)
        # allows to check the latency of event
        epochs.pick_channels(['EXG7','EXG8']).average().plot()
#        epochs.plot(block=True,events=off_events, event_color={65:'r',66:'y'})

    return epochs

if __name__ == '__main__':
    check_all=False
    # get options, default is no charts -p6 -c also switches on charts
    argv=sys.argv[1:]
    if (len(argv)):
        CHARTS=False
        try:
            opts,args= getopt.getopt(argv,"p:c",["charts=","all"])
        except:
            print ("invalid command line arguments:")
            print ("./clean_events -- -p <participant number> [--charts=<y/n>] [--all]")
            sys.exit()
        for opt, arg in opts:
            if opt == '-h':
                print("./clean_events -- -p <participant number> [--charts=<y/n>]")
            elif opt == '--all':
                check_all=True
            elif opt == '-p':
                SUBJ = int(arg)
            elif opt in ["-c","--charts"]:
                if arg=="n":
                    CHARTS = False
                else:
                    CHARTS = True

    if check_all:
        participants = _CHECKED
    else:
        participants = [ SUBJ ]
    results = []
    for part_num in participants:
        print(f"STARTING PARTICIPANT {part_num}")
        epochs = check_participant(part_num,CHARTS)
        results.append([part_num,len(epochs['tvns/off']),len(epochs['sham/off'])])
        #print (epochs)
    print("SUMMARY")
    print("participant, tvns_epochs, sham_epochs")
    print("-------------------------------------")
    print(results)
    print("-------------------------------------")
    totals = np.sum(np.array(results), axis=0)[1:]
    print(f"{len(participants)}, {totals}")
#    print (f"epochs remaining (_BAD_EPOCHS dropped): {len(epochs)}")
 #   print(epochs)
