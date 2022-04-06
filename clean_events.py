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


# For interactive plotting, load the following backend:
plt.use('Qt5Agg')

DATA_DIR = os.path.expanduser('~') + '/win-vr/eegdata'
# these can also be passed in as command line options
CHARTS = False
SUBJ = 13

#default offset for offevent
_OFFSET=-230/1000

#### definition of participant exceptions etc ###
# list of checked participants, allows validation that each subject has been visually inspected
_CHECKED = [6,10]

# False = default block behaviour
# invalid will be ignored (bad trial), if block start events are correct then
# simpler to use annotations to exclude a bad trial and this is then shown on plots
# otherwise array length must match define all blocks that are extracted
_BAD_EVENTS_BLOCKS = {8: [False, False, 'shamblock', 'invalid', 'invalid', 'nostim', False, False, False, False]}
# can also exclude epochs using annotations
_ANNOTATIONS = {16: [['852.98828125', '51.279296875', 'BAD_no_stim']]}  # [ startime, duration, 'BAD_reason']
# bad channels, these will be interpolated
_BAD_CHANNELS = { 6: ['C6'] }


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
        if (part_num in _BAD_CHANNELS):
            self.bad_channels =_BAD_CHANNELS[part_num]
        else:
            self.bad_channels = []
        # default
        if part_num % 2:
            self.first_condition = 'shamblock'
            self.last_condition = 'tvnsblock'
        else:
            self.first_condition = 'tvnsblock'
            self.last_condition = 'shamblock'
        # if events are missing and we have more than 8 pseudo blocks, or bad trials
        self.bad_blocks = part_num in _BAD_EVENTS_BLOCKS
        if part_num in _OFFSETS:
            self.stim_offset = _OFFSETS[part_num]
        else:
            self.stim_offset = _OFFSET

    def get_block_type(self, block_number):
        # false will return the normal condition, use just to exclude an invalid block
        if self.bad_blocks and _BAD_EVENTS_BLOCKS[self.number][block_number]:
            return _BAD_EVENTS_BLOCKS[self.number][block_number]
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
    for event in participant.events:
        # use block to work out condition of stim_off event
        # could use my own blocks event to define type including ignore.
        if event[2] == event_id['tvnsblock']:
            block = 'tvnsblock'
        elif event[2] == event_id['shamblock']:
            block = 'shamblock'
        if event[2] == event_id[event_label]:
            # check if we are overiding the codes due to missing / experiment issues
            if (participant.bad_blocks):
                block =participant.get_block_type(len(blocks))
                # can use invalid or nostim or whatever you like to drop events (not a defined event type)
                if block in event_id:
                    event[2] += event_id[block]
                    off_events.append(event)
            else:
                event[2] += event_id[block]
                off_events.append(event)

            # find block boundaries for annotations etc
            if len(blocks) == 0:
                blocks.append([event[0] / sample_rate, 0, 'block_1'])
            elif event[0] > last_event[0] + block_timeout:
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

def check_stim_artifacts(epochs,show_charts = False):
    # dont alter the original
    epochs_no_stim = epochs.copy()
    # test for stimulation by rejecting all epochs with stim artifact
    reject_criteria = dict(
        eeg=500e-6,  # 500 µV
    )
    # this should drop all the epochs
    epochs_no_stim = epochs_no_stim.drop_bad(reject=reject_criteria)
    if len(epochs_no_stim.events):
        if show_charts:
            epochs_no_stim.plot()
        # probably need to warn here
        print("Did not detect the stimulation artifact in all of the epochs, annotate BAD?: " + str(
            len(epochs_no_stim.events)))
    else:
            print(f"Stim sized artifact found all in all {len(epochs)} epochs")
    return epochs_no_stim

def check_participant(part_number,show_charts,epoch_event = 'stim/off'):
    participant = Participant(part_number)

    print(f"loading participant {participant.number} data {participant.filename}")
    # Read raw EEG file and visually inspect for bad channels

    raw_eeg = mne.io.read_raw_bdf(f"{DATA_DIR}/{participant.filename}")  # Load bdf file to enable re-referencing
    # add annotations for participant due to problems with trials
    annotations = participant.get_annotations()
    if annotations:
        raw_eeg.set_annotations(annotations)

    participant.load_clean_events(raw_eeg)

    if show_charts:
        mne.viz.plot_events(participant.events)

    # block annotations are useful for marking data to be ignored using BAD_
    off_events, block_annotations = get_block_events(epoch_event, participant)

    print(f"blocks extracted: \n{block_annotations}")

    off_event_id = {
        'tvns/off': 31 + 34,
        'sham/off': 32 + 34}

    print(f"identified {len(off_events)} off events")
    if show_charts:
        mne.viz.plot_events(off_events)

    raw_eeg.annotations.append(block_annotations[:, 0],
                               block_annotations[:, 1],
                               block_annotations[:, 2])
    if show_charts:
    # visual check stim channels (below in epochs)
    #    raw_eeg.pick_channels(['EXG7', 'EXG8']).plot(events, highpass=20, lowpass=30)
        # show annotations
        raw_eeg.plot()

    # annotations preceded by BAD epochs are dropped
    epochs = mne.Epochs(raw_eeg, off_events, event_id=off_event_id, tmin=-4, tmax=1, baseline=(None, -3.6),
               detrend=1, preload=True)
    # shift event timing to allow for latency from arduino button press
    epochs.shift_time(participant.stim_offset)

    if show_charts:
        epochs.pick_channels(['EXG7','EXG8']).plot()
        # allows to check the latency of event
        epochs.pick_channels(['EXG7','EXG8']).average().plot()

    check_stim_artifacts(epochs,show_charts)

    return epochs

if __name__ == '__main__':
    # get options, default is no charts -p6 -c also switches on charts
    argv=sys.argv[1:]
    if (len(argv)):
        CHARTS=False
        try:
            opts,args= getopt.getopt(argv,"p:c",["charts="])
        except:
            print ("invalid command line arguments:")
            print ("./clean_events -- -p <participant number> [--charts=<y/n>]")
            sys.exit()
        for opt, arg in opts:
            if opt == '-h':
                print("./clean_events -- -p <participant number> [--charts=<y/n>]")
            elif opt == '-p':
                SUBJ = int(arg)
            elif opt in ["-c","--charts"]:
                if arg=="n":
                    CHARTS = False
                else:
                    CHARTS = True

    epochs = check_participant(SUBJ,CHARTS)
    print (f"good epochs: {len(epochs)}")
