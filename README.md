# tvns-mne
MNE Python implementation to analyse the effects of taVNS on ERPs, using the paradigm of Sharon et al. (2020).

## Scripts

MNE uses interactive python. Therefore, scripts require -- prior to command line arguments. This is easy to miss and correct usage is shown in the examples below. For help / the full list of with command line arguments use ``./scriptname.py -- -h``

### 1. clean_events.py
This script is used for identifying issues at the participant level. It should be called with charts enabled (-c). It shows plots of all channels for visual inspection. Bad channels should be added to _BAD_CHANNELS dictionary which is keyed on participant.

Plots also allow the identification of trials with no stim artifact. These should be added to the _BAD_EPOCHS dictionary. 

It also possible to identify periods where trials are excluded, for example for reasons in the experimental notes. For example where the stimulation electrode was not applied in the correct location. These can be defined using annotations in _ANNOTAIONS. MNE will reject trials that overlap an annotation with a name that starts with BAD_.

Clean_events can also be run globally using the --all switch. This is useful as it will provide a count of valid epochs by condition.  

In the below example clean_events is run to inspect / process the first participant:

``./clean_events.py -- -c -p1``

### 2. eeg_tvns.py
Runs the main preprocessing routing and outputs 5s epochs to the out_epochs directory. It uses the dictionaries in clean_events to reject invalid trials / interpolate bad channels. Clean_events is included as module.

In the below example it is being run just for the first participant. Detrending is set to 1, which means the 5s epochs will be linearly detrended:

```./eeg_tvns.py -- -p1 -c --detrend=1```

### 3. evoke_tvns.py

Reads the files in out_epochs writes evoked files to out_evoked. Epochs can be time shifted so that a different event is at zero. High and low pass filters can also be applied. 

It can be run for single participants and all participants. It can be used to compare average waveforms for each participant. As well as charts, it can also generate topmaps with an interval defined by the tint parameter.

The below example creats evoked waveforms for all partipants centered around the onset event (-.2 to 1s). An 8Hz low pass filter is applied to remove alpha activity:

``./evoke_tvns.py -- --all -c --tshift=3.42 --tmin=-.2 --tmax=1 --lpass=8``

### 4. erp_tvns.py
Reads the files in out_evokes and combines them into a grand average. Performs a number of comparison based on the mean value in the event window. It can also use the max value if the --max switch is added. It can provide charts and topomaps based on all participants.

The below example will perform a comparisons of mean amplitude for Pz from 50 to 150ms:

``./erp_tvns.py -- --etmin=.05 --etmax=.15 --channels=Pz``

### 5. tvns_power.py
This script performs band power comparisons for the specified window, between conditions. 

The below example compates band power for all participants during stimulation:

``./tvns_power.py -- --all --tmin=-3.42 --tmax=0 ``

### 6. erp_events.py
This script was written to compare erp magnitudes for the same condition but at different timepoints.

The below example compares max amplitude at onset and offset:

``./erp_events.py -- --max``
