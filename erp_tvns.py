#!/usr/bin/env ipython
# aggregates accross all subjects to analyse erp P300 hopefully
import glob
import matplotlib as plt
import mne
import numpy as np
from scipy import stats

plt.use('Qt5Agg')

# timing of P3 window
TMIN=0
TMAX=1
ERP_TMIN=.3
ERP_TMAX=.5
#CHANNELS=['CP3','CP1','CPz','CP2','CP4','P3','P1','Pz','P2','P4','PO3','POz','PO4']
CHANNELS=['CP1','CPz','CP2','P1','Pz','P2']
#CHANNELS=['Pz']
#CHANNELS=[] # all channels

def process_grand():
    part_files = glob.glob("out_evoked/evoked*ave.fif")
    tvns_evokeds = []
    sham_evokeds = []
    tvns_means= []
    sham_means= []
    for file in part_files:
        print(f'LOADING {file}')
        evoked = mne.read_evokeds(file)
        print(f'datalen: {len(evoked[0]._data[0])} size: {evoked[0]._size} channels: {len(evoked[0].ch_names)} filename: {file}')

        tvns_evokeds.append(evoked[0].pick_channels(CHANNELS))#.detrend(1))
        sham_evokeds.append(evoked[1].pick_channels(CHANNELS))#.detrend(1))
        tvns_means.append(evoked[0].copy().crop(ERP_TMIN,ERP_TMAX).data.mean())
        sham_means.append(evoked[1].copy().crop(ERP_TMIN,ERP_TMAX).data.mean())


    tvns_grand=mne.grand_average(tvns_evokeds)
    sham_grand=mne.grand_average(sham_evokeds)

    def format_fig(figure, window_tit,canvas_tit):
        figure.canvas.manager.set_window_title(window_tit)
        figure.suptitle(canvas_tit)
        return figure

    #plot topomaps for every 50ms
    times=np.arange(TMIN,TMAX+.05,.05)
    format_fig(sham_grand.plot_topomap(times),
               'Sham Topomap by Time',
               'SHAM Evoked')
    format_fig(tvns_grand.plot_topomap(times),
               'tVNS Topomap by Time',
               'tVNS Evoked')

    #fig, anim = tvns_grand.animate_topomap(
    #    times=times, ch_type='eeg', frame_rate=2, time_unit='s', blit=False)


    fig=mne.viz.plot_compare_evokeds([tvns_grand, sham_grand],picks='eeg', combine='mean',show_sensors=True)
    format_fig(fig[0],'ERP Comparison',"Orange=Sham, Blue=tVNS")


    #PO sites only
    t_test = stats.ttest_rel(tvns_means,sham_means,alternative='greater')
    print("dependent t-test, tvns > sham:", t_test)

if __name__ == '__main__':
    process_grand()
