#!/usr/bin/env ipython
# compares p100 event between onset and offset
import glob
import matplotlib as plt
import mne
import numpy as np
from scipy import stats
import sys
import getopt


plt.use('Qt5Agg')

# timing of P3 window
TMIN=0
TMAX=1
ERP_TMIN=0.075
ERP_TMAX=0.15
TINT=.05
CHARTS = False
TOPOMAPS = False
DETREND = None
MAX = False
#CHANNELS=['CP3','CP1','CPz','CP2','CP4','P3','P1','Pz','P2','P4','PO3','POz','PO4']
#CHANNELS=['CP1','CPz','CP2','P1','Pz','P2']
CHANNELS=['Pz']
#CHANNELS=[] # all channels

def process_grand():
    print(f"PROCESS GRAND RP_TMIN={ERP_TMIN} {ERP_TMAX}")
    part_files = glob.glob("out_evoked/offset/evoked*ave.fif")
    tvns_off_evokeds = []
    sham_off_evokeds = []
    tvns_off_means= []
    sham_off_means= []
    for file in part_files:
        print(f'LOADING {file}')
        evoked = mne.read_evokeds(file)
        print(f'datalen: {len(evoked[0]._data[0])} size: {evoked[0]._size} channels: {len(evoked[0].ch_names)} filename: {file}')
        if DETREND is None:
            tvns_off_evokeds.append(evoked[0].pick_channels(CHANNELS))
            sham_off_evokeds.append(evoked[1].pick_channels(CHANNELS))
        else:
            tvns_off_evokeds.append(evoked[0].pick_channels(CHANNELS)).detrend(DETREND)
            sham_off_evokeds.append(evoked[1].pick_channels(CHANNELS)).detrend(DETREND)
        print(f'window for t-test from {ERP_TMIN},{ERP_TMAX}')
        if MAX:
            tvns_off_means.append(evoked[0].copy().crop(ERP_TMIN,ERP_TMAX).data.max())
            sham_off_means.append(evoked[1].copy().crop(ERP_TMIN,ERP_TMAX).data.max())
        else:
            tvns_off_means.append(evoked[0].copy().crop(ERP_TMIN,ERP_TMAX).data.mean())
            sham_off_means.append(evoked[1].copy().crop(ERP_TMIN,ERP_TMAX).data.mean())

    part_files = glob.glob("out_evoked/onset/evoked*ave.fif")
    tvns_on_evokeds = []
    sham_on_evokeds = []
    tvns_on_means= []
    sham_on_means= []
    for file in part_files:
        print(f'LOADING {file}')
        evoked = mne.read_evokeds(file)
        print(f'datalen: {len(evoked[0]._data[0])} size: {evoked[0]._size} channels: {len(evoked[0].ch_names)} filename: {file}')
        if DETREND is None:
            tvns_on_evokeds.append(evoked[0].pick_channels(CHANNELS))
            sham_on_evokeds.append(evoked[1].pick_channels(CHANNELS))
        else:
            tvns_on_evokeds.append(evoked[0].pick_channels(CHANNELS)).detrend(DETREND)
            sham_on_evokeds.append(evoked[1].pick_channels(CHANNELS)).detrend(DETREND)
        print(f'window for t-test from {ERP_TMIN},{ERP_TMAX}')
        if MAX:
            tvns_on_means.append(evoked[0].copy().crop(ERP_TMIN,ERP_TMAX).data.max())
            sham_on_means.append(evoked[1].copy().crop(ERP_TMIN,ERP_TMAX).data.max())
        else:
            tvns_on_means.append(evoked[0].copy().crop(ERP_TMIN,ERP_TMAX).data.mean())
            sham_on_means.append(evoked[1].copy().crop(ERP_TMIN,ERP_TMAX).data.mean())


    tvns_on_grand=mne.grand_average(tvns_on_evokeds)
    sham_on_grand=mne.grand_average(sham_on_evokeds)
    tvns_off_grand=mne.grand_average(tvns_off_evokeds)
    sham_off_grand=mne.grand_average(sham_off_evokeds)



    def format_fig(figure, window_tit,canvas_tit):
        figure.canvas.manager.set_window_title(window_tit)
        figure.suptitle(canvas_tit)
        return figure

    #plot topomaps for every 50ms
    times=np.arange(TMIN,TMAX+TINT,TINT)
    if CHARTS:
        fig=mne.viz.plot_compare_evokeds({'tvns_on':tvns_on_grand, 'tvns_off':tvns_off_grand, 'sham_on': sham_on_grand, 'sham_off':sham_off_grand},picks='eeg', combine='mean',show_sensors=True)
        format_fig(fig[0],'ERP Comparison',"Green=Sham, Red=tVNS")


    #PO sites only
    zeros = np.full((1, 21), 0)[0]
    t_test = stats.ttest_rel(tvns_on_means, tvns_off_means)
    print ('tvns on v tvns off: ',t_test)
    t_test = stats.ttest_rel(sham_off_means, sham_on_means)
    print ('sham on v sham_off: ',t_test)


if __name__ == '__main__':
    argv = sys.argv[1:]
    help_mess="./erp_tvns -c [--charts=n] [--topomaps] [--tmin=0] [--tmax=1] [--etmin=0.075] [--etmax=0.15] [--tint=.05] [--max] [--channels=Pz,Cz/all]"

    if (len(argv)):
        CHARTS = False
        try:
            opts, args = getopt.getopt(argv, "p:cf:", ["charts=",'etmin=','etmax=','tmin=','tmax=','tint=','topomaps','channels=','max'])
        except:
            print("invalid command line arguments:")
            print(help_mess)
            sys.exit()
        for opt, arg in opts:
            if opt == '-h':
                print(help_mess)
            elif opt == '-f':
                L_PASS = int(arg)
            elif opt == '--tint':
                T_INT = float(arg)
            elif opt == '--tmin':
                TMIN = float(arg)
            elif opt == '--tmax':
                TMAX = float(arg)
            elif opt == '--etmin':
                ERP_TMIN = float(arg)
            elif opt == '--etmax':
                ERP_TMAX = float(arg)
            elif opt == '--topomaps':
                TOPOMAPS = True
            elif opt == '--max':
                MAX = True
            elif opt == '--channels':
                if arg == 'all':
                    CHANNELS=[]
                else:
                    CHANNELS=arg.split(',')
            elif opt in ["-c", "--charts"]:
                if arg == "n":
                    CHARTS = False
                else:
                    CHARTS = True
    process_grand()
    print(argv)


