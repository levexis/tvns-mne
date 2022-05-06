#!/usr/bin/env ipython
# aggregates accross all subjects to analyse erp P300 hopefully
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
ERP_TMIN=.3
ERP_TMAX=.5
TINT=.05
CHARTS = False
TOPOMAPS = True
DETREND = None
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
        if DETREND is None:
            tvns_evokeds.append(evoked[0].pick_channels(CHANNELS))
            sham_evokeds.append(evoked[1].pick_channels(CHANNELS))
        else:
            tvns_evokeds.append(evoked[0].pick_channels(CHANNELS)).detrend(DETREND)
            sham_evokeds.append(evoked[1].pick_channels(CHANNELS)).detrend(DETREND)
        tvns_means.append(evoked[0].copy().crop(ERP_TMIN,ERP_TMAX).data.mean())
        sham_means.append(evoked[1].copy().crop(ERP_TMIN,ERP_TMAX).data.mean())


    tvns_grand=mne.grand_average(tvns_evokeds)
    sham_grand=mne.grand_average(sham_evokeds)

    def format_fig(figure, window_tit,canvas_tit):
        figure.canvas.manager.set_window_title(window_tit)
        figure.suptitle(canvas_tit)
        return figure

    #plot topomaps for every 50ms
    times=np.arange(TMIN,TMAX+TINT,TINT)
    if CHARTS:
        if TOPOMAPS:
            format_fig(sham_grand.plot_topomap(times),
                       'Sham Topomap by Time',
                       'SHAM Evoked')
            format_fig(tvns_grand.plot_topomap(times),
                       'tVNS Topomap by Time',
                       'tVNS Evoked')
        fig=mne.viz.plot_compare_evokeds([tvns_grand, sham_grand],picks='eeg', combine='mean',show_sensors=True)
        format_fig(fig[0],'ERP Comparison',"Orange=Sham, Blue=tVNS")


    #PO sites only
    zeros = np.full((1, 21), 0)[0]
    t_test = stats.ttest_rel(tvns_means, zeros)
    print ('tvns means v zeros: ',t_test)
    t_test = stats.ttest_rel(sham_means, zeros)
    print ('sham means v zeros: ',t_test)

    t_test = stats.ttest_rel(tvns_means,sham_means,alternative='greater')
    print("dependent t-test, tvns > sham:", t_test)


if __name__ == '__main__':
    argv = sys.argv[1:]
    help_mess="./erp_tvns -c [--charts=n] [--topomaps] [--tmin=0.3] [--tmax=0.5] [--tint=.05] [--channels=Pz,Cz/all]"

    if (len(argv)):
        CHARTS = False
        try:
            opts, args = getopt.getopt(argv, "p:cf:", ["charts=",'tmin=','tmax=','tint=','topomaps','channels='])
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
                T_MIN = float(arg)
            elif opt == '--tmax':
                T_MAX = float(arg)
            elif opt == '--topomaps':
                TOPOMAPS = True
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


