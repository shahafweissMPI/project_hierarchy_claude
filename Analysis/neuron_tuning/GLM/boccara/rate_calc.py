# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:43:36 2025

@author: su-weisss

https://github.com/davidespalla/code_ahv_speed_cells
TODO: save results.
"""



### to do
from numpy import isnan
import cupy as cp
#%pylab inline
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter
#Populating the interactive namespace from numpy and matplotlib
## Firing rate
spikes=pd.read_csv("CellData/spikes.csv").T
#timestamps are in seconds?
def calculateRate(spikes,totTime,binwidth,smoothwindow):
    bincenters=np.linspace(binwidth/2,totTime-binwidth/2,int(totTime/binwidth))
    spikecount=np.zeros(len(bincenters))
    rspikes=[x for x in spikes if not isnan(x)]
    for i in range(len(rspikes)):
        for j in range(len(bincenters)):
            if  ((bincenters[j]-binwidth/2) < rspikes[i] and rspikes[i] <= (bincenters[j]+binwidth/2)):
                spikecount[j]=spikecount[j]+1
    rate=spikecount/binwidth
    sigma=int(smoothwindow/binwidth)
    rate=gaussian_filter(rate,sigma)
    return rate
def calculateSpikeCount(spikes,totTime,binwidth):
    bincenters=np.linspace(binwidth/2,totTime-binwidth/2,int(totTime/binwidth))
    spikecount=np.zeros(len(bincenters))
    rspikes=[x for x in spikes if not isnan(x)]
    for i in range(len(rspikes)):
        for j in range(len(bincenters)):
            if  ((bincenters[j]-binwidth/2) < rspikes[i] and rspikes[i] <= (bincenters[j]+binwidth/2)):
                spikecount[j]=spikecount[j]+1
    return spikecount
def calculateSpikeTrain(spikes,totTime,samplerate):
    rspikes=[x for x in spikes if not isnan(x)]
    nbins=int(totTime/samplerate)
    binedges=np.linspace(0,totTime,nbins)
    spiketrain=np.zeros(nbins)
    for i in range(len(rspikes)):
        j= int(rspikes[i]/samplerate)
        if j<nbins:
            spiketrain[j]=1
    return spiketrain
totTime = 600.0
sRate = 0.02
smoothwindow = 0.25
rate=[]
for i in range(len(spikes.T)):
    rate.append(calculateRate(spikes[i],totTime,sRate,smoothwindow))
    if i%10==0:
        print("Done cell #: "+str(i))
#np.save("CellData/firingRate.npy",asarray(rate))
##2ms spike train for autocorrelogram
totTime = 600.0
sRate = 0.002 #2 ms window for autocorrelogram (see Boccara 2010, Langston 2010)
Trains=[]
for i in range(len(spikes.values)):
    Trains.append(calculateSpikeTrain(spikes.values[i],totTime,sRate))
    if i%10==0:
        print("Done cell #: "+str(i))
#np.save("CellData/spikeTrain2ms.npy",asarray(Trains))
#SHUFFLING
import random
def rshuffle(x,minshift):
    out=np.zeros(len(x))
    shift=random.randrange(minshift,len(x),1)
    for i in range(len(out)):
        newindex=(i+shift)%len(x)
        out[newindex]=x[i]
    return out
#Shuffle rates 100 times per cell
minshift=1500
nshuffle=100
rate1=rate
rateShuff=np.zeros((len(rate1.columns),nshuffle,len(rate1)))
for i in range(len(rate1.columns)):
    for j in range(nshuffle):
       # trialId=cellMetadata.loc[rate1.columns[i],:]["trialIndex"]
        x = rshuffle(rate1[rate1.columns[i]],minshift)
        rateShuff[i][j]=x
    print("Done cell "+str(i))
np.save("firingRateShuffled.npy",rateShuff)