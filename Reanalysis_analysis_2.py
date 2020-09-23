#!/usr/bin/env python
# coding: utf-8

import xarray as xr, pandas as pd, numpy as np, matplotlib.pyplot as plt, \
datetime, GeneralFunctions as GF
from sklearn import linear_model
from matplotlib.pyplot import cm
from numba import jit
from scipy import stats
import matplotlib.dates as mdates
import GeneralFunctions as GF
import seaborn as sns
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import summary_table

# Circular statistics 
def circ_stat(series,wdir,span,stat,res):
    
    """
    Info to follow 
    """
    
    # Radians
    rad=np.radians(wdir)
    thresh=np.radians(span)
    circ=2*np.pi
    
    # Associate function - making sure it's 'nan-compatible'
    stat="nan"+stat
    f = getattr(np,stat)
    
    # preallocate
    #out=np.zeros(len(series))*np.nan
    uang=np.radians(np.arange(1,360,res))
    out=np.zeros(len(uang)+1)
    count=0
    for ii in uang:
        delta=np.min(np.column_stack((circ-abs(rad-ii),np.abs(rad-ii))),axis=1)
        logi=delta<=thresh
        out[count]=f(series[logi])
        count+=1
        
    uang=np.degrees(uang)
    uang_out=np.zeros(len(uang)+1); uang_out[:-1]=uang; uang_out[-1]=uang[0]
    out[-1]=out[0]
    return uang_out,out


def circ_correl(series1,series2,wdir,span,res):
    
    """
    Info to follow 
    """
    
    # Radians
    rad=np.radians(wdir)
    thresh=np.radians(span)
    circ=2*np.pi
    
    # preallocate
    uang=np.radians(np.arange(1,360,res))
    out=np.zeros(len(uang)+1)
    nanidx=np.logical_and(~np.isnan(series1),~np.isnan(series2))
    count=0
    for ii in uang:
        delta=np.min(np.column_stack((circ-abs(rad-ii),np.abs(rad-ii))),axis=1)
        #delta=np.array([ circ - np.min([circ-abs(jj-ii),abs(jj-ii)]) for jj in rad])
        logi=delta<=thresh
        logi=np.logical_and(logi,nanidx)
        out[count]=np.corrcoef(series1[logi],series2[logi])[0,1]
        count+=1
        
    uang=np.degrees(uang)
    uang_out=np.zeros(len(uang)+1); uang_out[:-1]=uang; uang_out[-1]=uang[0]
    out[-1]=out[0]
    return uang_out,out
            

# # Manual inspection suggests dubious data quality during this period
st=datetime.datetime(year=2019,month=12,day=27,hour=17)
stp=datetime.datetime(year=2020,month=1,day=2,hour=6)

# Should limit all data to this period
stp_gl=datetime.datetime(year=2020,month=1,day=5,hour=11)

thresh=1 # Filter out hourly-mean winds < this value

# Import the reanalysis
fr="/home/lunet/gytm3/Everest2019/Research/Weather/Data/SouthCol_interpolated.nc"
r=xr.open_dataset(fr).to_dataframe()
ur=r["ws"]
u_wnd=r["u_wnd"]
v_wnd=r["v_wnd"]
tr=r["temp"]-273.15
pr=r["press"]#
# Wdir from the reanalysis
wdir=GF.calc_wdir(u_wnd,v_wnd)

# Ditto observations -- incl. filtering and alocation
fo="/home/lunet/gytm3/Everest2019/AWS/Logging/south_col.csv"
o=pd.read_csv(fo,parse_dates=True,index_col=0)
for i in o.columns: o[i]=o[i].astype(np.float)
o=o.loc[o.index<=stp_gl]
u=o["WS_AVG_2"]
ug=o["WS_MAX_2"]
t=o["T_HMP"]
p=o["PRESS"]
idx=np.logical_and(u>thresh,ug>thresh) # very minimal filter that removes 
# wind during v. slack conditions
idx=np.logical_and(idx,~np.logical_and(u.index>=st,u.index<=stp))
u.values[~idx]=np.nan
ug.values[~idx]=np.nan
# Read in the horizon angles 
hang=np.loadtxt("/home/lunet/gytm3/Everest2019/Research/Weather/Data/horz.txt")
hang_plot=np.zeros((len(hang)+1,2)); 
hang_plot[:-1,:]=hang[:,:]
hang_plot[-1,:]=hang[0,:]

# Extract overlapping reanalysis wind
ur_sub=ur.loc[ur.index.isin(u.index)]
wdir_sub=wdir.loc[ur.index.isin(u.index)]
u_sub=u.loc[u.index.isin(ur.index)]
ug_sub=ug.loc[u.index.isin(ur.index)]
# and pressure
pr_sub=p.loc[p.index.isin(ur.index)]

# Compute correlations
idx=np.logical_and(~np.isnan(ur_sub),~np.isnan(ug_sub))
glob_r=np.corrcoef(ur_sub[idx],ug_sub[idx])
rcirc=circ_correl(ug_sub,ur_sub,wdir_sub,15.0,5.)

# Ratio 
rat=ug_sub/ur_sub
rrat=circ_stat(rat,wdir_sub,15,"median",5.)

# Residual
#idx=np.logical_and(idx,np.logical_and(wdir_sub>225,wdir_sub<315))
resid=ug_sub[idx]-ur_sub[idx]
# Bias correct?
ur_sub=ur_sub+np.mean(resid)
resid=ug_sub[idx]-ur_sub[idx]

# Compute frequency
freq=circ_stat(np.ones(len(wdir_sub)),wdir_sub,45,"sum",45)
freq_pc=freq[1]/np.sum(freq[1])*100

# Draw figures
fig=plt.figure()
fig.set_size_inches(7,6)
ref=np.linspace(0,55,100)
ax1=fig.add_subplot(222)
ax1.scatter(ur_sub,ug_sub,s=1,color='k')
ax1.plot(ref,ref,color='red')
ax1.grid()
ax1.set_xlim([0,55])
ax1.set_ylim([0,55])

# Plot horizon angle
ax2=fig.add_subplot(221,projection="polar")
ax2.plot(np.radians(hang_plot[:,0]),hang_plot[:,1],color='k')
ax2.set_theta_direction(-1)
ax2.set_theta_direction(-1)
ax2.set_theta_zero_location("N")
#ax2.set_ylim([0,1.0])

# Plot the circular histogram?
ax2.bar(np.radians(freq[0]),freq_pc,color="grey",linewidth=0.5,edgecolor="red")

# Plot correlation
ax3=fig.add_subplot(223,projection="polar")
ax3.plot(np.radians(rcirc[0]),rcirc[1],color='k')
# PLot ratio
#ax2.plot(np.radians(rrat[0]),rrat[1])
ax3.set_theta_direction(-1)
ax3.set_theta_direction(-1)
ax3.set_theta_zero_location("N")
ax3.set_ylim([0,1.0])
#fig.set_size_inches(4,4)

ax4=fig.add_subplot(224)
ax4.hist(resid,bins=30,facecolor='w',edgecolor='k')
ax4.grid()
ax4.set_xlim([-12,12])
ax4.axvline(0,linestyle="--",color="red")














