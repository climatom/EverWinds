#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:56:43 2020

@author: gytm3
"""

import xarray as xr, pandas as pd, numpy as np, matplotlib.pyplot as plt, datetime, GeneralFunctions as GF
from sklearn import linear_model
from matplotlib.pyplot import cm
from numba import jit
from scipy import stats
import matplotlib.dates as mdates
import GeneralFunctions as GF
import seaborn as sns
import statsmodels.api as sm
from windrose import WindroseAxes


# # Manual inspection suggests dubious data quality during this period
st=datetime.datetime(year=2019,month=12,day=27,hour=17)
stp=datetime.datetime(year=2020,month=1,day=2,hour=6)

# Should limit all data to this period
stp_gl=datetime.datetime(year=2020,month=1,day=5,hour=11)

thresh=1 # Filter out hourly-mean winds < this value

# Lead time forecast
leads=[6,12,18,24,30,36,42,48,54,60,66,72,78,84,96,102,108,114,120,126]

# Import the Forecast
dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d %H:%M:%S')
fr="/home/lunet/gytm3/Everest2019/Forecasting/NG/Logging/fclog_new_South_Col.txt"
r=pd.read_csv(fr,parse_dates=['init_date'], date_parser=dateparse,delimiter="\t")
r["valid_date"]=pd.to_datetime(r["valid_date"])
r["dT"]=[ii.total_seconds()/(3600) for ii in r["valid_date"]-r["init_date"]]
r.index=r["valid_date"]
r["ws"]=(r["u"]**2+r["v"]**2)**0.5

# Ditto observations -- incl. filtering and alocation
fo="/home/lunet/gytm3/Everest2019/AWS/Logging/south_col.csv"
o=pd.read_csv(fo,parse_dates=True,index_col=0)
for i in o.columns: o[i]=o[i].astype(np.float)
o=o.loc[o.index<=stp_gl]
u=o["WS_AVG_2"]
ug=o["WS_MAX_2"]
idx=np.logical_and(u>thresh,ug>thresh) # very minimal filter that removes wind during v. slack conditions
idx=np.logical_and(idx,~np.logical_and(u.index>=st,u.index<=stp))
u.values[~idx]=np.nan
ug.values[~idx]=np.nan

# Clip to the forecast -- only look at  the 48-hour forecast
l=120
uf_sub=r.loc[np.logical_and(r.index.isin(ug.index),r["dT"]==48)]["ws"]
ug_sub=ug.loc[u.index.isin(uf_sub.index)]
resid=ug_sub[idx]-uf_sub[idx]
uf_sub=uf_sub+np.mean(resid)
idx=np.logical_and(~np.isnan(uf_sub),~np.isnan(ug_sub))
ecdf_r=uf_sub[idx][np.argsort(uf_sub[idx])]
ecdf_g=ug_sub[idx][np.argsort(ug_sub[idx])]
x=np.arange(np.sum(idx))/np.sum(idx).astype(np.float)
fig,ax=plt.subplots(1,1)
ax.plot(ecdf_r.values[:],x,color="grey",linewidth=2)
ax.plot(ecdf_g.values[:],x,color="black",linewidth=2)
ax.set_ylim([0,1.01])
ax.set_xlim([0,55])
ax.grid()

