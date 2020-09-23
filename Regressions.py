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
idx=np.logical_and(u>thresh,ug>thresh) # very minimal filter that removes wind during v. slack conditions
idx=np.logical_and(idx,~np.logical_and(u.index>=st,u.index<=stp))

print("Keeping %.1f%% of data"%((np.sum(idx)/np.float(len(idx)))*100))
u.values[~idx]=np.nan
ug.values[~idx]=np.nan

# Filter out the monsoon and 'sheltered' wind directions
mons=np.logical_and(ug.index.month>6,ug.index.month<10)
gap=np.logical_and(wdir>240,wdir<315)
u=u.loc[~mons]
ug=ug.loc[~mons]

# Extract overlapping reanalysis wind
gap=gap.loc[gap.index.isin(u.index)]
ur_sub=ur.loc[ur.index.isin(u.index)]
u_sub=u.loc[u.index.isin(ur.index)]
ug_sub=ug.loc[u.index.isin(ur.index)]
# and pressure
pr_sub=p.loc[p.index.isin(ur.index)]

# redeclare as x/y
x=ur_sub[gap]
y=ug_sub[gap]
idx=np.logical_and(~np.isnan(x),~np.isnan(y))

# Regression coefs
pug=np.polyfit(x[idx],y[idx],1)

# Compute gust residuals
resid_g=np.polyval(pug,x[idx])-y[idx]
ref=np.polyval(pug,np.arange(0,55))

# Assess hourly cycle
fig,ax=plt.subplots(1,1)
mug=ug_sub.groupby(ug_sub.index.hour).mean()
prug=pr_sub.groupby(pr_sub.index.hour).mean()
ax.plot(mug.index,mug.values[:])
ax2=ax.twinx()
ax2.plot(prug.index,prug.values[:])

fig,ax=plt.subplots(2,2)
fig.set_size_inches(8,8)
ax.flat[0].scatter(x[idx],y[idx],s=1,color='k')
ax.flat[0].plot(np.arange(0,55),ref,color='red')
ax.flat[1].hist(resid_g,bins=30,color='k')
ax.flat[2].scatter(x[idx],resid_g,color='k',s=1)
lags=24
rs=np.zeros(lags)
for i in range(1,lags+1): rs[i-1]=np.corrcoef(resid_g[i:],resid_g[:-i])[0,1]
ax.flat[3].stem(range(1,lags+1),rs,color='k')
for i in range(4): ax.flat[i].grid()
ax.flat[2].axhline(0,color='red')
fig.savefig("RegressionIssues.png",dpi=300)













