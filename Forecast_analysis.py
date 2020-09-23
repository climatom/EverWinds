#!/usr/bin/env python
# coding: utf-8

# # Introduction¶
# < Place holder >

# In[105]:



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


# In[106]:


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




# In[107]:


# Now clip forecast to obs and plot
fig,ax=plt.subplots(5,4)
fig.set_size_inches(9,10)
x=np.linspace(0,60)
count=0
query=20.
k=30
for lead in leads:
    uf_sub=r.loc[np.logical_and(r.index.isin(ug.index),r["dT"]==lead)]["ws"]
    #uf_sub=r.loc[np.logical_and(r.index.isin(ug.index),r["dT"]<=lead)]["ws"]
    ug_sub=ug.loc[u.index.isin(uf_sub.index)]
    idx=np.logical_and(~np.isnan(uf_sub),~np.isnan(ug_sub))
    rug=np.corrcoef(uf_sub[idx],ug_sub[idx])[0,1]
    ax.flat[count].scatter(uf_sub,ug_sub,s=1,color='k')
    ax.flat[count].grid()
    ax.flat[count].plot(x,x,color='red',linestyle="--")
    ax.flat[count].set_xlim([0,55])
    ax.flat[count].set_ylim([0,55])
    ax.flat[count].text(2,47,"Lead = %.0f h"%lead,fontsize=8)
    ax.flat[count].text(20,9,"r = %.2f"%rug,fontsize=8)
    slope,intercept=np.polyfit(uf_sub[idx],ug_sub[idx],1)
    y=np.polyval([slope,intercept],x)
    ax.flat[count].plot(x,y,color='yellow')
    err_raw=np.mean(np.abs(ug_sub[idx]-uf_sub[idx]))
    err=np.mean(np.abs(ug_sub[idx]-np.polyval([slope,intercept],uf_sub[idx])))
    ax.flat[count].text(28,22,r"$\alpha$ = %.2f"%intercept,fontsize=8)
    ax.flat[count].text(28,15,r"$\beta$ = %.2f"%slope,fontsize=8)
    ax.flat[count].text(12,2,"MAE = %.1f [%.1f] m/s"%(err,err_raw),fontsize=8)
    a,b,c,d,e=GF.pred_intervals(uf_sub[idx],ug_sub[idx],query)
    pk=1-stats.norm.cdf(k,loc=query*slope+intercept,scale=e)
    print("For lead: %.0f, prob of exceeding %.1f m/s when fc is %.1f is %.1f%%"%(lead,k,query,pk*100))
    print("For lead: %.0f, GFS speed = %.1f m/s when obs are %.1f%%"%(lead,(k-intercept)/slope,k))
    count+=1
fig.text(0.5, 0.05, 'GFS Wind Speed (m/s)', ha='center')
fig.text(0.04, 0.5, 'Observed Wind Speed (m/s)', va='center', rotation='vertical')
fig.savefig("/home/lunet/gytm3/Everest2019/Research/Weather/Figures/GFS_perf.pdf",dpi=300)


# In[108]:


pk


# In[111]:


# Now clip forecast to obs and plot
fig,ax=plt.subplots(2,3)
fig.set_size_inches(11,8)
x=np.linspace(0,60)
count=0
leads=[24,48,72,96,120,144]
queers=np.arange(5,40,1)
ks=np.arange(15,40,1)
out=np.zeros((len(ks),len(queers)))*np.nan
levs=np.linspace(1,99,30)
i=0
for lead in leads:
    uf_sub=r.loc[np.logical_and(r.index.isin(ug.index),r["dT"]==lead)]["ws"]
    ug_sub=ug.loc[u.index.isin(uf_sub.index)]
    idx=np.logical_and(~np.isnan(uf_sub),~np.isnan(ug_sub))
    ki=0
    for k in ks:
        qi=0
        for queer in queers:
            idx=np.logical_and(~np.isnan(uf_sub),~np.isnan(ug_sub))
            slope,intercept=np.polyfit(uf_sub[idx],ug_sub[idx],1)
            a,b,c,d,e=GF.pred_intervals(uf_sub[idx],ug_sub[idx],query)
            out[ki,qi]=(1-stats.norm.cdf(k,loc=queer*slope+intercept,scale=e))*100.
            qi+=1
        ki+=1
    cm=ax.flat[i].contourf(queers,ks,out,levels=levs,cmap="coolwarm")
    ax.flat[i].contour(queers,ks,out,levels=[5,10,50],colors=["k",])
    ax.flat[i].text(28,18,"Lead = %.0f h"%lead,fontsize=8)
    ax.flat[i].grid()
    i+=1
plt.subplots_adjust(right=0.9)
fig.text(0.5, 0.05, 'GFS Wind Speed (fc; m/s)', ha='center')
fig.text(0.05, 0.5, 'Threshold Wind Speed (k; m/s)', va='center', rotation='vertical')

cbar_pos=[0.92,0.12,0.025,0.76]
cax=fig.add_axes(cbar_pos)
cbar=plt.colorbar(cm,cax=cax,orientation="vertical")
cbar.set_ticks([10,20,30,40,50,60,70,80,90])
cbar.set_label("$p(g > k  |  fc)$ [%]")
fig.savefig("/home/lunet/gytm3/Everest2019/Research/Weather/Figures/GFS_probs.pdf",dpi=300)


