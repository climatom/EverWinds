#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script interpolates the reanalysis wind data to the elevation of the 
South Col. Note that it has *already* been interpolated to the lat/lon
of the South Col
"""

import xarray as xa, numpy as np, matplotlib.pyplot as plt 
from numba import jit


# #Functions
@jit
def vint(z,v):
    n=v.shape[0]
    out=np.zeros((n))*np.nan
    zi=z.values/g
    for i in range(n):
        x=zi[i,:,0,0][::-1] # Needs to be flipped so that increasing
        y=v.values[i,:,0,0][::-1] # Ditto
        out[i]=np.interp(col_z,x,y)
        
    return out

@jit
def vint_press(z):
    n=z.shape[0]
    out=np.zeros((n))*np.nan
    zi=z.values/g
    y=z.level.values[:][::-1]
    for i in range(n):
        x=zi[i,:,0,0][::-1] # Needs to be flipped so that increasing
        out[i]=np.interp(col_z,x,y)
        
    return out

## Constants
g=9.81

## Params
# File
di="/home/lunet/gytm3/Everest2019/Research/Weather/Data/"
fi=di+"merged.nc"
# South Col elevation
col_z=7945.


## Main
# Open it
d=xa.open_dataset(fi)

# Compute wind
ws=np.sqrt(d.u**2+d.v**2)

# Interpolate to the *elevation* of the South Col
ws_col=\
xa.DataArray(vint(d.z,ws),dims=["time"],coords={"time":ws.time.values[:]},\
             name="ws")

# Ditto temperature
temp_col=\
xa.DataArray(vint(d.z,d.t),dims=["time"],coords={"time":ws.time.values[:]},\
             name="temp")

# Ditto ozone
o3_col=\
xa.DataArray(vint(d.z,d.o3),dims=["time"],coords={"time":ws.time.values[:]},\
             name="o3")

# Ditto p
p_col=\
xa.DataArray(vint_press(d.z),dims=["time"],coords={"time":ws.time.values[:]},\
             name="press")

# Ditto u
u_col=\
xa.DataArray(vint(d.z,d.u),dims=["time"],coords={"time":ws.time.values[:]},\
             name="u_wnd")

# Ditto v
v_col=\
xa.DataArray(vint(d.z,d.v),dims=["time"],coords={"time":ws.time.values[:]},\
             name="v_wnd")

# In dataset
out=xa.merge([ws_col,temp_col,o3_col,p_col,u_col,v_col])

# Add meta
us=["m s**-1","degC","kg kg**-1","hPa","m s**-1","m s**-1"]
name=["ws","temp","o3","press","u_wnd","v_wnd"]
for i in range(len(name)): out[name[i]].attrs["units"]=us[i]
out.to_netcdf(di+"SouthCol_interpolated.nc")


