#!python
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from ncdf_io import ncdf_io
from time_convert import time_convert
from matplotlib import rcParams,rc
from Lambert_frame_tick import *
from regression import regress
from sinusoidal import feature
from distribution import ci2nstd
from time_convert import convert_ds_gmonth_tarray


"""
Make sure to have the JPL mascon ocean file (run the JPL processing first if the file does not exist) 
before running this script

"""


# input file
dir3='/Users/joedhsu/Research/Rsync/Data_p/GRACE05_CSR/004_154/Global/'
file3='slf.landf.t60.s300.m.deg1.scaCLM4.5bgc_gict_hydt_ind_decom_multi_MB_decom_multi.IVANS_IJ05_IMBIE_GERUO_ICE5_COMP.GAC.t100.ptcorr.rf.com004_154'
ncslf=ncdf_io(dir3+file3+'.nc')

# read file
ncslf.verbose=0
ds_slf=ncslf.read_ncdf2xarray()

# convert time stamp
ds_slf=convert_ds_gmonth_tarray(ds_slf, offset=1)


# ds_slf['z'].sel(time=[datetime.datetime(2003,10,15),datetime.datetime(2004,10,15),datetime.datetime(2005,10,15)
#                   ,datetime.datetime(2009,10,15),datetime.datetime(2010,10,15),datetime.datetime(2011,10,15)
#                   ,datetime.datetime(2012,10,15),datetime.datetime(2013,10,15),datetime.datetime(2014,10,15)],
#                 method='nearest').plot.contourf(col='time',col_wrap=3,levels=np.linspace(-1,1,10))

# output dataset
ds_slf.to_netcdf('./data/slf.nc')
ds_new=xr.open_dataset('./data/jpl_mascon_ocean.nc')


# interpolation
#  kwargs set to fill_value= None is forcing the function to 
#  extrapolate the value where indicated location of interpolation 
#  is out of the original data
ds_slf_interp=ds_slf.interp(coords={'lon':ds_new.lon,'lat':ds_new.lat}
                            ,method='linear',kwargs={'fill_value':None}) 

# slf correction in obp
ds_corr=ds_new-ds_slf_interp

# output dataset
ds_corr.to_netcdf('./data/jpl_mascon_ocean-slf.nc')


# perform regression
taxis=time_convert(ds_corr['z'].time).tarray_month2year()
reg=regress(taxis)
reg.multivar_regress(ds_corr['z'].isel(lon=0,lat=0).values,predef_var='semisea_sea_lin')
dimname=reg.dm_order

beta=np.zeros([ds_corr['z'].lon.shape[0],ds_corr['z'].lat.shape[0],len(dimname)])+np.nan
se=np.zeros([ds_corr['z'].lon.shape[0],ds_corr['z'].lat.shape[0],len(dimname)])+np.nan


# import time
#
# start = time.time()
ii=0
for i in ds_corr['z'].lon.values :
    jj=0
    for j in ds_corr['z'].lat.values :
        if ~ds_corr['z'].sel(lon=i,lat=j).sum(skipna=False).isnull():
            dict1=reg.multivar_regress(ds_corr['z'].sel(lon=i,lat=j).values,
                                       predef_var='semisea_sea_lin')
            beta[ii,jj]=dict1['beta']
            se[ii,jj]=dict1['se']
        jj+=1
    ii+=1
# end = time.time()
# print(end - start)

xr_beta=xr.DataArray(beta, coords=[ds_corr['z'].lon,ds_corr['z'].lat,dimname],
                             dims=['lon','lat','reg_variate'])
xr_se=xr.DataArray(se, coords=[ds_corr['z'].lon,ds_corr['z'].lat,dimname],
                             dims=['lon','lat','reg_variate'])


ds_corr_beta=xr.Dataset()
ds_corr_beta['beta']=xr_beta.T
ds_corr_beta['se']=xr_se.T


# calculate the annual signal
from sinusoidal import feature
annsig=feature()
dict2=annsig.amp_phase(ds_corr_beta['beta'].sel(reg_variate='anncos').values,
                        ds_corr_beta['beta'].sel(reg_variate='annsin').values,
                        np.pi*2.,
                        error=True,
                        ampcos_error = ds_corr_beta['se'].sel(reg_variate='anncos').values,
                        ampsin_error = ds_corr_beta['se'].sel(reg_variate='annsin').values)


# ds_ann=xr.Dataset()
ds_corr_beta['annamp']=xr.DataArray(dict2['amp'], coords=[ds_corr['z'].lat,ds_corr['z'].lon],
                             dims=['lat','lon'])
ds_corr_beta['annphase']=xr.DataArray(dict2['phase'], coords=[ds_corr['z'].lat,ds_corr['z'].lon],
                             dims=['lat','lon'])
ds_corr_beta['annamperr']=xr.DataArray(dict2['amperr'], coords=[ds_corr['z'].lat,ds_corr['z'].lon],
                             dims=['lat','lon'])
ds_corr_beta['annphaseerr']=xr.DataArray(dict2['phaseerr'], coords=[ds_corr['z'].lat,ds_corr['z'].lon],
                             dims=['lat','lon'])


# regression dataset
ds_corr_beta.to_netcdf('./data/jpl_mascon_ocean-slf.reg.nc')
