#!python
import netCDF4 as nc4
import numpy as np
import xarray as xr
from collections import OrderedDict

class ncdf_io :
    """
    The class is derived from the module.netcdf_io
    The methods are updated base on the functions 
    to become more generic. 

    1) read_ncdf : 
        use only netCDF4 to output 
        1. var 
        2. dim
        in the form of dictionaries

    2) read_ncdf2xarray :
        use the read_ncdf method and transform 
        the dictionaries to Xarrary.Dataset
        for easy application. It outputs
        ds : a Xarray.Dataset
        
    Class's package dependency
    1. numpy
    2. netCDF4
    3. xarray
    4. collections

    update history
    7/18/18 - create the file include read_ncdf and read_ncdf2xarray
    7/19/18 - create the class ncdf_io and two methods
    8/03/18 - change the meaning of dimorder to read some ncdf file 
              that has dimorder different from dimnesion order in the data

    """
    
    def __init__ (self,filename,verbose=1):
        """
        Parameters:
            filename: string, 
                must input file name and path for the IO file      
            verbose: np.int, kwarg
                must input integer to show detail of the netcdf header
                0 for no output
                1 for output dim, var name and size 
        Attributes:
            self.path: string, 
                the IO file location
        """

        self.path=filename
        self.verbose=verbose
        
    def read_ncdf(self,var=[],dimorder=[],ncformat='NETCDF4_CLASSIC'):
        """
        The method output variable in the form of dictionary

        Returns:
            dim_dict : 
                contain all dimensions in the ncfile
            var_dict : 
                contain all variables (or listed variable) in the ncfile
            
        kwargs:
            var : list of strings, optional
                list of the variable one want to output
                if the list is not provided all variables
                will be output.

            dimorder: list of strings, optional
                list of the dimension ordered 
                ex:['lon','lat','time'] when the data dimension
                order is not as the input dimension order

            ncformat: string, optional
                put the netcdf file format. i.e. NETCDF4_CLASSIC,
                NETCDF3_CLASSIC, NETCDF4, NETCDF3, etc...
                more format can be found 
                http://unidata.github.io/netcdf4-python/

        """

        # open the netcdf file
        rootgrp = nc4.Dataset(self.path,mode='r',format=ncformat) 

        # dump variable name and dimension name
        dump_var=rootgrp.variables
        varname = [v[0] for v in dump_var.items()]
        dump_dim=rootgrp.dimensions
        dimname = [v[0] for v in dump_dim.items()]
        
        # reorder the var if dimorder is given
        ndim=len(dimname)
        indexorder=[]
        if dimorder != []:
            if self.verbose > 0:
                print '\n'
                print "!!!!! Variable dimension is reodered !!!!!"
                print " Original order       :", dimname
                print " Base on user setting :", dimorder
            for i in range(len(dimorder)):
                for j in range(len(dimname)):
                    if dimorder[i] in dimname[j]:
                        indexorder.append(j)
        else :
            indexorder=range(ndim)
        
        # read in dimensions 
        if self.verbose > 0:
            print '\n'
            print '#   ncfile dimensions '
            print '----------------------------------'
        dim_dict = OrderedDict() # to keep the order of the dimension listed in the dict as set
        if self.verbose > 0:
            for i in indexorder:
                dim_dict[dimname[i]]=np.array(rootgrp.variables[dimname[i]][:])
                print dimname[i],len(dim_dict[dimname[i]])
        else:
            for i in indexorder:
                dim_dict[dimname[i]]=np.array(rootgrp.variables[dimname[i]][:])
                
        # read in variables
        if self.verbose > 0:
            print '\n'
            print '#   ncfile variables '
            print '----------------------------------'
        nvar=len(varname)
        var_dict={}

        # loop through all variable and dim
        for i in range(nvar):

            # if the varname is not listed all variable will be output
            if var == [] :
                # exclude dimension 
                if varname[i] not in dimname :
                    var_dict[varname[i]]=np.array(rootgrp.variables[varname[i]][:])
                    #var_dict[varname[i]]=np.transpose(var_dict[varname[i]],indexorder)
                    if self.verbose > 0:
                        print varname[i],var_dict[varname[i]].shape
                        print "    dim in set/default order:"
                        newdim=[dimname[ind] for ind in indexorder]
                        print "    ",newdim  
                        
            # output varname listed in var
            else :
                # exclude dimension and only include varname in the list
                if varname[i] not in dimname and varname[i] in var:
                    var_dict[varname[i]] = np.array( rootgrp.variables[varname[i]][:] )
                    #var_dict[varname[i]] = np.transpose( var_dict[varname[i]],indexorder )
                    if self.verbose > 0:
                        print varname[i],var_dict[varname[i]].shape
                        print "    dim in set/default order:"
                        newdim=[dimname[ind] for ind in indexorder]
                        print "    ",newdim

        return dim_dict, var_dict

    
    
    def read_ncdf2xarray(self,var=[],dimorder=[],ncformat='NETCDF4_CLASSIC'):
        """
        The method output variable in the form of Xarray.Dataset

        Parameters:
            filename: string
                file name and path to the file

        Returns:
            ds : xarray.Dataset
                xarray.Dataset includes all variable in the netcdf file 
                if varname is not set.
            
        kwargs:
            var : list of strings, optional
                list of the variable one want to output
                if the list is not provided all variables
                will be output.

            dimorder: list of strings, optional
                list of the dimension ordered 
                ex:['lon','lat','time'] when the data dimension
                order is not as the input dimension order

            ncformat: string, optional
                put the netcdf file format. i.e. NETCDF4_CLASSIC,
                NETCDF3_CLASSIC, NETCDF4, NETCDF3, etc...
                more format can be found 
                http://unidata.github.io/netcdf4-python/

        """

        dim_dict, var_dict = self.read_ncdf(var=var,dimorder=dimorder,ncformat=ncformat)
        nvar=len(var_dict.items())
        ndim=len(dim_dict.items())

        # dump variable name and dimension name
        varname = [v[0] for v in var_dict.items()]
        dimname = [v[0] for v in dim_dict.items()]
        
        # order the dimension based on name
        
        
        # store each variable as a var
        dict1={}
        for i in np.arange(nvar):
            dict1[varname[i]]=(dimname,var_dict[varname[i]])
        ds=xr.Dataset(dict1,coords=dim_dict)    
        
        return ds        
        



