#!python

import numpy as np
import datetime
import pandas as pd

class time_convert:
    """
    The class is for general time conversion 
    ex: year in decimal to year, month, day,....
    

    1) year2year_mon : 
       input time array need to be representing month as 
           1/24.+1./12.*np.arange(12.)
       below decimal point, and above decimal point representing 
       year. 
       output year, month, string (year-month)

        
    Class's package dependency
    1. numpy


    update history
    7/20/18 - create the class time_convert and year2year_mon

    """
    
    def __init__(self,time_array) :
        """
        Parameters:
            time: numpy array, 
                representing the time series xaxis value     
        Attributes:
            self.origin_fmt: numpy array, 
                numpy array, store the original xaxis value for future reference
        """
        self.origin_fmt=time_array
        return

    def year2year_mon(self):
        """
        Input time array in the class attribute is used 
        The time array need to be in the form of xxxx.ooooooooo  
        where xxxx = year and oooooo= any element in 1/24.+1./12.*np.arange(12.)
       
        Output year, month, string (year-month)      
        The method output variable in the form of dictionary of all three variables

        Returns: a dictionary
            year : numpy array 
                contain the year value for all element in the xaxis array 
            month : numpy array
                contain the month value for all element in the xaxis array 
            string : list of string
                contain the "year-month" string for all element in the xaxis array 

        """
        yeardate=self.origin_fmt
        mon=yeardate-np.floor(yeardate)
        year=yeardate-mon
        mon1year=1/24.+1./12.*np.arange(12.)
        month=[]
        year_month_str=[]
        i=0
        for m in mon:
            mon_ind=np.where(np.abs(mon1year-m)<1E-8)[0][0]
            month.append(mon_ind+1)
            year_month_str.append("%0.4i-%0.2i"%(year[i],month[i]))
            i+=1
        year=np.squeeze(np.array(year,dtype=int))
        month=np.squeeze(np.array(month,dtype=int))        
            
        return {'year': year,'month': month, 'string':year_month_str}
    
    def gmonth2year_mon(self):
        """
        This method is changing the GRACE month to the corresponding 
        year month in the form of datetime

        Output year, month, string (year-month)      
        The method output variable in the form of dictionary of all three variables

        Returns: a dictionary
            year : numpy array 
                contain the year value for all element in the xaxis array 
            month : numpy array
                contain the month value for all element in the xaxis array 
            string : list of string
                contain the "year-month" string for all element in the xaxis array 

        """
        gmonth=self.origin_fmt
        # GRACE month start from Jan 2002
        year=np.array(gmonth,dtype=int)/12
        mon=gmonth-year*12
        year[np.where(mon==0)]=year[np.where(mon==0)]-1
        mon[np.where(mon==0)]=12
        year=year+2002
        year_month_str=["%0.4i-%0.2i"%(year[i],mon[i]) for i in range(len(year))]
        year=np.squeeze(np.array(year,dtype=int))
        month=np.squeeze(np.array(mon,dtype=int))        

        return {'year': year,'month': month, 'string':year_month_str}    
    
    
    def tarray_month2year(self):
        """
        This method is changing the GRACE month to the corresponding 
        year month in the form of datetime

        Output year, month, string (year-month)      
        The method output variable in the form of dictionary of all three variables

        Returns: a dictionary
            year : numpy array 
                contain the year value for all element in the xaxis array 
            month : numpy array
                contain the month value for all element in the xaxis array 
            string : list of string
                contain the "year-month" string for all element in the xaxis array 

        """
        tarray=self.origin_fmt
        yeardate=(np.array(tarray.dt.month)-1)/12.+1/24.+np.array(tarray.dt.year)

        return yeardate
    
    

def tarray_month(itime,ftime):
    """
    Parameters:
        itime: list 
            in the order of [year,month]
        ftime: list 
            in the order of [year,month]
    Returns:
        time : pd.indexes.timeindex
            time index in the form of pandas time index array

    """
    itime=np.array(itime,dtype=int)
    ftime=np.array(ftime,dtype=int)
    ini = datetime.datetime(itime[0],itime[1],1)
    fin = datetime.datetime(ftime[0],ftime[1],1)
    nmonth=((ftime[0]-1)-itime[0])*12+(12-itime[1]+1)+ftime[1]
    # 1st each month will be the output date
    time = pd.date_range(ini, freq='MS', periods=nmonth)
    # change array from 1st to 15th in each month 
    dtime=datetime.timedelta(days=14)
    time=time+dtime

    return time


    
    
    