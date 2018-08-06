#!python

import numpy as np 


class feature:
    """
    Calculate the feature of the sinusoidal signal

    """  
    
    
    def amp_phase(self, ampcos, ampsin, omega, ampcos_error=-1 ,ampsin_error=-1):
        """
        The method calculate the amplitude and phase (maximum value of the sinusoidal signal)
        
        Parameters:
            ampcos : np.float,
                regression coeffcient for the cosine wave
                
            ampsin : np.float,
                regression coeffcient for the sine wave 
                
            omega : np.float,
                phase speed (omega) of the cos & sin wave  
                ex: annual sin/cos wave of np.cos(year*2.*np.pi) would have a omega = 2*np.pi
                
            ampcos_error : np.float, optional (kwarg)
                starndard error of the regression coeffcient for the cosine wave
                
            ampsin_error : np.float, optional (kwarg)
                starndard error of the regression coeffcient for the sine wave 
                
        Returns:
            amp : np.float
                amplitude of the sinusoidal wave
            phase : np.float
                phase (max value) in time of the sinusoidal wave 
            amperr : np.float, output 0 if any of the kwarg is not given 
                standard error of the amplitude of the sinusoidal wave
            phaseerr : np.float, output 0 if any of the kwarg is not given 
                standard error of the phase (max value) in time of the sinusoidal wave 
            
                
        """
        amp = np.sqrt(ampcos**2 + ampsin**2)
        phi = np.arctan(ampcos/ampsin)
        phase = (np.pi/2-phi)/omega
        
        if ampcos_error > 0 and ampcos_error > 0:
            amp_error = np.sqrt(ampcos_error**2 + ampsin_error**2)
            phi_error = np.arctan(ampcos_error/ampsin_error)
            phase_error = (np.pi/2-phi_error)/omega
        else :
            amp_error = 0.
            phase_error = 0.
            
            
        return {'amp':amp, 'phase':phase, 'amperr':amp_error, 'phaseerr':phase_error}
        
        
        
        
        





        
    """
    The method calculates the sinusoidal phase and total amplitude.
    It also calculate the corresponding error if the error is given
    
    
      for finding the error in annual phase, 
        1. calculate the annual signal (Ann)
        2. calculate two types of possible error (Err) : A*cos+B*sin or A*cos-B*sin (A, B are error in cos sin amplitude)
        3. combine annual signal with two types of error, each type has two combination Ann+Err or Ann-Err, therefore, there are total four types of possible changes
   
    Input:
    amplitude of cosine   : ampcos   (matrix)
    amplitude of sine     : ampsin   (matrix)
    uncertainty of ampcos : ampcos_error  (matrix) optional
    uncertainty of ampsin : ampsin_error  (matrix) optional
   
    Output: 
    phase       : 'phase'       (matrix)  unit day
    phase error : 'phase error' (matrix)  unit days of uncertainty
      if the uncertainty of ampcos or ampsin is not assigned the output would be nan

   
     """    
    
    
    def annual_phase(ampcos,ampsin,ampcos_error=-1 ,ampsin_error=-1):
 
    # initialize
    ori_shape=ampcos.shape                           # original dimension
    ampcos_array=ampcos.reshape(ampcos.size)         # convert to array
    ampsin_array=ampsin.reshape(ampsin.size)         # convert to array 
    ann_array=np.zeros((ampcos_array.shape[0],365))
    time=np.arange(365.)/365.
    phase_array=np.zeros(ampcos_array.shape)

    # calculate the annual signal
    for k in range(time.shape[0]):
        ann_array[:,k]=ampcos_array*np.cos(time[k]*2*np.pi)+ampsin_array*np.sin(time[k]*2*np.pi)

    # find the max value in the annual signal (the time of that max value is the phase)      
    for i in range(ampcos_array.shape[0]):
        temp=ann_array[i,:]
        maxval=np.max(temp)
        if np.size(np.where(temp==maxval)[0]) != 0 :
           index=np.where(temp==maxval)[0][0]
           phase_array[i]=np.round(time[index]*365.)

    # convert the array back to matrix 
    phase=phase_array.reshape(ori_shape)
    phase_error=np.zeros(phase.shape)

    if np.nansum(ampcos_error)>0 and np.nansum(ampsin_error)>0 :
       ampcos_error_array=ampcos_error.reshape(ampcos_error.size)         # convert to array
       ampsin_error_array=ampsin_error.reshape(ampsin_error.size)         # convert to array 
       ann_error_array=np.zeros((ampcos_error_array.shape[0],365,2))
       phase_error_array=np.zeros(ampcos_error_array.shape)
      
       # calculate the annual error signal
       for k in range(time.shape[0]):
           ann_error_array[:,k,0]=ampcos_error_array*np.cos(time[k]*2*np.pi)+ampsin_error_array*np.sin(time[k]*2*np.pi)
           ann_error_array[:,k,1]=ampcos_error_array*np.cos(time[k]*2*np.pi)-ampsin_error_array*np.sin(time[k]*2*np.pi)


       # find the max spread in the annual error signal (the time of that max value is the phase)      
       for i in range(ampcos_error_array.shape[0]):
           phase_spread=np.zeros(2)
           for kk in range(2):
               temp=-ann_error_array[i,:,kk]+ann_array[i,:]
               temp2=ann_error_array[i,:,kk]+ann_array[i,:]
               maxval=np.max(temp)
               maxval2=np.max(temp2)
               if np.size(np.where(temp==maxval)[0]) != 0 and np.size(np.where(temp2==maxval2)[0]) != 0 :
                  index=np.where(temp==maxval)[0][0]
                  index2=np.where(temp2==maxval2)[0][0]
                  phase1=np.round(time[index]*365.)
                  phase2=np.round(time[index2]*365.)
                  diff=np.abs(phase1-phase2)
                  if diff > 365./2. :
                     #print "pass 0"
                     phase_spread[kk]=365.-np.max([phase1,phase2])+np.min([phase1,phase2])
                  else :
                     #print "no pass 0"
                     phase_spread[kk]=diff
           phase_error_array[i]=np.max(phase_spread)

       # convert the array back to matrix 
       phase_error=phase_error_array.reshape(ori_shape)/2.

    else: 
       phase_error=np.float('nan')           