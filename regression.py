#!python

import numpy as np
from numpy import linalg as la

class regress :
    """
    The regress class is a modification and improvement of the module.regress.py
    The class is created to be more generic in importing different number of variate
    when performing multi-variate regression. 
    
    method :
        1) multivar_regress
        2) def_vars
    
    
    """
    
    def __init__(self, axis, axis_rel=0., err_weighted='N',err=1.):
        """
        Parameters:
            axis : np.array,
                one dimension numpy array that define the time/space
                of each variate
                
            axis_rel : np.float
                number representing the reference value on the axis. 
                Default is set to 0 => meaning the axis remains as it is
                if list_variates is set manually => keep axis_rel=0.
                    unless the list_variates is also corrected based on
                    the axis_rel value.
                
            err_weighted : string
                'Y' or 'N' to determine whether the weighted regression is 
                perfermed or not. The weighted regression propogate the 
                measurement error in the observation to the regression 
                
            err : np.float or np.array
                measurement error per observation (can be different per obs.)
                but need to be the same length as the axis array
                
        Attributes:
            self.axis: np.array
                axis of the time/space of each variate after the relative 
                frame is removed. This is stored as attribute for future 
                reference
                
        """
        
        self.axis=axis-axis_rel
        self.axis_rel=axis_rel
        self.err_weight=err_weighted
        self.err=err
        
        return

        
    def def_vars(self, predef_var='', list_variates=[]):
        """
        Creating the designed matrix X for the regression
        
        Parameters:
        
            list_variate : list,
                list of variates used for the regression
                [var1, var2, .....]
                variate that one want to relate to the original data 
                Design matrix X shows the var1, var2,... as colume
                Each Column represents a variate
        
                        /  1  x11 x21 ... xn1 \          / y1 \
                        |  1  x12 x22 ... xn2 |          | y2 |
                        |  .   .   .       .  |          |  . |
                    X = |  .   .   .       .  |      Y = |  . |
                        |  .   .   .       .  |          |  . |
                        \  1  x1p x2p ... xnp /          \ yp /

                where x1j (j = 0 -> p) means how each variate changes with time/space
                      xi1 (i = 0 -> n) means different variates
                      yj  means the original data at a single location or time

            predef_var: string,
                kwarg to call for predefined variates 
                options - semisea_sea_lin, semisea_sea_lin_quad, fort13_fort14_lin
                
        Return:
            X : np.array 
                numpy matrix representing the designed matrix
                
        Attribute:
            self.dm_order : list
                list of name of the list_variates order
                put in the designed matrix.


        """
    
        if predef_var == '':    
            self.num_var=len(list_variates) # number of variates 
            X=np.zeros([len(list_variates[0]),self.num_var],dtype=np.float64)
            self.dm_order=[]
            # designed matrix
            for i in range(self.num_var):
                X[:,i]=list_variates[i]   
                self.dm_order.append('var%i'%(i+1))
        else:
            # variates below is assuming the time variable is in the unit of year
            con=np.zeros([self.axis.shape[0],],dtype=np.float64)+1.
            lin=np.zeros([self.axis.shape[0],],dtype=np.float64)+(self.axis)
            quad=np.zeros([self.axis.shape[0],],dtype=np.float64)+(self.axis)**2

            anncos=np.zeros([self.axis.shape[0],],dtype=np.float64)+np.cos(self.axis*2.*np.pi)
            annsin=np.zeros([self.axis.shape[0],],dtype=np.float64)+np.sin(self.axis*2.*np.pi)
            semianncos=np.zeros([self.axis.shape[0],],dtype=np.float64)+np.cos(self.axis*4.*np.pi)
            semiannsin=np.zeros([self.axis.shape[0],],dtype=np.float64)+np.sin(self.axis*4.*np.pi)
            fortnight1_cos=np.zeros([self.axis.shape[0],],dtype=np.float64)\
            +np.cos(self.axis*2.*np.pi*365./13.66)
            fortnight1_sin=np.zeros([self.axis.shape[0],],dtype=np.float64)\
            +np.sin(self.axis*2.*np.pi*365./13.66)
            fortnight2_cos=np.zeros([self.axis.shape[0],],dtype=np.float64)\
            +np.cos(self.axis*2.*np.pi*365./14.77)
            fortnight2_sin=np.zeros([self.axis.shape[0],],dtype=np.float64)\
            +np.sin(self.axis*2.*np.pi*365./14.77)
 
            # Fit (constant + linear trend + annual + semiannual)          
            if predef_var == 'semisea_sea_lin':
                X=np.zeros([self.axis.shape[0],6],dtype=np.float64)
                X[:,0]=con[:]
                X[:,1]=lin[:]
                X[:,2]=anncos[:]
                X[:,3]=annsin[:]
                X[:,4]=semianncos[:]
                X[:,5]=semiannsin[:]
                self.dm_order=['con','lin','anncos','annsin','semianncos','semiannsin']
            # Fit (constant + linear trend + annual + semiannual + quadratic)
            elif predef_var == 'semisea_sea_lin_quad':
                X=np.zeros([self.axis.shape[0],7],dtype=np.float64)
                X[:,0]=con[:]
                X[:,1]=lin[:]
                X[:,2]=anncos[:]
                X[:,3]=annsin[:]
                X[:,4]=semianncos[:]
                X[:,5]=semiannsin[:]
                X[:,6]=quad[:]
                self.dm_order=['con','lin','anncos','annsin','semianncos','semiannsin','quad']
            # Fit (constant + linear + fortnightly1 + fortnightly2)
            elif flag == 'fort13_fort14_lin':
                X=np.zeros([self.axis.shape[0],6],dtype=np.float64)
                X[:,0]=con[:]
                X[:,1]=lin[:]
                X[:,2]=fortnight1_cos[:]
                X[:,3]=fortnight1_sin[:]
                X[:,4]=fortnight2_cos[:]
                X[:,5]=fortnight2_sin[:] 
                self.dm_order=['con','lin','fortnight13_cos','fortnight13_sin'\
                               ,'fortnight14_cos','fortnight14_sin']
#         self.dm=X
        return X
            
    def multivar_regress(self, Y, predef_var='', list_variates=[], annsig=False) :
        """
        The method is the multi-variate-regression that simultaniously fit 
        multiple variates (x1,x2,....) in one single regression to derive
        the best fitting line to represent the original data (Y)
        
        Parameters:
            Y: np.array 
              the original data value (Vector of responses)
              
            list_variates: list, 
                list of variates used for the regression
                [var1, var2, .....] to form the design matrix
                
                    Design matrix = X and Vector of response Y

                        /  1  x11 x21 ... xn1 \          / y1 \
                        |  1  x12 x22 ... xn2 |          | y2 |
                        |  .   .   .       .  |          |  . |
                    X = |  .   .   .       .  |      Y = |  . |
                        |  .   .   .       .  |          |  . |
                        \  1  x1p x2p ... xnp /          \ yp /

            predef_var: string,
                kwarg to call for predefined variates 
                options - semisea_sea_lin, semisea_sea_lin_quad, fort13_fort14_lin}


         Returns:
            beta: 
                the coefficients (dim = num of variates) that relate to each variate
                It's order is following the order of variates in the list_variates
            err: 
                the errors that makes the linear combined regression value to deviate 
                from the original data (Y)
        
                                 / beta0 \          / e1 \
                                 | beta1 |          | e2 |
                                 |   .   |          |  . |
                          beta = |   .   |    err = |  . |
                                 |   .   |          |  . |
                                 \ betan /          \ ep /
                                 
        
         Mathematical expression:

              R^2= sum (Y-beta*X)   
              find the minimum of the equation => first order differential = 0
              first order diffferential equation is  X'Y=(X'X) (beta)
              so  beta = (X'X)^-1 (X'Y)
              where prime means transpose of matrix, ^-1 means inversion of matrix

              to solve the weighted system
              R^2= sum W(Y-beta*X)   
              find the minimum of the equation => first order differential = 0
              first order diffferential equation is  X'WY=(X'WX) (beta)
              so  beta = (X'WX)^-1 (X'WY)
              where 
                        /1/E^2  0    0   ...  0   \
                        |  0  1/E^2  0   ...  0   |
                        |  0    0  1/E^2 ...  0   |
                    W = |  .    .    .            |
                        |  .    .    .            |
                        \  0    0    0   ...1/E^2 /
                        
              the matrix representing the reciprical of the error square 
              at each observed value (can be different at each time step)
              The weighted regression propogate the measurement error in 
              the observation to the regression [Schuckmann et al., 2011,
              Llovel et al., 2014]
            
                
        """

        #-- reshape the Y array to column
        Y=np.reshape(Y,[Y.shape[0],1])
        # determine the design matrix
        X=self.def_vars(predef_var=predef_var, list_variates=list_variates)
        
        #-- determine to perform weighted regression or not
        if self.err_weight.upper() in ['Y']:
            ww=np.zeros(len(time))+1./self.err**2           # the variance of obs. error
            W=np.diag(ww) 
            term1=np.dot(np.transpose(X),np.dot(W,X))       #(X'WX)            
        else :
            term1=np.dot(np.transpose(X),X)                 #(X'X)
            
        #-- calculate the determinant for matrix inverse 
        detterm1=la.det(term1)
        if detterm1 == np.float64(0.) :
           print "Error64: No inverse of (X'X) or (X'WX) "
           return
        
        #-- calculate the matrix inverse 
        term1_1=la.inv(term1)                               #(X'X)^-1 or (X'WX)^-1
        
        #-- calculate the right hand side
        if self.err_weight.upper() in ['Y']:  
            term2=np.dot(np.transpose(X),np.dot(W,Y))       #(X'WY)            
        else :
            term2=np.dot(np.transpose(X),Y)                 #(X'Y) 
            
        #-- matrix multiplication of RHS and inverse matrix
        beta=np.dot(term1_1,term2)                          #(X'X)^-1(X'Y) or (X'WX)^-1(X'WY)
                
        #-- store attribute in the regression
        self.nvars=X.shape[1]                              # store number of variates
        self.dof=len(self.axis)-self.nvars                 # store degree of freedom in the regress
     
        #-- calculate uncertainties
        # the diag of rmse*sqrt(X_var)^-1 represents the std of uncertainty 
        # in the regression for each variate
        Yreg=np.dot(X,np.reshape(beta,[beta.shape[0],1]))  # (X) dot (beta) => regression model
        Yreg=np.reshape(Yreg,[Yreg.shape[0],1])
        misfit=Y-Yreg     
        rmse=np.sqrt(np.sum(misfit**2)/(self.dof))         # RMSE between the model and obs.
        X_var=np.dot(np.transpose(X),X)
        se=rmse*np.sqrt(np.diag(la.inv(X_var)))  
        
        #-- output the design matrix in the form of list_variates
        if list_variates == []:
            list_variates = [X[:,i] for i in range(X.shape[1])] 
            

            
        return {'beta':beta[:,0],'se':se,'rmse':rmse,
                  'model':Yreg[:,0],'list_variates':list_variates}

    
    

