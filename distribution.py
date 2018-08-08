import scipy.stats
import sys


class ci2nstd:
    """
        Input confidence interval "ci" in %, degree of freedoms,
        and number of tails in the designed test.
        
        The program will output the critical value alpha and the
        associated number of standard deviation based on the method picked 
        (different distribution)
    """
    def __init__(self) :
        return
    
    def t_dist(self, ci, dof, ntail=2):
        """
        Input confidence interval "ci" in %, degree of freedoms,
        and number of tails in the designed test.
        
        The program will output the critical value alpha and the
        associated number of standard deviation based on the t-dist.

        Parameters:
            ci : numpy float
                represents the confidence interval in %
            dof : numpy int
                represents the degree of freedom in the test
            ntail: kwarg, numpy int, with only two options (1 or 2)
                number of tails in the designed test
        
        Returns: 
            nstd : numpy float
                associated number of standard deviation based on the t-dist
                
        Attributes:
            self.alpha
            self.nstd
            self.dof
            self.ntail

        """
        if ntail != 1 and ntail != 2 :
            print "ntail can only be 1 or 2 "
            sys.exit('')
        else:
            self.alpha=1.-ci/100.
            self.nstd=scipy.stats.t.ppf(1.0-(self.alpha/ntail),dof)
            self.dof=dof
            self.ntail=ntail
        return self.nstd