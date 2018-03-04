# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

class ParameterDistribution(object):
    """
    Class to handle the parameter distribution of the utility function.
    There are two possible ways to specify a parameter distribution:  

    - Arm bandit
    space  =[{'name': 'var_1', 'type': 'bandit', 'domain': [(-1,1),(1,0),(0,1)]},
             {'name': 'var_2', 'type': 'bandit', 'domain': [(-1,4),(0,0),(1,2)]}]

    - Continuous domain
    space =[ {'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality':1},
             {'name': 'var_2', 'type': 'continuous', 'domain':(-3,1), 'dimensionality':2},
             {'name': 'var_3', 'type': 'bandit', 'domain': [(-1,1),(1,0),(0,1)], 'dimensionality':2},
             {'name': 'var_4', 'type': 'bandit', 'domain': [(-1,4),(0,0),(1,2)]},
             {'name': 'var_5', 'type': 'discrete', 'domain': (0,1,2,3)}]

    - Discrete domain
    space =[ {'name': 'var_3', 'type': 'discrete', 'domain': (0,1,2,3)}]
             {'name': 'var_3', 'type': 'discrete', 'domain': (-10,10)}]


    - Mixed domain
    space =[{'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality' :1},
            {'name': 'var_4', 'type': 'continuous', 'domain':(-3,1), 'dimensionality' :2},
            {'name': 'var_3', 'type': 'discrete', 'domain': (0,1,2,3)}]

    Restrictions can be added to the problem. Each restriction is of the form c(x) <= 0 where c(x) is a function of
    the input variables previously defined in the space. Restrictions should be written as a list
    of dictionaries. For instance, this is an example of an space coupled with a constraint

    space =[ {'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality' :2}]
    constraints = [ {'name': 'const_1', 'constraint': 'x[:,0]**2 + x[:,1]**2 - 1'}]

    If no constraints are provided the hypercube determined by the bounds constraints are used.

    Note about the internal representation of the vatiables: for variables in which the dimaensionality
    has been specified in the domain, a subindex is internally asigned. For instance if the variables
    is called 'var1' and has dimensionality 3, the first three positions in the internal representation
    of the domain will be occupied by variables 'var1_1', 'var1_2' and 'var1_3'. If no dimensionality
    is added, the internal naming remains the same. For instance, in the example above 'var3'
    should be fixed its original name.



    param continuous: whether the .
    param constraints: list of dictionaries as indicated above (default, none)

    """
    def __init__(self, continuous=False, support=None, prob_dist=None, sample_generator=None):
        if continuous==True and sample_gen is None:
            pass
        else:
            self.continuous = continuous
            self.support = support
            self.prob_dist = prob_dist
            self.sample_generator = sample_generator
    
    def sample(self):
        if continuous:
            parameter = sample_generator()
        else:
            parameter = sample_discrete(support,prob_dist)
        return parameter
