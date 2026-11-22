import math
import numpy as np

def decay_function(half_life, t0, td, tc = 0):

    """
    This is a function that will return the decay decay term from pre and post bombardment
    decay. This function is very simple and based on the half_life of the specific isotope
    and the respective times of bombardment and pure decay.

    t0 = the time the target was in beam (in seconds)
    td = the time the target was between the beam and counting (in seconds)
    tc = the time that the target was being counted (in seconds)
    
    """

    # defining the decay constant
    lam = np.log(2) / half_life
    # print(lam)

    # calculating exponentials
    bombardment_decay = 1 - np.exp(-lam * t0)
    # print(bombardment_decay)
    post_beam_decay = np.exp(-lam * td)
    # print(post_beam_decay)
    if tc > 0:
        counting_decay = 1 - np.exp(-lam * tc)
    else:
        counting_decay = 1

    decay_term = bombardment_decay * post_beam_decay * counting_decay

    # retuening the product
    return decay_term
   

t_beam = 60 * 60 # The rough estimate of the target bombardment time [1hr]
t_counting = 90 * 60 # The longest time at which we would want to see an activity of ~166 Bq [90 min]
In_halflife = 54.29 * 60

print(f'{decay_function(In_halflife, t_beam, t_counting)}')