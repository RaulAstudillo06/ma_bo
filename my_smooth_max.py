import numpy as np
import scipy.misc

def smooth_max(y, k):

    m = np.max(y)
    #val = scipy.misc.logsumexp(k*(x-m))/k + m
    val = np.log(np.sum(np.exp(k*(y - m))))/k + m
    #print(val)
    return val


def smooth_max_gradient(y, k):
    m = np.max(y)
    tmp = np.exp(k*(y - m))
    gradient = np.divide(tmp, np.sum(tmp))
    return gradient
