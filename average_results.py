import numpy as np

n_files
name = 'results_maEI'

average = np.zeros((n_files,2))

for i in range(n_files):
    average += np.loadtxt(name+str(i)+'.txt')
    
np.savetxt(name+'_average.txt',average)