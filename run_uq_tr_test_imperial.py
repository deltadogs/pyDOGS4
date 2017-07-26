import numpy as np
import dogs, uq
import scipy.io as io
import tr

# The second time of running the algorithm.
yE = np.array([])
SigmaT = np.array([])
T = np.array([])

# Read from surr_J_new.
zs = np.loadtxt("allpoints/surr_J_new.dat")

xx = uq.data_moving_average(zs, 40).values
ind = tr.transient_removal(xx)
sig = np.sqrt(uq.stationary_statistical_learning_reduced(xx[ind:], 18)[0])
t = len(zs)  # not needed for Alpha-DOGS
print("-------------------")
print(' len of transient = %', ind/t*100)
J = np.abs(np.mean(xx[ind:]))

yE = np.hstack((yE, J))
SigmaT = np.hstack((SigmaT, sig))
T = np.hstack((T, t))  # not needed for Alpha-DOGS
data = {'yE': yE, 'SigmaT': SigmaT, 'T': T}
io.savemat("allpoints/Yall", data)

print("The second time running the iteration")
print(' len of yE = ', len(yE))
# print('iter k = ', k)
print('function evaluation at this iteration: ', J)
