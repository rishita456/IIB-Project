import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
import config
from scipy.signal import butter,filtfilt
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
import math


def calc_func_find_Kp_0_part0(K, x, y):


    a = K[0]
    b = K[1]
    c = K[2]

    # return np.linalg.norm((-y) + (Kp_0*((pg)**-1)*theta[0:-1] + Kd_0*((pg)**-1)*theta_steps[0:-1]))#/math.sqrt(len(y))
    return np.linalg.norm((y) - (a*(x) + b*x + c) )#/math.sqrt(len(y))



x_base = np.linspace(0,3,1000)
y_base = 2*np.sin(5*x_base) + 5*np.cos(10*x_base)
y_base = y_base #+ 0.001*np.random.randn(1000)




est = np.array([])
limits = np.arange(0,1000+1,5)
for i in range(len(limits)-1):

    limit = (limits[i], limits[i+1])
    x = x_base[limit[0]:limit[1]]
    y = y_base[limit[0]:limit[1]]

    result = minimize(calc_func_find_Kp_0_part0, (1,1,1), args=(x, y))

    est = np.append(est, ((result.x[0]*(x) + result.x[1]*x + result.x[2])))

plt.plot(x_base, est, label = '5')


est = np.array([])
limits = np.arange(0,1000+1,10)
for i in range(len(limits)-1):

    limit = (limits[i], limits[i+1])
    x = x_base[limit[0]:limit[1]]
    y = y_base[limit[0]:limit[1]]

    result = minimize(calc_func_find_Kp_0_part0, (1,1,1), args=(x, y))

    est = np.append(est, ((result.x[0]*(x) + result.x[1]*x + result.x[2])))

plt.plot(x_base, est, label = '10')

est = np.array([])
limits = np.arange(0,1000+1,20)
for i in range(len(limits)-1):

    limit = (limits[i], limits[i+1])
    x = x_base[limit[0]:limit[1]]
    y = y_base[limit[0]:limit[1]]

    result = minimize(calc_func_find_Kp_0_part0, (1,1,1), args=(x, y))

    est = np.append(est, ((result.x[0]*(x) + result.x[1]*x + result.x[2])))

plt.plot(x_base, est, label = '20')


est = np.array([])
limits = np.arange(0,1000+1,40)
for i in range(len(limits)-1):

    limit = (limits[i], limits[i+1])
    x = x_base[limit[0]:limit[1]]
    y = y_base[limit[0]:limit[1]]

    result = minimize(calc_func_find_Kp_0_part0, (1,1,1), args=(x, y))

    est = np.append(est, ((result.x[0]*(x) + result.x[1]*x + result.x[2])))

plt.plot(x_base, est, label = '40')

plt.plot(x_base, y_base, label = 'original')
plt.legend()
plt.show()

