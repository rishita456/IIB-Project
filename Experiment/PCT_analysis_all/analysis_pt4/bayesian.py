
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
import config
from scipy.signal import butter,filtfilt
import statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import curve_fit

gains = config.TRIAL_GAINS_9
# gains = np.array(gains).flatten()

# Create an array of ones with the same length as original_array
ones_array = np.ones(len(gains[1]))

# Use np.insert() to insert ones before each element in the original array
phase_1_gains = np.insert(gains[1], np.arange(len(gains[1])), ones_array)

final_gains = [gains[0], phase_1_gains]
final_gains = np.hstack(final_gains)


plt.hist(final_gains)
plt.show()

