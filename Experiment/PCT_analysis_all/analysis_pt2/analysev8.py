import numpy as np
import matplotlib.pyplot as plt
import config


tests = np.ones((1,10))
gains = np.array(config.TRIAL_GAINS_5).flatten()
gains = gains[0:-2]
pos = np.linspace()
plt.hist(gains, density=True )
plt.show()