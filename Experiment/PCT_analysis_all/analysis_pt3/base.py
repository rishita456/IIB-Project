import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
import config
import plotly.graph_objects as go
from scipy.signal import butter,filtfilt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf


def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# training_gains = [2, 5, 7, 10, 12, 15, 17, 20, 22]
training_gains = [5,]
trial = 1
P_id = 5

ars = []

fracs = []

for gain in training_gains:
    time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
    position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
    theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')

    # detrend position data

    x = np.reshape(time, (len(time), 1))
    model = LinearRegression()
    pos = position[0:-2]-750
    model.fit(x, pos)
    # calculate trend
    trend = model.predict(x)
    detrended_pos = [pos[i]-trend[i] for i in range(0, len(time))]

    time = time[1::2]
    detrended_pos = detrended_pos[1::2]



    sequence = []
    steps = []
    start = detrended_pos[0]

    for i in range(len(detrended_pos)-1):
        steps.append(detrended_pos[i+1]-detrended_pos[i])
    print(steps[1:10])
    steps = np.array(steps)
    for i in range(len(steps)):
        if steps[i]>0:
            sequence.append(1)

        else:
            sequence.append(0)

    sequence = np.array(sequence)
    counts, _ = np.histogram(sequence)
    fracs.append(counts[1]/(counts[0]+counts[1]))

    # low pass 

    # Filter requirements.
    T = 30         # Sample Period
    fs = 120      # sample rate, Hz
    cutoff = 1      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2       # sin wave can be approx represented as quadratic
    n = int(T * fs) # total number of samples

    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(steps, cutoff, fs, order)
    plt.plot(steps)
    plt.plot(y)
    plt.show()
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #             y = steps,
    #             line =  dict(shape =  'spline' ),
    #             name = 'signal with noise'
    #             ))
    # fig.add_trace(go.Scatter(
    #             y = y,
    #             line =  dict(shape =  'spline' ),
    #             name = 'filtered signal'
    #             ))
    # fig.show()

    # plt.plot(steps)
    # plt.plot(y)
    # plt.show()
    y_train = y[1000:2000]

    # plt.figure(figsize=(10, 6))
    # plot_acf(y, lags=20)  # Plot ACF up to lag 20
    # plt.title('Autocorrelation Function (ACF) of Time Series')
    # plt.xlabel('Lag')
    # plt.ylabel('Autocorrelation')
    # plt.grid(True)
    # plt.show()

    # Create lagged version of the time series for AR model
    lag = 50  # First-order AR model
    X = np.vstack([y[lag-i:-i] for i in range(1, lag+1)]).T
    X_train = np.vstack([y_train[lag-i:-i] for i in range(1, lag+1)]).T

    x_test = np.zeros_like(X)

    # Add constant term for intercept in the model
    # X = sm.add_constant(X)

    # Fit the AR model using OLS
    model = sm.OLS(y[lag:], X)
    results = model.fit()

    # Display results
    print(results.summary())

    # Extract AR coefficient from results
    ar_coef = results.params[0]
    ars.append(ar_coef)


    print("Estimated AR coefficient:", ar_coef)
    # step_counts, _ = np.histogram(y,50, [-1,1], density=True)
    # plt.plot(np.linspace(-1,1,50), step_counts, label = 'Gain - '+ str(gain))
    # plt.show()

    # Predict the values using the fitted AR model
    y_pred = results.predict(X)


    # Plot the original data and the fitted values
    plt.figure(figsize=(10, 6))
    plt.plot(y, label='Original Data')
    plt.plot(np.arange(lag, len(y)), y_pred, label='Fitted AR Model', linestyle='--')
    plt.title('AR Model Fit')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    plt.scatter(np.arange(lag, len(y)), y_pred - y[lag:])
    plt.show()

    plt.hist(y_pred - y[lag:])
    plt.show()




    # # fft 
    # fft_result = np.fft.fft(steps)
    # freq = np.fft.fftfreq(len(time), time[1] - time[0])  # Frequency bins
    # fft_result2 = np.fft.fft(y)
    # freq2 = np.fft.fftfreq(len(time), time[1] - time[0])  # Frequency bins

    # # Plot FFT
    # plt.figure(figsize=(10, 6))
    # plt.plot(freq[0:-1], np.abs(fft_result))
    # plt.plot(freq2[0:-1], np.abs(fft_result2))
    # plt.title('FFT of Time Series')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.show()

    # plt.hist(y)
    # plt.show()

    
plt.plot(training_gains, ars)
plt.show()

plt.hist(sequence)
plt.show()