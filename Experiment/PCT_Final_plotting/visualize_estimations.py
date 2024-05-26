import pygame
import math
import config
import matplotlib.pyplot as plt
import numpy as np
from  simulate import simulate
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples
from gameObjectsV2 import Pendulum, Cart
from constant_timescale_alltrials import get_dist
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
from sklearn.metrics import r2_score 

# reason for butterworth - no passband ripple, high attenuation and smooth roll off
def butter_lowpass_filter(data, cutoff=4, fs=120, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def get_current_point(index, pos_es):

    return(pos_es[index]), 0

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx

    
   
def main():

    
    gain = 1
    trial = 1
    P_id = 9

    # time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
    # position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
    # theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
    time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_9/test_trials/85sec_trial/time.npy')
    position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_9/test_trials/85sec_trial/position.npy')
    theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_9/test_trials/85sec_trial/angle.npy')
    calibration_index =  np.where(time < 15)
    calibration_index2 =  np.where(time < 30)
    time = time[0:calibration_index[-1][-1]]
    theta = theta[0:calibration_index[-1][-1]]
    position = position[0:calibration_index[-1][-1]]
    # time = time[calibration_index[-1][-1]:calibration_index2[-1][-1]]
    # theta = theta[calibration_index[-1][-1]:calibration_index2[-1][-1]]
    # position = position[calibration_index[-1][-1]:calibration_index2[-1][-1]]
    time_og = time
    position_og = position
    theta_og = theta

    x = np.reshape(time_og, (len(time_og), 1))
    model = LinearRegression()

    try:
        # pos = position[0:-2]-750
        pos = position-750
        model.fit(x, pos)
    except ValueError:
        pos = position[0:-2]-750
        # pos = position-750
        model.fit(x, pos)
    # calculate trend
    trend = model.predict(x)
    detrended_pos = [pos[i]-trend[i] for i in range(0, len(time))]
    plt.plot(pos)
    plt.plot(detrended_pos)
    plt.show()
    # print(len(theta))
    theta_steps = []
    steps = []

    for i in range(len(detrended_pos)-1):
        steps.append(detrended_pos[i+1]-detrended_pos[i])
        theta_steps.append(theta[i+1]-theta[i])
    steps = np.array(steps)
    steps_og = steps


 
    steps = steps_og[0::2]

    if np.count_nonzero(steps) < len(steps_og)/2:
        steps = steps_og[1::2]

    steps = np.repeat(steps, 2)
    steps = np.insert(steps,0, steps[0])
    steps = np.insert(steps,0, steps[0])
    steps_og2 = steps
  

    detrended_pos = np.array(detrended_pos)
    theta_steps = np.array(theta_steps)
    theta_steps = np.insert(theta_steps, 0,theta[0])
    theta_steps_og = theta_steps
    

    # Filter the data
    y_og = butter_lowpass_filter(steps_og2)

    lenz = (len(theta)//200)*200
 
    opt_kp, opt_kd, filter_noise_samples, fit_noise_samples = get_dist(len(theta)//200)
    opt_kp = np.repeat(opt_kp, 200)
    opt_kd = np.repeat(opt_kd, 200)

    step_est = (opt_kp)*((gain)**-1)*theta[0:lenz] + opt_kd*((gain)**-1)*theta_steps[0:lenz]
    # y_og = y_og[2:lenz+2]
    plt.plot(y_og)
    plt.plot(step_est)
    plt.title(str(r2_score(y_og[0:lenz], step_est)))
    plt.show()



    time = time[0:len(step_est)]
    theta = theta[0:len(step_est)]
    position = position[0:len(step_est)+2]
    # time = time[0:len(steps)]
    # theta = theta[0:len(steps)]
    # position = position[0:len(steps)+2]

    # filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
    # fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
    
    # plt.plot(time, filtered_steps, label = 'filtered steps')
    # plt.plot(time, step_est, label = 'step estimates')
    # plt.legend()
    # plt.show()

    dt = 0.017

    

    start = (position[0], theta[0])
    pos_es, ang_es, cartacc = simulate(butter_lowpass_filter(step_est+filter_noise_samples), start, dt, time, 1)
    # pos_es, ang_es, cartacc = simulate(steps_og2, start, dt, time, 10000)
   
    x = np.reshape(time, (len(time), 1))
    model = LinearRegression()

    try:
        # pos = position[0:-2]-750
        pos = pos_es-750
        model.fit(x, pos)
    except ValueError:
        pos = pos_es[0:-1]-750
        # pos = position-750
        model.fit(x, pos)
    # calculate trend
    trend = model.predict(x)
    detrended_pos = [pos[i]-trend[i] for i in range(0, len(time))]
    plt.plot(pos)
    plt.plot(detrended_pos)
    plt.show()
    
    score = 0
    # Initialize Pygame
    pygame.init()

    # Create the display
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    pygame.display.set_caption("Balancing Inverted Pendulum")

    
    time = time - time[0]
    # Simulation parameters
    previous_time = pygame.time.get_ticks() / 1000.0

    # Initialise game objects
    cart = Cart()
    pendulum = Pendulum()
    pendulum.previous_time = previous_time
    cart.previous_time = previous_time
    pendulum.set_cart(cart)

    
    clock = pygame.time.Clock()

    # Main game loop
    running = True
    now = 0
    end = 60
    start_time = 0
    play = False
    index = 0
    while now<end and running == True:

        pendulum.c = 0
        clock.tick(120)
        now = pygame.time.get_ticks() / 1000.0
        visual_gain = gain
        if play:
            if abs(math.degrees(pendulum.pendulum_angle)) < 2:
                screen.fill((20, 255, 20)) 
                score = score + (2-math.degrees(pendulum.pendulum_angle))
        
        pygame.draw.rect(screen, (0,128,0), (0, 10, pendulum.pendulum_angle, 10))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                print('wooooooooop')
                now = pygame.time.get_ticks() / 1000.0
                start_time = now
                pendulum.c = 1
                end = now + time[-1    ]
                play = True
                index = find_nearest(time, ((pygame.time.get_ticks() / 1000.0)-start_time))

        keys = pygame.key.get_pressed()

        if keys[pygame.K_SPACE]:
            index = find_nearest(time, ((pygame.time.get_ticks() / 1000.0)-start_time))
            screen_position, raw_position = get_current_point(index, pos_es)
            # screen_position, raw_position = get_current_point(index, position)
            pendulum.raw_position_history.append(raw_position)
            
            cart.cart_x = screen_position
            pendulum.pendulum_angle = math.radians(ang_es[index])
            # pendulum.pendulum_angle = math.radians(theta[index])
            

            # Draw cart
            cart.draw_cart(screen)
            # Draw pendulum
            pendulum.draw_pendulum(screen, visual_gain)
            now = pygame.time.get_ticks() / 1000.0
            print("final system displayed - ", pygame.time.get_ticks())
       
            print(index)

        
        pendulum.update_pendulum_state(pendulum.cart.cart_x, play, start_time, 1)
        # Clear the screen
        screen.fill(config.BACKGROUND_COLOR)
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render(str(int((1*score))), True, (255,255,255), (0,0,255))
        textRect = text.get_rect()
        textRect.center = (config.WIDTH - 100 ,  100)
        screen.blit(text, textRect)
        if play:

            if abs(math.degrees(pendulum.pendulum_angle)) < 2:
                screen.fill((20, 255, 20))
                screen.blit(text, textRect) 
        
        # Draw cart
        cart.draw_cart(screen)

        # # Draw pendulum
        pendulum.draw_pendulum(screen, visual_gain)



        pygame.display.flip()

    # Quit Pygame
    pygame.quit()
    print(now)
    

    time2, angle2, position2, raw_hand_position = pendulum.get_history()

    plt.plot(time2,angle2, label = 'simulation')
    plt.plot(time, theta, label = 'original')
    plt.plot(time, np.zeros(len(time)))
    plt.xlabel('Time')
    plt.ylabel('Pendulum angle')
    plt.ylim = (-5,5)
    plt.show()
    plt.plot(time2,position2[1:-1], label = 'simulation')
    plt.plot(time, position[2:], label = 'original')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.show()

    # print(rawposition[2:22][0::2]-rawposition[0:20][0::2])
    # print(position[2:22][0::2]-position[0:20][0::2])
    # print((position[2:22]-position[0:20])/(rawposition[2:22]-rawposition[0:20]))
   

if __name__ == "__main__":
    main()

# /Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot_exp_data/Tsep_6_g1_2_g2_20