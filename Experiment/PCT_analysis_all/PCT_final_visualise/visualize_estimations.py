import pygame
import math
import config
import matplotlib.pyplot as plt
import numpy as np
from  simulate import simulate
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples
from gameObjectsV2 import Pendulum, Cart

def get_current_point(index, pos_es):

    return(pos_es[index]), 0

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

    
   
def main():

    
    gain = 2
    trial = 1
    P_id = 9

    time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
    position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
    theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
    position_og = position
    steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time, position, gain*theta, gain, 1, 6, size= np.array([200]))
    time = time[0:len(step_est)]
    theta = theta[0:len(step_est)]
    position = position[0:len(step_est)+2]
    # time = time[0:len(steps)]
    # theta = theta[0:len(steps)]
    # position = position[0:len(steps)+2]

    filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
    fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
    
    plt.plot(time, filtered_steps, label = 'filtered steps')
    plt.plot(time, step_est, label = 'step estimates')
    plt.legend()
    plt.show()

    dt = 0.017

    

    start = (position[1], theta[1])
    pos_es, ang_es, cartacc = simulate(step_est  + filter_noise_samples + fit_noise_samples, start, dt, time)
    # pos_es, ang_es, cartacc = simulate(steps, start, dt, time)
   
    
    
    score = 0
    # Initialize Pygame
    pygame.init()

    # Create the display
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    pygame.display.set_caption("Balancing Inverted Pendulum")

    

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
    plt.plot(time, position[1:-1], label = 'original')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.show()

    # print(rawposition[2:22][0::2]-rawposition[0:20][0::2])
    # print(position[2:22][0::2]-position[0:20][0::2])
    # print((position[2:22]-position[0:20])/(rawposition[2:22]-rawposition[0:20]))
   

if __name__ == "__main__":
    main()

# /Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot_exp_data/Tsep_6_g1_2_g2_20