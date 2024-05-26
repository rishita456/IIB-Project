import pygame
import sys
import math
import config
from gameObjectsV2 import Pendulum, Cart
import matplotlib.pyplot as plt
from calibrate import Calibrate
from multiprocessing import shared_memory
import pandas as pd
import numpy as np
import quaternions_exp
import config
import datetime

def get_data(varsPerDataType,noDataTypes,sharedMemoryName,frameLength = 1000,sim = False,idxesToPlot = None):

    if sim == False:
        dataEntries = varsPerDataType * noDataTypes
        SHARED_MEM_NAME = sharedMemoryName
        shared_block = shared_memory.SharedMemory(size= dataEntries * 8, name=SHARED_MEM_NAME, create=False)
        shared_array = np.ndarray(shape=(noDataTypes,varsPerDataType), dtype=np.float64, buffer=shared_block.buf)
        if idxesToPlot is not None:
            df = pd.DataFrame(shared_array[idxesToPlot])
        else:
            df = pd.DataFrame(shared_array)
    else:
        shared_array = np.random.randint(0,5,size = (43,7))
    # load the most recent shared memory onto a dataframe
    
    # we will get structure of database as rigidBody1 - [q_X,q_Y,q_Z,q_W,X,Y,Z]
    if sim == False:
        df_locations = [df.iloc[a,:] for a in range(0,df.shape[0])] # split rows into list
        df_locations_quaternionObjs = [quaternions_exp.quaternionVector(loc = [a.iloc[4]/1000,a.iloc[5]/1000,a.iloc[6]/1000],quaternion=[a.iloc[0],a.iloc[1],a.iloc[2],a.iloc[3]]) for a in df_locations]
        df_directions =  pd.DataFrame([q.qv_mult(q.quaternion,[0,-1,0]) for q in df_locations_quaternionObjs])
        dfPlot = pd.DataFrame({'x':df.iloc[:,4],'y':df.iloc[:,5],'z':df.iloc[:,6],'dirX':df_directions.iloc[:,0],'dirY':df_directions.iloc[:,1],'dirZ':df_directions.iloc[:,2]})
       
    return dfPlot.iloc[:,3],dfPlot.iloc[:,4],dfPlot.iloc[:,5]

def set_calibration(cal, screen, cart, pendulum, key, display):

    running = True
    while running:

        keys = key.get_pressed()

        if keys[pygame.K_o]:
            cal.calibrate_origin()
            print(cal.origin)
        
        elif keys[pygame.K_l]:
            cal.calibrate_xy_plane_left()
            print(cal.left_scale)

        elif keys[pygame.K_r]:
            cal.calibrate_xy_plane_right()
            print(cal.right_scale)
            running = False
    # Clear the screen
    screen.fill(config.BACKGROUND_COLOR)
    # pendulum.update_pendulum_state()
    # Draw cart
    cart.draw_cart(screen)

    # # Draw pendulum
    pendulum.draw_pendulum(screen)



    display.flip()




def main():

    score = 0
    participant_id = 11
    # Initialize Pygame
    pygame.init()

    # Create the display
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    pygame.display.set_caption("Balancing Inverted Pendulum")

    

    # Simulation parameters
    dt = 0.02
    previous_time = pygame.time.get_ticks() / 1000.0

    # Initialise game objects
    cart = Cart()
    pendulum = Pendulum()
    pendulum.previous_time = previous_time
    cart.previous_time = previous_time
    pendulum.set_cart(cart)

    
    
    # initialise calibration
    cal = Calibrate()
    
    clock = pygame.time.Clock()

    # set calibration

    # set_calibration(cal, screen, cart, pendulum, pygame.key, pygame.display)
    # Main game loop
    running = True
    now = 0
    end = 60
    start_time = 0
    play = False
    while now<end and running == True:
        pendulum.c = 0
        clock.tick(120)
        now = pygame.time.get_ticks() / 1000.0
        visual_gain = config.VISUAL_ANGLE_GAIN
        gain = config.ANGLE_GAIN
        if play:
            if abs(math.degrees(pendulum.pendulum_angle)) < 2:
                screen.fill((20, 255, 20)) 
                score = score + (2-math.degrees(pendulum.pendulum_angle))
        if play and now > start_time + 0: #calibration or initial stabilisation
            visual_gain = 1
        if play and now > start_time + 0: # disturbance sep 
            visual_gain = 1
        
        pygame.draw.rect(screen, (0,128,0), (0, 10, pendulum.pendulum_angle, 10))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                print('wooooooooop')
                now = pygame.time.get_ticks() / 1000.0
                start_time = now
                pendulum.c = 1
                end = now + 10
                play = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            pendulum.cart.cart_x = pendulum.cart.cart_x-1
            pendulum.update_pendulum_state(pendulum.cart.cart_x, play, start_time, gain)
             # Draw cart
            cart.draw_cart(screen)
            # Draw pendulum
            pendulum.draw_pendulum(screen, visual_gain)
  
            

        elif keys[pygame.K_RIGHT]:
            pendulum.cart.cart_x = pendulum.cart.cart_x+1
            pendulum.update_pendulum_state(pendulum.cart.cart_x, play, start_time, gain)
             # Draw cart
            cart.draw_cart(screen)
            # Draw pendulum
            pendulum.draw_pendulum(screen,visual_gain)
    

        elif keys[pygame.K_o]:
            start_x = cal.calibrate_origin()
            # pendulum.position_history[0] = start_x
            # pendulum.position_history[1] = start_x
            print(cal.origin)
        
        elif keys[pygame.K_l]:
            cal.calibrate_xy_plane_left()
            print(cal.left_scale)

        elif keys[pygame.K_r]:
            cal.calibrate_xy_plane_right()
            print(cal.right_scale)
            cal.calibration_end == True


        elif keys[pygame.K_SPACE]:
    
            cal.right_scale = -cal.left_scale
            screen_position, raw_position = cal.get_current_point()
            pendulum.raw_position_history.append(raw_position)
            pendulum.update_pendulum_state(screen_position, play, start_time, gain)

            # Draw cart
            cart.draw_cart(screen)
            # Draw pendulum
            pendulum.draw_pendulum(screen, visual_gain)
            now = pygame.time.get_ticks() / 1000.0
            print("final system displayed - ", pygame.time.get_ticks())



        # elif keys[pygame.K_SPACE]:
        #     now = datetime.datetime.now()
        #     cal.left_scale = -cal.right_scale
        #     pendulum.update_pendulum_state(cal.get_dist())
        #      # Draw cart
        #     cart.draw_cart(screen)
        #     # Draw pendulum
        #     pendulum.draw_pendulum(screen)
        else:
            
             # Draw cart
            cart.draw_cart(screen)
            # Draw pendulum
            pendulum.draw_pendulum(screen, visual_gain)

        
        pendulum.update_pendulum_state(pendulum.cart.cart_x, play, start_time, gain)
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
        # pendulum.update_pendulum_state()
        # Draw cart
        cart.draw_cart(screen)

        # # Draw pendulum
        pendulum.draw_pendulum(screen, visual_gain)



        pygame.display.flip()

    # Quit Pygame
    pygame.quit()
    print(now)
    

    time, angle, position, raw_hand_position = pendulum.get_history()
    # np.save("/Users/rishitabanerjee/Desktop/Project stuff/BrainMachineInterfaces/Experiment/run3", [x,y])
    plt.plot(time,angle)
    plt.plot(time, np.zeros(len(time)))
    plt.xlabel('Time')
    plt.ylabel('Pendulum angle')
    plt.title('Time response')
    plt.ylim = (-5,5)
    plt.show()
    plt.plot(time,position[1:-1])
    plt.show()
   
    
    

    # np.save("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_30/time", np.array(time))
    # np.save("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_30/position", np.array(position))
    # np.save("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_30/angle", np.array(angle))
    # np.save("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_30/raw_position", np.array(raw_hand_position))

    path = '/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/participant_' + str(participant_id) + '/info.txt'
    f = open(path, "a")
    f.write('\n' + str(int((1*score))))
    f.close()
    sys.exit()

if __name__ == "__main__":
    main()

# /Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot_exp_data/Tsep_6_g1_2_g2_20