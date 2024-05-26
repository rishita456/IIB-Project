import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
from gameObjects import Pendulum, Cart
from calibrate import Calibrate
from multiprocessing import shared_memory
import pandas as pd
import numpy as np
import quaternions_exp
import config

def update_view(camera_position, camera_rotation):
    glLoadIdentity()
    glTranslatef(*camera_position)
    glRotatef(camera_rotation[0], 1, 0, 0)
    glRotatef(camera_rotation[1], 0, 1, 0)

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

def main():
    
    # Initialize Pygame
    pygame.init()
    display = (800, 600)

    # Initial camera position and orientation
    camera_position = [0, 0, 0]
    camera_rotation = [0, 0, 0]
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Balancing Inverted Pendulum")


    # Set up the perspective
    glMatrixMode(GL_PROJECTION)
    gluPerspective(120, (4/3), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -40)
    glRotatef(-90, 1, 0, 0)
    glMatrixMode(GL_MODELVIEW)
    
    clock = pygame.time.Clock()

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

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()

        if keys[K_1]:
            camera_rotation[1] += 0.1
            update_view(camera_position, camera_rotation)
        elif keys[K_2]:
            camera_rotation[1] -= 0.1
            update_view(camera_position, camera_rotation)
        elif keys[K_3]:
            camera_rotation[0] += 0.1
            update_view(camera_position, camera_rotation)
        elif keys[K_4]:
            camera_rotation[0] -= 0.1
            update_view(camera_position, camera_rotation)
        elif keys[K_5]:
            camera_position[2] += 0.05
            update_view(camera_position, camera_rotation)
        elif keys[K_6]:
            camera_position[2] -= 0.05
            update_view(camera_position, camera_rotation)

        elif keys[K_o]:
            cal.calibrate_origin()
            print(cal.origin)
        
        elif keys[K_l]:
            cal.calibrate_xy_plane_left()
            print(cal.left_scale)

        elif keys[K_r]:
            cal.calibrate_xy_plane_right()
            print(cal.right_scale)


        elif keys[K_LEFT]:
            pendulum.update_pendulum_state(-0.001)
                        
        elif keys[K_RIGHT]:
            pendulum.update_pendulum_state(0.001)
            

        elif keys[K_SPACE]:
            cart.cart_velocity = cal.get_dist()/config.SENSITIVITY_3D
            cart.update_cart()
            
            
            # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            # cart.draw_cart_3d()
            # pendulum.draw_pendulum_3d()
            # glutSwapBuffers()

        # else:
        #     cart.cart_velocity = 0
        #     cart.update_cart()
        
        pendulum.update_pendulum_state(0)


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        cart.draw_cart_3d()
        pendulum.draw_pendulum_3d()
        glutSwapBuffers()
        pygame.display.flip()
    

if __name__ == "__main__":
    main()
