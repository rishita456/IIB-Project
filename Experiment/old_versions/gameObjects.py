import pygame
import sys
import math
import config
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import math


class Pendulum():

    def __init__(self):

       # Initial state
        self.pendulum_angle = 0
        self.pendulum_angular_velocity = 0
        self.pendulum_angle_3D = 0
        self.pendulum_angular_velocity_3D = 0
        self.cart = None
        self.previous_time = 0
        self.angle_history = []
        self.time_history = []
        self.position_history = []
        self.c = 0
    
    def set_cart(self, cart):
        self.cart = cart

    def draw_pendulum(self, screen, gain):
        # Calculate pendulum position
        pendulum_center_x = self.cart.cart_x + config.CART_WIDTH // 2
        pendulum_center_y = config.HEIGHT//2 - config.CART_HEIGHT
        pendulum_end_x = pendulum_center_x + config.PEN_LENGTH * math.sin(gain*self.pendulum_angle)
        pendulum_end_y = pendulum_center_y - config.PEN_LENGTH * math.cos(gain*self.pendulum_angle)

        # Draw pendulum
        pygame.draw.line(screen, config.PENDULUM_COLOR, (pendulum_center_x, pendulum_center_y), (pendulum_end_x, pendulum_end_y), config.PENDULUM_WIDTH)

    def draw_pendulum_3d(self):

        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        glColor(config.PENDULUM_COLOR)
        
        glPushMatrix()
        glRotatef(-math.degrees(config.ANGLE_GAIN*self.pendulum_angle_3D), 0, 1, 0)
        
        
        gluCylinder(quadric, config.PENDULUM_RADIUS_3D, config.PENDULUM_RADIUS_3D, config.PENDULUM_HEIGHT_3D, 20, 10)
        glTranslatef(self.cart.cart_x_3D,0, config.CART_DEPTH_3D)
        
        glPopMatrix()

    def update_pendulum_state(self, force, play, start_time, gain):
        
        current_time = pygame.time.get_ticks() / 1000.0
        elapsed_time = current_time - self.previous_time

        # Physics calculations
        pendulum_acceleration = (config.G * np.sin(self.pendulum_angle) + np.cos(self.pendulum_angle) * (-force - config.M1 * config.PEN_LENGTH * self.pendulum_angular_velocity**2 * np.sin(self.pendulum_angle))) / (config.PEN_LENGTH * (4/3 - config.M1 * np.cos(self.pendulum_angle)**2 / (config.M1 + config.M2)))
        cart_acceleration = (force + config.M1 * config.PEN_LENGTH * (self.pendulum_angular_velocity**2 * np.sin(self.pendulum_angle) - pendulum_acceleration * np.cos(self.pendulum_angle))) / (config.M1 + config.M2)
        # cart_acceleration = 
        # pendulum_acceleration = (config.G*np.sin(self.pendulum_angle)/config.PEN_LENGTH) - (cart_acceleration*np.cos(self.pendulum_angle)/config.PEN_LENGTH)

        self.pendulum_angular_velocity += pendulum_acceleration * elapsed_time
        self.pendulum_angle += gain*self.pendulum_angular_velocity * elapsed_time
        self.cart.cart_velocity += cart_acceleration * elapsed_time
        self.cart.cart_x += config.POSITION_GAIN*self.cart.cart_velocity * elapsed_time

        
        
        if play:
            if self.c ==1:
                self.pendulum_angle = 0.05
            self.time_history.append(current_time - start_time)
            self.angle_history.append(math.degrees(self.pendulum_angle))
            

        # Physics calculations 3D
        pendulum_acceleration_3D = (config.G * np.sin(self.pendulum_angle_3D) + np.cos(self.pendulum_angle_3D) * (-force - config.M1 * config.PENDULUM_HEIGHT_3D * self.pendulum_angular_velocity_3D**2 * np.sin(self.pendulum_angle_3D))) / (config.PENDULUM_HEIGHT_3D * (4/3 - config.M1 * np.cos(self.pendulum_angle_3D)**2 / (config.M1 + config.M2)))
        cart_acceleration_3D = (force + config.M1 * config.PENDULUM_HEIGHT_3D * (self.pendulum_angular_velocity_3D**2 * np.sin(self.pendulum_angle_3D) - pendulum_acceleration_3D * np.cos(self.pendulum_angle_3D))) / (config.M1 + config.M2)

        self.pendulum_angular_velocity_3D += pendulum_acceleration_3D * elapsed_time
        self.pendulum_angle_3D += self.pendulum_angular_velocity_3D * elapsed_time
        self.cart.cart_velocity_3D += cart_acceleration_3D * elapsed_time
        self.cart.cart_x_3D += self.cart.cart_velocity_3D * elapsed_time


        # pendulum_acceleration_3D = (
        #     config.G * math.sin(self.pendulum_angle) - 
        #     math.cos(self.pendulum_angle) * (self.cart.cart_velocity ** 2) / config.PENDULUM_HEIGHT_3D 
        # )

        # self.pendulum_angular_velocity_3D += pendulum_acceleration_3D * elapsed_time
        # self.pendulum_angle_3D += -(self.cart.cart_velocity/config.PENDULUM_HEIGHT_3D ) + self.pendulum_angular_velocity_3D * elapsed_time


        print('system state updated - ', pygame.time.get_ticks())
        self.previous_time = current_time

    def get_history(self):
        return self.time_history, self.angle_history


class Cart():

    def __init__(self):

       # Initial state
        self.cart_x = (config.WIDTH - config.CART_WIDTH) // 2
        self.cart_x_3D = 0
        self.cart_velocity = 0
        self.previous_time = 0
        self.cart_velocity_3D = 0

    def draw_cart(self, screen):
        
        # Draw cart
        pygame.draw.rect(screen, config.CART_COLOR, (self.cart_x, config.HEIGHT//2 - config.CART_HEIGHT, config.CART_WIDTH, config.CART_HEIGHT))

    def draw_cart_3d(self):

        glColor(0, 1, 0)
        glTranslate(self.cart_x_3D, 0, 0)
        glPushMatrix()
        glScalef(config.CART_WIDTH_3D, config.CART_HEIGHT_3D, config.CART_DEPTH_3D)
        glutSolidCube(1)
        glPopMatrix()

    def update_cart(self):

        current_time = pygame.time.get_ticks() / 1000.0
        elapsed_time = current_time - self.previous_time

        # Update cart position
        
        self.cart_x += config.SENSITIVITY_2D*self.cart_velocity * elapsed_time
        self.cart_x_3D += config.SENSITIVITY_3D*self.cart_velocity * elapsed_time

        self.previous_time = current_time

   

