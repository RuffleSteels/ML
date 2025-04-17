from __future__ import division, print_function
from math import sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CartPole:
    def __init__(self, physics):
        self.physics = physics
        self.mass_cart = 1.0
        self.mass_pole = 0.3
        self.mass = self.mass_cart + self.mass_pole
        self.length = 0.7 # actually half the pole length
        self.pole_mass_length = self.mass_pole * self.length
    def init_display(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-0.5, 3.5)
        self.cart_patch = patches.Rectangle((0, -0.25), 0.8, 0.25, facecolor='cyan')
        self.base_patch = patches.Rectangle((0, -0.5), 0.02, 0.25, facecolor='r')
        self.pole_line, = self.ax.plot([], [], lw=2)
        self.ax.add_patch(self.cart_patch)
        self.ax.add_patch(self.base_patch)

    def show_cart(self, state_tuple, pause_time=0.05):
        x, x_dot, theta, theta_dot = state_tuple
        pole_x = x + 4*self.length * sin(theta)
        pole_y = 4*self.length * cos(theta)
        self.pole_line.set_data([x, pole_x], [0, pole_y])
        self.cart_patch.set_xy((x - 0.4, -0.25))
        self.base_patch.set_xy((x - 0.01, -0.5))
        self.ax.set_title(f"x: {x:.2f}, θ: {theta:.2f}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(pause_time)

    def simulate(self, action, state_tuple):
        x, x_dot, theta, theta_dot = state_tuple
        costheta, sintheta = cos(theta), sin(theta)
        # costheta, sintheta = cos(theta * 180 / pi), sin(theta * 180 / pi)

        # calculate force based on action
        force = self.physics.force_mag if action > 0 else (-1 * self.physics.force_mag)

        # intermediate calculation
        temp = (force + self.pole_mass_length * theta_dot * theta_dot * sintheta) / self.mass
        theta_acc = (self.physics.gravity * sintheta - temp * costheta) / (self.length * (4/3 - self.mass_pole * costheta * costheta / self.mass))

        x_acc = temp - self.pole_mass_length * theta_acc * costheta / self.mass

        # return new state variable using Euler's method
        new_x = x + self.physics.tau * x_dot
        new_x_dot = x_dot + self.physics.tau * x_acc
        new_theta = theta + self.physics.tau * theta_dot
        new_theta_dot = theta_dot + self.physics.tau * theta_acc
        new_state = (new_x, new_x_dot, new_theta, new_theta_dot)

        return new_state

    def sample_states(self, n):
        x_range = (-2.4, 2.4)
        x_dot_range = (-1.5, 1.5)
        theta_range = (-12 * np.pi / 180, 12 * np.pi / 180)  # radians
        theta_dot_range = (-2.0, 2.0)

        states = []
        for _ in range(n):
            x = np.random.uniform(*x_range)
            x_dot = np.random.uniform(*x_dot_range)
            theta = np.random.uniform(*theta_range)
            theta_dot = np.random.uniform(*theta_dot_range)
            states.append((x, x_dot, theta, theta_dot))

        return states

class Physics:
    gravity = 9.8
    force_mag = 10.0
    tau = 0.01
