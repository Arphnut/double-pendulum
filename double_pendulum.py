#!/usr/bin/python3
"""
Created at 14:39:39 on samedi 22-01-2022

Author: Etienne

Name:
-----
double_pendulum.py

Description:
------------
This script is about display some Pendulums and some Double Pendulums, using matplotlib animation.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.integrate import solve_ivp

G = 9.81  # The gravitation constant


class QueuePosition(list):
    """
    A class that complement the list one to make it easier for us to append 'n' elements.
    It is a list containing 'self.size' queue (FIFO).
    It is implemented to append element to each of the list at the same time.

    Attributes:
    -----------
    size (int): The number of list in the QueuePosition element
    max_length (int): The maximum length of QueuePosition
    """

    def __init__(self, size=4, max_length=10 ** 4):
        super(QueuePosition, self).__init__()
        self.size = size
        for _ in range(self.size):
            super(QueuePosition, self).append([])
        self.max_length = max_length

    def append(self, el):
        """
        Append the i^th element of 'el' to the end of the i^th queue of 'self'.
        Because it is a queue, if the max_number of element is reached, the first element is deleted
        Parameters:
        -----------
        el (list): A list with self.size element
        """
        if len(el) != self.size:
            raise ValueError("The length of 'el' ({}) should be the same one as self.size ({})")
        if len(self[0]) < self.max_length:
            for i in range(self.size):
                self[i].append(el[i])  # self[i] is a true List and thus it won't create a loop
        else:
            for i in range(self.size):
                del self[i][0]
                self[i].append(el[i])

    def add(self, listadd):
        """
        This function add to the QueuePosition a list of element.

        Parameters:
        -----------
        listadd (array): An array of size (self.size, ...)
        """
        assert len(listadd) == self.size
        for i in range(len(listadd[0])):
            self.append(listadd[:, i])


class Pendulum:
    """
    A Class describing the behaviour of a pendulum.

    Attributes:
    -----------
    theta (array): A 2-sized array describing the angle and momentum of the pendulum at the initial time
    mass (float): The mass of the pendulum that will be used in the moving equation (doesn't change the behaviour of a single pendulum.
    length (float): The length of the rod of the pendulum.
    timelapse (array): Determine the different time for which the position of the pendulum will be computed when 'moving'.
    list_theta (QueuePosition): Contain the behaviour of the Pendulum through its angles with time.
    positions (QueuePosition): Contain the behaviour of the Pendulum through its position with time."""

    def __init__(
        self,
        theta=[0, 0],
        mass=1,
        length=1,
        timelapse=np.linspace(0, 10, 1001),
        max_length=10 ** 4,
    ):
        self.theta = np.array(theta)
        self.mass = mass
        self.length = length
        self.timelapse = timelapse
        self.list_theta = QueuePosition(2, max_length)
        self.list_theta.append(self.theta)
        self.positions = QueuePosition(2, max_length)
        self.positions.append(self.get_position())

    def derive(self, theta=None):
        """
        Compute the derivative of the angle theta.
        """
        if theta is None:
            theta = self.theta
        return np.array([theta[1], -G / self.length * np.sin(theta[0])])

    def move(self, epsilon=0.01):
        """
        Make the angle move from on degree.
        So far it is done using very simple euler scheme.
        A better function using scipy's function solve_ivp exist.
        """
        self.theta = self.theta + epsilon * self.derive()
        self.list_theta.append(self.theta)
        self.positions.append(self.get_position())

    def get_position(self, theta=None):
        """
        Compute a position according to an angle "theta"
        """
        if theta is None:
            theta = self.theta
        x, y = -self.length * np.sin(theta[0]), -self.length * np.cos(theta[0])
        return [x, y]

    def update_positions(self):
        """
        Update all the positions using list_theta.
        """
        pos = np.array(self.list_theta)
        self.positions = [self.get_position(pos[:, i]) for i in range(len(pos[0]))]

    # Begin with "exact" solution using scipy, and then try EUler and RK4 scheme...
    def deriv_for_solver(self):
        """
        Return the derivative of theta, adapted for the solve_ivp function of scipy.
        """

        def deriv_theta(t, y):
            return self.derive(y)

        return deriv_theta

    def move_exact(self):
        """
        Move the pendulum and update the positions solving a differential equation using scipy.
        """
        derivative = self.deriv_for_solver()
        results = solve_ivp(derivative, (0.0, 10.0), self.theta, t_eval=self.timelapse)
        self.list_theta.add(results.y)
        self.update_positions()
        self.theta = [self.list_theta[0][-1], self.list_theta[1][-1]]


class DoublePendulum:
    """
    A double pendulum class
    """

    # TODO there is a prblem in the behaviour of the pendulum. Look for Errors in 'derive'

    def __init__(
        self,
        theta=[0, 0, 0, 0],
        mass=[1, 1],
        length=[1, 1],
        timelapse=np.linspace(0, 10, 1001),
        max_length=10 ** 4,
    ):
        self.theta = np.array(theta)  # FOrm theta1, theta2, dtheta1, dtheta2
        self.mass = np.array(mass)
        self.length = np.array(length)
        self.timelapse = timelapse
        self.list_theta = QueuePosition(4, max_length)
        self.list_theta.append(self.theta)
        self.positions = QueuePosition(4, max_length)
        self.positions.append(self.get_position())

    def derive(self, theta=None):
        if theta is None:
            theta = self.theta

        cost = np.cos(theta[0] - theta[1])
        sint = np.sin(theta[0] - theta[1])
        num1 = (
            self.mass[1] * G * np.sin(theta[1]) * cost
            - self.mass[1]
            * sint
            * (self.length[0] * theta[2] ** 2 * cost + self.length[1] * theta[3] ** 2)
            - np.sum(self.mass) * G * np.sin(theta[0])
        )
        denom1 = np.sum(self.mass) * self.length[0] - self.mass[1] * self.length[0] * cost ** 2
        num2 = (
            np.sum(self.mass)
            * (
                self.length[0] * theta[2] ** 2 * sint
                - G * np.sin(theta[1])
                + G * np.sin(theta[0]) * cost
            )
            + self.mass[1] * self.length[1] * theta[3] ** 2 * sint * cost
        )

        denom2 = np.sum(self.mass) * self.length[1] - self.mass[1] * self.length[1] * cost ** 2
        return np.array([theta[2], theta[3], num1 / denom1, num2 / denom2])

    def move(self, epsilon=0.01):
        self.theta = self.theta + epsilon * self.derive()
        self.list_theta.append(self.theta)
        self.positions.append(self.get_position())

    def get_position(self, theta=None):
        if theta is None:
            theta = self.theta
        x1, y1 = -self.length[0] * np.sin(theta[0]), -self.length[0] * np.cos(theta[0])
        x2, y2 = x1 - self.length[1] * np.sin(theta[1]), y1 - self.length[1] * np.cos(theta[1])
        return [x1, y1, x2, y2]

    def update_positions(self):
        """
        Update all the positions using list_theta
        """
        pos = np.array(self.list_theta)
        self.positions = [self.get_position(pos[:, i]) for i in range(len(pos[0]))]

    def deriv_for_solver(self):
        """
        Return the derivative of theta, adapted for the solve_ivp function of scipy.
        """

        def deriv_theta(t, y):
            return self.derive(y)

        return deriv_theta

    def move_exact(self):
        """
        Move the double pendulum and update the positions solving a differential equation using scipy
        """
        derivative = self.deriv_for_solver()
        results = solve_ivp(derivative, (0.0, 10.0), self.theta, t_eval=self.timelapse)
        self.list_theta.add(results.y)
        self.update_positions()
        self.theta = [
            self.list_theta[0][-1],
            self.list_theta[1][-1],
            self.list_theta[2][-1],
            self.list_theta[3][-1],
        ]


class DisplayPendulum:
    """
    A class to display a Pendulum.

    Attributes:
    -----------
    pendulums (list): A list of Pendulum
    display_mem (int): The display memory, that is the number of passed points that will be displayed on the window.
    timelapse (array): Determine the different time for which the position of the pendulums will be computed when 'moving';
    nb_pend (int): The number of pendulum to display
    fig (matplotlib.figure.Figure): The figure all will be ploted on.
    ax (matplotlib.axes._subplots.AxesSubplot): The axes describing all points.
    center (matplotlib.lines.Line2D): Plot the center of the pendulums.
    points_pendulums (matplotlib.lines.Line2D): Plot the position of all pendulums
    line_rods (list): A list of Line2D representing the rods of all pendulums.
    line_pendulums (list): A list of Line2D representing the past position of the pendulums
    animation (matplotlib.animation.FuncAnimation): An animation showing the behaviour of the pendulums.
    """

    def __init__(self, pendulums=None, display_mem=50, timelapse=np.linspace(0, 10, 1001)):
        self.timelapse = timelapse
        self.display_mem = display_mem
        if pendulums is None:
            self.pendulums = [Pendulum(timelapse=self.timelapse)]
        else:
            self.pendulums = pendulums
        self.nb_pend = len(self.pendulums)

    def move(self):
        """
        Move all the pendulums
        """
        for pend in self.pendulums:
            pend.move_exact()

    def get_frame(self, frame):
        """
        Get the image for the position of the pendulums at time 'frame'.
        """
        positions = np.array([pend.positions for pend in self.pendulums])
        frame_end = frame + self.display_mem
        for i in range(self.nb_pend):
            self.line_rods[i].set_data(
                [0, positions[i, frame_end, 0]], [0, positions[i, frame_end, 1]]
            )
            self.line_pendulums[i].set_data(
                positions[i, frame:frame_end, 0], positions[i, frame:frame_end, 1]
            )
        self.center.set_data([0], [0])
        self.points_pendulums.set_data(positions[:, frame_end, 0], positions[:, frame_end, 1])
        return self.center, self.points_pendulums, *self.line_rods, *self.line_pendulums

    def create_animation(self):
        """
        Create an animation showing the behaviour of the pendulums
        """
        self.fig, self.ax = plt.subplots()
        ax_lim = 1.2 * self.pendulums[0].length
        self.ax.set_xlim([-ax_lim, ax_lim])
        self.ax.set_ylim([-ax_lim, ax_lim])
        self.ax.set_aspect("equal")
        # ls=none means no linestyle
        (self.center,) = self.ax.plot([0], [0], ls="none", marker="o", color="black")
        (self.points_pendulums,) = self.ax.plot([], [], ls="none", marker="o", color="blue")
        self.line_rods = [self.ax.plot([], [], color="black")[0] for i in range(self.nb_pend)]
        self.line_pendulums = [
            plt.plot([], [], color="blue", alpha=0.2)[0] for i in range(self.nb_pend)
        ]
        self.animation = anim.FuncAnimation(
            fig=self.fig,
            func=self.get_frame,
            frames=len(self.timelapse) - self.display_mem,
            blit=True,
            interval=20,
        )
        plt.show()


class DisplayDoublePendulum:
    """
    A class to display Double Pendulums.

    Attributes:
    -----------
    pendulums (list): A list of DoublePendulum
    display_mem (int): The display memory, that is the number of passed points that will be displayed on the window.
    timelapse (array): Determine the different time for which the position of the pendulums will be computed when 'moving';
    nb_pend (int): The number of pendulum to display
    fig (matplotlib.figure.Figure): The figure all will be ploted on.
    ax (matplotlib.axes._subplots.AxesSubplot): The axes describing all points.
    center (matplotlib.lines.Line2D): Plot the center of the pendulums.
    points_pendulums (matplotlib.lines.Line2D): Plot the position of all pendulums
    line_rods (list): A list of Line2D representing the rods of all pendulums.
    line_pendulums (list): A list of Line2D representing the past position of the pendulums
    animation (matplotlib.animation.FuncAnimation): An animation showing the behaviour of the pendulums.
    Display the double pendulum.
    """

    def __init__(self, pendulums=None, display_mem=50, timelapse=np.linspace(0, 10, 1001)):
        self.timelapse = timelapse
        self.display_mem = display_mem
        if pendulums is None:
            self.pendulums = [DoublePendulum(timelapse=self.timelapse)]
        else:
            self.pendulums = pendulums
        self.nb_pend = len(self.pendulums)

    def move(self):
        """
        Move all the pendulums
        """
        for pend in self.pendulums:
            pend.move_exact()

    def get_frame(self, frame):
        """
        Get the image for the position of the pendulums at time 'frame'
        """
        positions = np.array([pend.positions for pend in self.pendulums])
        frame_end = frame + self.display_mem
        for i in range(self.nb_pend):
            self.line_rods1[i].set_data(
                [0, positions[i, frame_end, 0]], [0, positions[i, frame_end, 1]]
            )
            self.line_rods2[i].set_data(
                [positions[i, frame_end, 0], positions[i, frame_end, 2]],
                [positions[i, frame_end, 1], positions[i, frame_end, 3]],
            )
            self.line_pendulums[i].set_data(
                positions[i, frame:frame_end, 2], positions[i, frame:frame_end, 3]
            )
        self.center.set_data([0], [0])
        self.points_pendulums1.set_data(positions[:, frame_end, 0], positions[:, frame_end, 1])
        self.points_pendulums2.set_data(positions[:, frame_end, 2], positions[:, frame_end, 3])
        return (
            self.center,
            self.points_pendulums1,
            self.points_pendulums2,
            *self.line_rods1,
            *self.line_rods2,
            *self.line_pendulums,
        )

    def create_animation(self):
        """
        Create an animation showing the behavious of the pendulums
        """
        self.fig, self.ax = plt.subplots()
        ax_lim = 1.2 * np.sum(self.pendulums[0].length)
        self.ax.set_xlim([-ax_lim, ax_lim])
        self.ax.set_ylim([-ax_lim, ax_lim])
        self.ax.set_aspect("equal")
        # ls=none means no linestyle
        (self.center,) = self.ax.plot([0], [0], ls="none", marker="o", color="black")
        (self.points_pendulums1,) = self.ax.plot([], [], ls="none", marker="o", color="blue")
        (self.points_pendulums2,) = self.ax.plot([], [], ls="none", marker="o", color="red")
        self.line_rods1 = [self.ax.plot([], [], color="black")[0] for i in range(self.nb_pend)]
        self.line_rods2 = [self.ax.plot([], [], color="black")[0] for i in range(self.nb_pend)]
        self.line_pendulums = [
            plt.plot([], [], color="red", alpha=0.2)[0] for i in range(self.nb_pend)
        ]
        self.animation = anim.FuncAnimation(
            fig=self.fig,
            func=self.get_frame,
            frames=len(self.timelapse) - self.display_mem,
            blit=True,
            interval=20,
        )
        plt.show()
