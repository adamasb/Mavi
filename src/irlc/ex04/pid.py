from irlc import savepdf
import numpy as np
import matplotlib.pyplot as plt
from irlc.ex04.locomotive import LocomotiveEnvironment

class PID:
    def __init__(self, dt, Kp, Ki, Kd, target):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt          # discretization time
        self.target = target  # target, in our case just a number.
        self.I = 0            # Internal variables for integral/derivative terms; use these or define your own.
        self.e_prior = 0      # Previous value of the error. Used in the derivative term. Remember to update it in the pi-function.

    def reset(self):
        self.I = 0
        self.e_prior = 0

    def pi(self, x):
        """
        Policy for the PID class. x is always a scalar (float) and the output u is a scalar.
        Should implement \nref{c9pid}

        :param x: Input state (float)
        :return: Action to take (float)
        """
        e = self.target - x #!b
        self.I = self.I + e * self.dt
        u = self.Kp * e + self.Ki * self.I + self.Kd * (e - self.e_prior)/self.dt
        self.e_prior = e #!b Compute u here.
        return u


def pid_explicit():
    env = LocomotiveEnvironment(m=70, slope=0, dt=0.05, Tmax=15)
    pid = PID(dt=0.05, Kp=40, Kd=0, Ki=0, target=0)
    x = [env.reset()]
    for _ in range(200): # Simulate for 200 steps, i.e. 0.05 * 200 seconds.
        x_cur = x[-1] # x is the last state [position, velocity]. Note that you only need to pass position to your PID controller.
        u = pid.pi(x_cur[0]) #!b #!b Compute action here using the pid class.
        u = np.clip(u, -100, 100) # clip actions.
        xp_, reward, done, _ = env.step(u)
        x.append(xp_)

    x = np.stack(x)
    plt.plot(x[:,0], 'k-', label="PID state trajectory")
    savepdf("pid_basic")
    plt.show()

if __name__ == "__main__":
    pid_explicit()
