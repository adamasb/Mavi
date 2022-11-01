import numpy as np
from irlc import savepdf
from irlc.ex04.pid import PID
from irlc import Agent

class PIDCarAgent(Agent):
    def __init__(self, env, v_target=0.5, use_both_x5_x3=True):
        """
        Define two pid controllers: One for the angle, the other for the velocity.

        self.pid_angle = PID(dt=self.discrete_model.dt, Kp=x, ...)
        self.pid_velocity = PID(dt=self.discrete_model.dt, Kp=z, ...)

        I did not use Kd/Ki, however you need to think a little about the targets.
        """
        self.pid_angle = PID(dt=env.discrete_model.dt, Kp=1.0, Ki=0, Kd=0, target=0) #!b # self.pid_angle = ...
        self.pid_velocity = PID(dt=env.discrete_model.dt, Kp=1.5, Ki=0, Kd=0, target=v_target) #!b Define PID controllers here.
        self.use_both_x5_x3 = use_both_x5_x3 # Using both x3+x5 seems to make it a little easier to get a quick lap time, but you can just use x5 to begin with.
        super().__init__(env)

    def pi(self, x, t=None):
        """
        Call PID controller. The steering angle controller should initially just be based on
        x[5] (distance to the centerline), but you can later experiment with a linear combination of x5 and x3 as input.

        To control the velociy, you should use x[0], the velocity of the car in the direction of the car.

        Remember to start out with a low value of v_target, then tune the controller and look at the animation.

        """

        xx = x[5] + x[3] if self.use_both_x5_x3 else x[5] #!b
        u = np.asarray([self.pid_angle.pi(xx), self.pid_velocity.pi(x[0])]) #!b Compute action here. No clipping necesary.
        return u


if __name__ == "__main__":
    from irlc.ex01.agent import train
    from irlc.utils.video_monitor import VideoMonitor
    from irlc.car.car_model import CarEnvironment
    import matplotlib.pyplot as plt

    env = CarEnvironment(noise_scale=0,Tmax=30, max_laps=1)
    env = VideoMonitor(env)
    agent = PIDCarAgent(env, v_target=1, use_both_x5_x3=True) # I recommend lowering v_target to make the problem simpler.

    stats, trajectories = train(env, agent, num_episodes=1, return_trajectory=True)
    env.close()
    t = trajectories[0]
    plt.clf()
    plt.plot(t.state[:,0], label="velocity" )
    plt.plot(t.state[:,5], label="s (distance to center)" )
    plt.xlabel("Time/seconds")
    plt.legend()
    savepdf("pid_car_agent")
    plt.show()
