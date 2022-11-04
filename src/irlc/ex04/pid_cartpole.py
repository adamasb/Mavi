import numpy as np
import matplotlib.pyplot as plt
np.random.seed(32)
from irlc import Agent, savepdf
from irlc.ex04.pid import PID
from irlc.ex01.agent import train
from irlc import VideoMonitor
from irlc.ex04.model_cartpole import GymThetaCartpoleEnvironment

class PIDCartpoleAgent(Agent):
    def __init__(self, env, dt, Kp=1.0, Ki=0.0, Kd=0.0, target=0, balance_to_x0=True):
        """ Balance_to_x0 = True implies the agent should also try to get the cartpole to x=0 (i.e. center).
        If balance_to_x0 = False implies it is good enough for the agent to get the cart upright.
        """
        self.pid = PID(dt=dt, Kp = Kp, Ki=Ki, Kd=Kd, target=target)
        self.balance_to_x0 = balance_to_x0
        super().__init__(env)

    def pi(self, x, t=None): #!f
        """ Compute action using self.pid. You have to think about the inputs as they will depend on
        whether balance_to_x0 is true or not.  """
        u = -self.pid.pi(x[-2] + (.12*x[0]+.02*x[1] if self.balance_to_x0 else 0) )
        u = np.clip(u, -self.env.max_force, self.env.max_force) # Clip max torque.
        return u


def get_offbalance_cart(waiting_steps=30):
    env = GymThetaCartpoleEnvironment(Tmax=10)
    env = VideoMonitor(env)
    env.reset()
    env.state[0] = 0
    env.state[1] = 0
    env.state[2] = 0  # Upright, but leaning slightly to one side.
    env.state[3] = 0
    for _ in range(waiting_steps):  # Simulate the environment for 30 steps to get things out of balance.
        env.step(1)
    return env

def plot_trajectory(trajectory):
    t = trajectory
    plt.plot(t.time, t.state[:,2], label="Stick angle $\\theta$" )
    plt.plot(t.time, t.state[:,0], label="Cart location $x$")
    plt.xlabel("Time")
    plt.legend()

if __name__ == "__main__":
    """
    First task: Bring the balance upright from a slightly off-center position. 
    For this task, we do not care about the x-position, only the angle theta which should be 0 (upright)
    """
    env = get_offbalance_cart(20)
    agent = PIDCartpoleAgent(env, dt=env.dt, Kp=120, Ki=0, Kd=10, balance_to_x0=False) #!b agent = PIDCartpoleAgent(env, env.dt, ...) #!b Define your agent here (including parameters)
    _, trajectories = train(env, agent, num_episodes=1, return_trajectory=True, reset=False)  # Note reset=False to maintain initial conditions.
    plot_trajectory(trajectories[0])
    env.close()
    savepdf("pid_cartpoleA")
    plt.show()

    """
    Second task: We will now also try to bring the cart towards x=0.
    """
    env = get_offbalance_cart(20)
    agent = PIDCartpoleAgent(env, dt=env.dt, Kp=120, Ki=0, Kd=10, balance_to_x0=True) #!b agent = PIDCartpoleAgent(env, env.dt, ...) #!b Define your agent here (including parameters)
    _, trajectories = train(env, agent, num_episodes=1, return_trajectory=True, reset=False)  # Note reset=False to maintain initial conditions.
    plot_trajectory(trajectories[0])
    env.close()
    savepdf("pid_cartpoleB")
    plt.show()

    """
    Third task: Bring the cart upright theta=0 and to the center x=0, but starting from a more challenging position. 
    """
    env = get_offbalance_cart(30)
    agent = PIDCartpoleAgent(env, dt=env.dt, Kp=120, Ki=0, Kd=10, balance_to_x0=True) #!b agent = PIDCartpoleAgent(env, env.dt, ...) #!b Define your agent here (including parameters)
    _, trajectories = train(env, agent, num_episodes=1, return_trajectory=True, reset=False)  # Note reset=False to maintain initial conditions.
    plot_trajectory(trajectories[0])
    env.close()
    savepdf("pid_cartpoleC")
    plt.show()