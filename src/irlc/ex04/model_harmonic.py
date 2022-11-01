from irlc.ex04.model_linear_quadratic import LinearQuadraticModel
from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment
import numpy as np
import pyglet
from pyglet.shapes import Rectangle, Circle, Line
from pyglet.graphics import OrderedGroup
from irlc.utils.pyglet_rendering import PygletViewer, CameraGroup


"""
   Simulate a Harmonic oscillator governed by equations:

   d^2 x1 / dt^2 = -k/m x1 + u(x1, t)

   where x1 is the position and u is our externally applied force (the control)
   k is the spring constant and m is the mass. See:

   https://en.wikipedia.org/wiki/Simple_harmonic_motion#Dynamics

   for more details.
   In the code, we will re-write the equations as:

   dx/dt = f(x, u),   u = u_fun(x, t)

   where x = [x1,x2] is now a vector and f is a function of x and the current control.
   here, x1 is the position (same as x in the first equation) and x2 is the velocity.

   The function should return ts, xs, C

   where ts is the N time points t_0, ..., t_{N-1}, xs is a corresponding list [ ..., [x_1(t_k),x_2(t_k)], ...] and C is the cost.
   """

class HarmonicOscilatorModel(LinearQuadraticModel):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }
    """
    See: https://books.google.dk/books?id=tXZDAAAAQBAJ&pg=PA147&lpg=PA147&dq=boeing+747+flight+0.322+model+longitudinal+flight&source=bl&ots=L2RpjCAWiZ&sig=ACfU3U2m0JsiHmUorwyq5REcOj2nlxZkuA&hl=en&sa=X&ved=2ahUKEwir7L3i6o3qAhWpl4sKHQV6CdcQ6AEwAHoECAoQAQ#v=onepage&q=boeing%20747%20flight%200.322%20model%20longitudinal%20flight&f=false
    """
    def __init__(self, k=1., m=1., drag=0.0, Q=None, R=None):
        self.k = k
        self.m = m
        A = [[0, 1],
             [-k/m, 0]]

        B = [[0], [1/m]]
        d = [[0], [drag/m]]

        A, B, d = np.asarray(A), np.asarray(B), np.asarray(d)
        if Q is None:
            Q = np.eye(2)
        if R is None:
            R = np.eye(1)
        self.viewer = None
        super().__init__(A=A, B=B, Q=Q, R=R, d=d)

    def reset(self): # Return the initial state. In this case x = 1 and dx/dt = 0.
        return [1, 0]

    def render(self, x, mode="human"):
        """ Render the environment. You don't have to understand this code.  """
        if self.viewer is None:
            self.viewer = HarmonicViewer()
        self.viewer.update(x)
        return self.viewer.render(return_rgb_array=mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


class DiscreteHarmonicOscilatorModel(DiscretizedModel): #!s=a
    def __init__(self, dt=0.1, discretization_method=None, **kwargs):
        model = HarmonicOscilatorModel(**kwargs)
        super().__init__(model=model, dt=dt, discretization_method=discretization_method)
        self.cost = model.cost.discretize(self, dt=dt) #!s


class HarmonicOscilatorEnvironment(ContiniousTimeEnvironment): #!s=c
    def __init__(self, Tmax=80, supersample_trajectory=False, **kwargs):
        model = DiscreteHarmonicOscilatorModel(**kwargs)
        self.dt = model.dt
        super().__init__(discrete_model=model, Tmax=Tmax, supersample_trajectory=supersample_trajectory) #!s


class HarmonicViewer(PygletViewer):
    def __init__(self):
        width = 1100
        self.scale = width / 6
        dw = self.scale * 0.1
        super().__init__(screen_width=width, xmin=-width/2, xmax=width/2, ymin=-width/5, ymax=width/5)
        batch = self.batch
        cgroup = CameraGroup(order=1)
        self.base = Rectangle(-dw/2, -dw/2, dw, dw, color=(0, 0, 0), batch=batch)
        self.obj = Circle(0, 0, radius=dw, color=(0,0,0), batch=batch, group=cgroup)
        self.obj2 = Circle(0, 0, radius=dw*0.9, color=(int(.7*255),)*3, batch=batch, group=cgroup)
        xx = np.linspace(0,1)
        coil = np.sin(xx*2*np.pi*5)*self.scale*0.1
        self.coil = [Line(xx[i], coil[i], xx[i+1], coil[i+1], width=2, color=(int(0.3*255),)*3,
                          batch=batch, group=OrderedGroup(0)) for i in range(coil.size - 1)]
        self.cgroup = cgroup

    def update(self, x):
        self.cgroup.translate(x[0]*self.scale, 0)
        xx = np.linspace(0, 1)
        for i in range(len(self.coil)-1):
            self.coil[i].x, self.coil[i].x2 = xx[i]*x[0]*self.scale, xx[i+1]*x[0]*self.scale

# An agent that takes null (zero) actions:

from irlc import Agent
class NullAgent(Agent):
    def pi(self, x, k=None):
        return np.asarray([0])

if __name__ == "__main__":
    from irlc import VideoMonitor, train
    env = VideoMonitor(HarmonicOscilatorEnvironment())
    train(env, NullAgent(env), num_episodes=1, max_steps=200)
    env.close()