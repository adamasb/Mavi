from irlc import utils
import os
from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment
from irlc.ex04.model_harmonic import HarmonicOscilatorModel
import numpy as np
import pyglet
from pyglet.shapes import Rectangle, Triangle
from irlc.utils.pyglet_rendering import PygletViewer, CameraGroup
from gym.spaces import Box


class LocomotiveModel(HarmonicOscilatorModel):
    viewer = None
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }

    def __init__(self, m=1., slope=0.0, target=0):
        """
        Slope is the uphill slope of the train (in degrees). E.g. slope=15 makes it harder for the engine.

        :param m:
        :param slope:
        """
        self.target = target
        self.slope = slope
        super().__init__(m=m, k=0., drag=-np.sin(slope/360*2*np.pi) * m * 9.82)
        self.action_space = Box(low=np.asarray([-100.]), high=np.asarray([100.]), dtype=np.float)

    def reset(self):
        return [-1, 0]

    def render(self, x, mode="human"):
        """ Initialize a viewer and update the states. """
        if self.viewer is None:
            self.viewer = LocomotiveViewer(self)
        self.viewer.locomotive_group.translate(x=x[0]*self.viewer.scale, y=0)
        return self.viewer.render(return_rgb_array=mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


class DiscreteLocomotiveModel(DiscretizedModel):
    def __init__(self, *args, dt=0.1, **kwargs):
        model = LocomotiveModel(*args, **kwargs)
        super().__init__(model=model, dt=dt)
        self.cost = model.cost.discretize(self, dt=dt)


class LocomotiveEnvironment(ContiniousTimeEnvironment):
    def __init__(self, *args, dt=0.1, Tmax=5, **kwargs):
        model = DiscreteLocomotiveModel(*args, dt=dt, **kwargs)
        self.dt = model.dt
        super().__init__(discrete_model=model, Tmax=Tmax)


class LocomotiveViewer(PygletViewer):
    def __init__(self, train):
        # self.train = train
        batch = pyglet.graphics.Batch()
        cgroup = CameraGroup(order=0)
        width = 1100
        scale = width / 4
        dw = scale * 0.1
        red = (200, 40, 40)

        super().__init__(screen_width=width, xmin=-width/2, xmax=width/2, ymin=-width/5, ymax=width/5)
        self.track = Rectangle(-2*scale, -dw/2, 4*scale, dw/2, color=(int(.7*255),)*3, batch=batch, group=cgroup)
        self.trg = Triangle(train.target * scale-dw/2, -dw/2, train.target*scale, 0, train.target*scale+dw/2, -dw/2, color=red, batch=batch, group=cgroup)

        image = pyglet.image.load(os.path.dirname(utils.__file__) + "/locomotive.png")
        image.anchor_x = image.width // 2
        lgroup = CameraGroup(order=10, pg=cgroup)

        self.locomotive = pyglet.sprite.Sprite(image, x=0, y=0, batch=batch, group=lgroup)
        dw2 = 2
        self.ltarg = Rectangle(-dw2, 0, dw2*2, scale*0.2, color=red, batch=batch, group=CameraGroup(pg=lgroup, order=11))

        self.scale = scale
        self.locomotive.scale_x = self.locomotive.scale_y = 0.5

        cgroup.rotate(train.slope/180*np.pi)
        self.locomotive_group = lgroup
        self.batch = batch

    def draw(self):
        self.batch.draw()
