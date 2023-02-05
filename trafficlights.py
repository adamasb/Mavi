import ray
from ray.rllib.policy.policy import PolicySpec
import gym
import numpy as np
import random

# from ray.rllib.examples.env.multi_agent import MultiAgentTrafficEnv


#import the MultiAgentTrafficEnv 

# from MultiAgentTrafficEnv import MultiAgentTrafficEnv

"""the multiagenttrafficenv doesnt seem to actually exist..."""


# Env, in which all agents (whose IDs are entirely determined by the env
# itself via the returned multi-agent obs/reward/dones-dicts) step
# simultaneously.
env = MultiAgentTrafficEnv(num_cars=2, num_traffic_lights=1)

# Observations are a dict mapping agent names to their obs. Only those
# agents' names that require actions in the next call to `step()` should
# be present in the returned observation dict (here: all, as we always step
# simultaneously).
print(env.reset())
# ... {
# ...   "car_1": [[...]],
# ...   "car_2": [[...]],
# ...   "traffic_light_1": [[...]],
# ... }

# In the following call to `step`, actions should be provided for each
# agent that returned an observation before:
new_obs, rewards, dones, infos = env.step(
    actions={"car_1": ..., "car_2": ..., "traffic_light_1": ...})

# Similarly, new_obs, rewards, dones, etc. also become dicts.
print(rewards)
# ... {"car_1": 3, "car_2": -1, "traffic_light_1": 0}

# Individual agents can early exit; The entire episode is done when
# dones["__all__"] = True.
print(dones)
# ... {"car_2": True, "__all__": False}





algo = pg.PGAgent(env="my_multiagent_env", config={
    "multiagent": {
        "policies": {
            # Use the PolicySpec namedtuple to specify an individual policy:
            "car1": PolicySpec(
                policy_class=None,  # infer automatically from Algorithm
                observation_space=None,  # infer automatically from env
                action_space=None,  # infer automatically from env
                config={"gamma": 0.85},  # use main config plus <- this override here
                ),  # alternatively, simply do: `PolicySpec(config={"gamma": 0.85})`

            # Deprecated way: Tuple specifying class, obs-/action-spaces,
            # config-overrides for each policy as a tuple.
            # If class is None -> Uses Algorithm's default policy class.
            "car2": (None, car_obs_space, car_act_space, {"gamma": 0.99}),

            # New way: Use PolicySpec() with keywords: `policy_class`,
            # `observation_space`, `action_space`, `config`.
            "traffic_light": PolicySpec(
                observation_space=tl_obs_space,  # special obs space for lights?
                action_space=tl_act_space,  # special action space for lights?
                ),
        },
        "policy_mapping_fn":
            lambda agent_id, episode, worker, **kwargs:
                "traffic_light"  # Traffic lights are always controlled by this policy
                if agent_id.startswith("traffic_light_")
                else random.choice(["car1", "car2"])  # Randomly choose from car policies
    },
})

while True:
    print(algo.train())