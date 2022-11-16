

"""Example of using RLlib's debug callbacks.
Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

from typing import Dict, Tuple
import argparse
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.",
)
parser.add_argument("--stop-iters", type=int, default=2000)


class MyCallbacks(DefaultCallbacks):
    wandb = None
    gradients = 0 
    def on_algorithm_init(self, *args, algorithm=None):
        print("Initializing the callback logger..")
        # algorithm.config
        # This gets us a base environment of sorts.
        MyCallbacks.env = algorithm.env_creator(algorithm.config['env_config'])



        if self.wandb is None:
            import wandb
            wandb.init(project="my-test-project")
            wandb.config = {
                "learning_rate": 0.001,
                "epochs": 100,
                "batch_size": 128
            }
            # for loss in range(10):
            #     wandb.log({"loss": np.sqrt(loss)})
            # wandb.watch(model, log_freq=100)
            self.wandb = wandb

        a = 234
        pass


    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        # print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))
        episode.user_data["pole_angles"] = []
        episode.hist_data["pole_angles"] = []

    def on_evaluate_start(self, *args, **kwargs):
        print("on eval start")
        pass


    def on_evaluate_end(self, *args, **kwargs):
        print("on eval end")
        pass



    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        # pole_angle = abs(episode.last_observation_for()[2])
        # raw_angle = abs(episode.last_raw_obs_for()[2])
        # assert pole_angle == raw_angle
        pole_angle = 0
        episode.user_data["pole_angles"].append(pole_angle)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        pole_angle = np.mean(episode.user_data["pole_angles"])
        # print(
        #     "episode {} (env-idx={}) ended with length {} and pole "
        #     "angles {}".format(
        #         episode.episode_id, env_index, episode.length, pole_angle
        #     )
        # )
        episode.custom_metrics["pole_angle"] = pole_angle
        episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        pass
        # print("on_sample_end, returned sample batch of size {}".format(samples.count))

    def evaluation_call(self, a3policy):
        # evaluation callback using hacks, etc.
        # a3policy.model
        state = MyCallbacks.env.reset()
        model = a3policy.model
        stats = model.value_function_for_env_state(state)
        vv = stats['v']
        vv_norm = (vv-vv.min()) / (0.0001 + vv.max() - vv.min())
        p = stats['p']
        rin = stats['rin']
        rout = stats['rout']

        image_array = [state, vv, vv_norm, p, rin, rout]
        image_array = [i*255 for i in image_array]
        import PIL
        images = [PIL.Image.fromarray(image.astype(np.uint8)) for image in image_array]
        # images = self.wandb.Image(image_array, caption="Top: Output, Bottom: Input")
        self.wandb.log({"Layout (Green=agent) | V | V_norm | p | rin | rout ":  [self.wandb.Image(image) for image in images]})
        # import torchvision
        pass

    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
        print("Learned..")
        result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
        assert False
        print(
            "policy.learn_on_batch() result: {} -> sum actions: {}".format(
                policy, result["sum_actions_in_train_batch"]
            )
        )

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # print("Algorithm.train() result: {} -> {} episodes".format(algorithm, result["episodes_this_iter"]))
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True
        lstats = result['info']['learner']
        if 'default_policy' in lstats:
            lstats = lstats['default_policy']['learner_stats']
            # print("> CALLBACK HAD NON-ZERO INFO", lstats)
            self.gradients = self.gradients + 1
            print(" logging gradients")
            self.wandb.log("Training results with learner info", self.gradients)
            assert False
            
        else:
            # print("no l stats")
            lstats = {}

        hstats = {k: np.mean(v) for k, v in result['hist_stats'].items()}

        stats = lstats | hstats |  result['timers'] | result['counters']
        # print(stats)
        # print(hstats)
        self.wandb.log( stats)

        # print("ON TRAIN", result['info'])



    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, Tuple[Policy, SampleBatch]],
        **kwargs
    ):
        # print("postprocessed {} steps".format(postprocessed_batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1
