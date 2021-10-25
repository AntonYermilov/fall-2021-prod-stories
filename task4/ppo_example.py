import os
import sys
import shutil
from gym import spaces

import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


sys.path.append(os.path.abspath("mapgen"))
os.environ["PYTHONPATH"] = os.path.abspath("mapgen")
from mapgen import Dungeon


class ModifiedDungeon(Dungeon):
    """Use this class to change the behavior of the original env (e.g. remove the trajectory from observation, like here)"""
    def __init__(
        self,
        width=20,
        height=20,
        max_rooms=3,
        min_room_xy=5,
        max_room_xy=12,
        max_steps: int = 400,
        observation_size=11,
        vision_radius=5,
    ):
        super().__init__(
            width=width,
            height=height,
            max_rooms=max_rooms,
            min_room_xy=min_room_xy,
            max_room_xy=max_room_xy,
            observation_size=observation_size,
            vision_radius=vision_radius,
            max_steps=max_steps
        )

    def step(self, action):
        observation, reward, done, info = super().step(action)
        reward = 1.0 * info['is_new']
        return observation, reward, done, info
    

if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    tune.register_env("Dungeon", lambda config: ModifiedDungeon(**config))

    CHECKPOINT_ROOT = "tmp/ppo/dungeon"
    shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

    ray_results = os.getenv("HOME") + "/ray_results1/"
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["log_level"] = "INFO"
    config["framework"] = "torch"
    config["env"] = "Dungeon"
    config["env_config"] = {
        "width": 20,
        "height": 20,
        "max_rooms": 3,
        "min_room_xy": 5,
        "max_room_xy": 10,
        "observation_size": 11,
        "vision_radius": 5
    }

    config["model"] = {
        "conv_filters": [
            [16, (3, 3), 2],
            [32, (3, 3), 2],
            [32, (3, 3), 1],
        ],
        "post_fcnet_hiddens": [32],
        "post_fcnet_activation": "relu",
        "vf_share_layers": False,
    }

    config["num_workers"] = 0
    config["exploration_config"] = {
        "type": "Curiosity",  # <- Use the Curiosity module for exploring.
        "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
        "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
        "feature_dim": 288,  # Dimensionality of the generated feature vectors.
        # Setup of the feature net (used to encode observations into feature (latent) vectors).
        "feature_net_config": {
            "fcnet_hiddens": [],
            "fcnet_activation": "relu",
            "conv_filters": [
                [16, (3, 3), 2],
                [32, (3, 3), 2],
                [32, (3, 3), 1],
            ],
        },
        "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
        "inverse_net_activation": "relu",  # Activation of the "inverse" model.
        "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
        "forward_net_activation": "relu",  # Activation of the "forward" model.
        "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
        # Specify, which exploration sub-type to use (usually, the algo's "default"
        # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
        "sub_exploration": {
            "type": "StochasticSampling",
        }
    }

    config["rollout_fragment_length"] = 100
    config["entropy_coeff"] = 0.1
    config["lambda"] = 0.95
    config["vf_loss_coeff"] = 1.0

    agent = ppo.PPOTrainer(config)

    N_ITER = 20
    s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

    writer = SummaryWriter()

    for n in range(N_ITER):
        result = agent.train()
        file_name = agent.save(CHECKPOINT_ROOT)

        writer.add_scalar('entropy_loss', result["info"]["learner"]["default_policy"]["learner_stats"]["entropy"],
                          result['timesteps_total'])
        writer.add_scalar('vf_loss', result["info"]["learner"]["default_policy"]["learner_stats"]["vf_loss"],
                          result['timesteps_total'])
        writer.add_scalar('policy_loss', result["info"]["learner"]["default_policy"]["learner_stats"]["policy_loss"],
                          result['timesteps_total'])
        writer.add_scalar('episode_reward_mean', result['episode_reward_mean'], result['timesteps_total'])

        print(s.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            file_name
        ))

        # sample trajectory
        if (n + 1) % 5 == 0:
            env = Dungeon(20, 20, 3, min_room_xy=5, max_room_xy=10, vision_radius=5)
            obs = env.reset()
            Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).save('tmp.png')

            frames = []

            for _ in range(500):
                action = agent.compute_single_action(obs)

                frame = Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).quantize()
                frames.append(frame)

                obs, reward, done, info = env.step(action)
                if done:
                    break

            print('Explore rate: ', info["total_explored"] / info["total_cells"])
            writer.add_scalar('total_explored', info["total_explored"], result['timesteps_total'])
            writer.add_scalar('total_cells', info["total_cells"], result['timesteps_total'])
            writer.add_scalar('avg_explored_per_step', info["avg_explored_per_step"], result['timesteps_total'])

            frames[0].save(f"out.gif", save_all=True, append_images=frames[1:], loop=0, duration=1000/60)
