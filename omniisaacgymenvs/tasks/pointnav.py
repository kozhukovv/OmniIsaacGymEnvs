#############################################
#            OWN IMPLEMENTATION             #
#############################################

import math
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.pointnav_basic import Potato
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

import torch

class NavigationTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)
        self._max_episode_length = 500
        
        # these must be defined in the task class
        self._num_observations = 4
        self._num_actions = 1
        
        # call the parent class constructor to initialize key RL variables
        RLTask.__init__(self, name, env)
        
    def update_config(self, sim_config):
        
        # extract task config from main config dictionary
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # parse task config parameters
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._robot_positions = torch.tensor([0.0, 0.0, 2.0])

        # reset and actions related variables
        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
    
    def set_up_scene(self, scene) -> None:
        # first create a single environment
        self.get_cartpole()

        # call the parent class to clone the single environment
        super().set_up_scene(scene)

        # construct an ArticulationView object to hold our collection of environments
        self._cartpoles = ArticulationView(
            prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view", reset_xform_properties=False
        )

        # register the ArticulationView object to the world, so that it can be initialized
        scene.add(self._cartpoles)

    def get_potato(self):
        # add a single robot to the stage
        potato = Potato(
            prim_path=self.default_zero_env_path + "/Potato", name="Potato", translation=self._cartpole_positions
        )

        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(
            "Potato", get_prim_at_path(potato.prim_path), self._sim_config.parse_actor_config("Potato")
        )

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply randomized joint positions and velocities to environments
        indices = env_ids.to(dtype=torch.int32)
        self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        self._cartpoles.set_joint_velocities(dof_vel, indices=indices)

        # reset the reset buffer and progress buffer after applying reset
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0