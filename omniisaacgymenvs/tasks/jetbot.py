# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from fileinput import close
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.controllers.differential_controller import DifferentialController

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import GeometryPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationActions
from omni.isaac.range_sensor import _range_sensor
from omni.isaac.core.utils.rotations import quat_to_euler_angles
import omni.kit.commands
from pxr import Gf

import numpy as np
import torch
import math


class JetbotTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._jetbot_positions = torch.tensor([0.0, 0.0, 0.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 1000
        
        self.collision_range = 0.22 # 0.11 or 0.20

        self.ranges_count = 360
        self._num_observations = self.ranges_count + 2 # +2 for angle and distance (polar coords)
        self._num_actions = 1

        self._diff_controller = DifferentialController(name="simple_control",wheel_radius=0.03, wheel_base=0.1125)


        RLTask.__init__(self, name, env)

        # init tensors that need to be set to correct device
        self.prev_goal_distance = torch.zeros(self._num_envs).to(self._device)
        self.prev_heading = torch.zeros(self._num_envs).to(self._device)
        self.target_position = torch.tensor([1.5, 1.5, 0.0]).to(self._device)
        return

    def set_up_scene(self, scene) -> None:
        """Add prims to the scene and add views for interracting with them. Views are useful to interract with multiple prims at once."""
        self.add_prims_to_stage(scene)
        super().set_up_scene(scene)
        self._jetbots = ArticulationView(prim_paths_expr="/World/envs/.*/jetbot_with_lidar/jetbot_with_lidar", name="jetbot_view")
        self._targets = GeometryPrimView(prim_paths_expr="/World/envs/.*/target_cube", name="target_view")
        scene.add(self._jetbots)

    def add_prims_to_stage(self, scene):
        # applies articulation settings from the task configuration yaml file
        #self._sim_config.apply_articulation_settings("Cartpole", get_prim_at_path(cartpole.prim_path), self._sim_config.parse_actor_config("Cartpole"))
        
        asset_path = "/home/eetu/jetbot_isaac/content/jetbot_with_lidar.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path=self.default_zero_env_path + "/jetbot_with_lidar")

        add_reference_to_stage(
            usd_path="/home/eetu/jetbot_isaac/content/obstacles.usd",
            prim_path=self.default_zero_env_path + "/obstacles"
        )

        result, lidar = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path=self.default_zero_env_path + "/jetbot_with_lidar/jetbot_with_lidar/chassis/Lidar",
            parent=None,
            min_range=0.1,
            max_range=20.0,     
            draw_points=False,
            draw_lines=False,
            horizontal_fov=360.0,
            vertical_fov=30.0,
            horizontal_resolution=360/self.ranges_count, # 5
            vertical_resolution=4.0,
            rotation_rate=0.0,
            high_lod=False,
            yaw_offset=0.0,
            enable_semantics=False,
        )
        lidar.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, 0.0, 0.11))

        add_reference_to_stage(
            usd_path="/home/eetu/jetbot_isaac/content/target_cube.usd",
            prim_path=self.default_zero_env_path + "/target_cube",
        )


    def get_observations(self) -> dict:
        """Return lidar ranges and polar coordinates as observations to RL agent."""
        self.ranges = torch.zeros((self._num_envs, self.ranges_count)).to(self._device)

        for i in range(self._num_envs):
            np_ranges = self.lidarInterface.get_linear_depth_data(self._lidarpaths[i]).squeeze()
            self.ranges[i] = torch.tensor(np_ranges)
        
        #print(self.ranges.shape)

        self.positions, self.rotations = self._jetbots.get_world_poses()
        self.target_positions, _ = self._targets.get_world_poses()
        yaws = []
        for rot in self.rotations:
            yaws.append(quat_to_euler_angles(rot)[2])
        yaws = torch.tensor(yaws).to(self._device)

        #print("position", self.position)
        #print("yaw", yaws)
        #print("target pos", self.target_pos)
        goal_angles = torch.atan2(self.target_positions[:,1] - self.positions[:,1], self.target_positions[:,0] - self.positions[:,0])

        self.headings = goal_angles - yaws
        self.headings = torch.where(self.headings > math.pi, self.headings - 2 * math.pi, self.headings)
        self.headings = torch.where(self.headings < -math.pi, self.headings + 2 * math.pi, self.headings)

        self.goal_distances = torch.linalg.norm(self.positions - self.target_positions, dim=1).to(self._device)

        obs = torch.hstack((self.ranges, self.headings.unsqueeze(1), self.goal_distances.unsqueeze(1)))
        self.obs_buf[:] = obs

        observations = {
            self._jetbots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        """Perform actions to move the robot."""
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        indices = torch.arange(self._jetbots.count, dtype=torch.int32, device=self._device)
        # self._cartpoles.set_joint_efforts(forces, indices=indices)
        
        controls = torch.zeros((self._num_envs,2))
        #print(actions)
        for i in range(self._num_envs):
            controls[i] = self._diff_controller.forward([0.2, 2*actions[i].item()])

        #self._jetbots.apply_action(ArticulationActions(joint_velocities=controls))
        #joint_velocities = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]) * 10
        self._jetbots.set_joint_velocities(controls)
        #self._jetbots.apply_action(self._diff_controller.forward(np.array([0.2, 0.0])))
        
        # may not be needed for obs and actions randomization
        #if self._dr_randomizer.randomize:
            #print("randomize reset env ids", reset_env_ids)
            #omni.replicator.isaac.physics_view.step_randomization(reset_env_ids)

    def reset_idx(self, env_ids):
        """Resetting the environment at the beginning of episode."""
        num_resets = len(env_ids)

        self.goal_reached = torch.zeros(self._num_envs, device=self._device)
        self.collisions = torch.zeros(self._num_envs, device=self._device)

        # apply resets
        root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
        root_vel = torch.zeros((num_resets, 6), device=self._device)
        #  + torch.tensor([-0.2, -0.3, 0.0], device=self._device)
        self._jetbots.set_world_poses(root_pos, root_rot, indices=env_ids)
        self._jetbots.set_velocities(root_vel, indices=env_ids)

        target_pos = self.initial_target_pos[env_ids] + self.target_position
        
        self._targets.set_world_poses(target_pos, indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        """This is run when first starting the simulation before first episode."""
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
        jetbot_paths = self._jetbots.prim_paths
        self._lidarpaths = [path + "/chassis/Lidar" for path in jetbot_paths]

        # get some initial poses
        self.initial_root_pos, self.initial_root_rot = self._jetbots.get_world_poses()
        self.initial_target_pos, _ = self._targets.get_world_poses()
        #self.target_pos, _ = self._targets.get_world_poses()

        # randomize all envs
        indices = torch.arange(self._jetbots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

        if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self)

    def calculate_metrics(self) -> None:
        """Calculate rewards for the RL agent."""
        rewards = torch.zeros_like(self.rew_buf)

        closest_ranges, indices = torch.min(self.ranges, 1)
        self.collisions = torch.where(closest_ranges < self.collision_range, 1.0, 0.0).to(self._device)

        closer_to_goal = torch.where(self.goal_distances < self.prev_goal_distance, 1, -1)
        self.prev_goal_distance = self.goal_distances
        self.goal_reached = torch.where(self.goal_distances < 0.1, 1, 0).to(self._device)

        closer_to_heading = torch.where(torch.abs(self.headings) < torch.abs(self.prev_heading), 1, 0)
        correct_heading = torch.where(torch.abs(self.headings) < 0.2, 1, 0)
        heading_bonus = torch.where(torch.logical_or(correct_heading, closer_to_heading), 1, -1)

        self.prev_heading = self.headings

        rewards -= self.collisions * 20
        rewards += closer_to_goal * 0.005
        #rewards += closer_to_heading * 0.01
        rewards += heading_bonus * 0.005
        rewards += self.goal_reached * 20

        #print("collisions", self.collisions[0].item(), "closer to goal", closer_to_goal[0].item(), "heading bonus", heading_bonus[0].item(), "heading", self.headings[0].item())

        self.rew_buf[:] = rewards

    def is_done(self) -> None:
        """Flags the environnments in which the episode should end."""
        #self.reset_buf[:] = torch.zeros(self._num_envs)
        resets = torch.where(self.progress_buf >= self._max_episode_length - 1, 1.0, self.reset_buf.double())
        resets = torch.where(self.collisions.bool(), 1.0, resets.double())
        resets = torch.where(self.goal_reached.bool(), 1.0, resets.double())
        self.reset_buf[:] = resets
