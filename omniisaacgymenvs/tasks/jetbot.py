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


from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.controllers.differential_controller import DifferentialController

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import GeometryPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.objects import VisualCuboid
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
        self.target_position = np.array([1.5, 1.5, 0.0])

        self.ranges_count = 72
        self._num_observations = self.ranges_count + 1  # +1 for angle
        self._num_actions = 1

        self._diff_controller = DifferentialController(name="simple_control",wheel_radius=0.03, wheel_base=0.1125)


        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        self.add_prims_to_stage(scene)
        super().set_up_scene(scene)
        self._jetbots = ArticulationView(prim_paths_expr="/World/envs/.*/jetbot_with_lidar/jetbot_with_lidar", name="jetbot_view")
        self._targets = GeometryPrimView(prim_paths_expr="/World/envs/.*/target_cube", name="target_view")
        scene.add(self._jetbots)

    def add_prims_to_stage(self, scene):
        #cartpole = Cartpole(prim_path=self.default_zero_env_path + "/Cartpole", name="Cartpole", translation=self._cartpole_positions)
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
            min_range=0.15,
            max_range=20.0,
            draw_points=False,
            draw_lines=True,
            horizontal_fov=360.0,
            vertical_fov=30.0,
            horizontal_resolution=5.0,
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

        # scene.add(VisualCuboid(
        #         prim_path=self.default_zero_env_path + "/target_cube", # The prim path of the cube in the USD stage
        #         name="target_cube", # The unique name used to retrieve the object from the scene later on
        #         position=self.target_position, # Using the current stage units which is in meters by default.
        #         size=np.array([0.1, 0.1, 0.1]), # most arguments accept mainly numpy arrays.
        #         color=np.array([1.0, 0, 0]), # RGB channels, going from 0-1
        # ))

    def get_observations(self) -> dict:
        #dof_pos = self._cartpoles.get_joint_positions(clone=False)
        #dof_vel = self._cartpoles.get_joint_velocities(clone=False)

        #cart_pos = dof_pos[:, self._cart_dof_idx]
        #cart_vel = dof_vel[:, self._cart_dof_idx]
        #pole_pos = dof_pos[:, self._pole_dof_idx]
        #pole_vel = dof_vel[:, self._pole_dof_idx]

        #self.obs_buf[:, 0] = cart_pos
        #self.obs_buf[:, 1] = cart_vel
        #self.obs_buf[:, 2] = pole_pos
        #self.obs_buf[:, 3] = pole_vel

        self.ranges = torch.zeros((self._num_envs, self.ranges_count))

        for i in range(self._num_envs):
            np_ranges = self.lidarInterface.get_linear_depth_data(self._lidarpaths[i]).squeeze()
            self.ranges[i] = torch.tensor(np_ranges)
        
        #print(self.ranges.shape)

        self.obs_buf[:, :self.ranges_count] = self.ranges

        self.positions, rotations = self._jetbots.get_world_poses()
        yaws = []
        for rot in rotations:
            yaws.append(quat_to_euler_angles(rot)[2])

        #print("position", self.position)
        #print("yaw", yaws)
        #print("target pos", self.target_pos)
        #goal_angles = np.arctan2(self.target_position[1] - self.position[1], self.target_position[0] - self.position[0])

        # heading = goal_angle - yaw
        # if heading > math.pi:
        #     heading -= 2 * math.pi

        # elif heading < -math.pi:
        #     heading += 2 * math.pi
        
        # #print("heading", heading)

        # #print(np.hstack((jetbot_pos, jetbot_vel)))
        # self.obs = np.hstack((self.ranges.squeeze(), yaw))

        observations = {
            self._jetbots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)
        

        # forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        # forces[:, self._cart_dof_idx] = self._max_push_effort * actions[:, 0]

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

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        #self.current_step = 0
        #self.goal_reached = False
        #self.collision = False

        # apply resets
        root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
        root_vel = torch.zeros((num_resets, 6), device=self._device)

        self._jetbots.set_world_poses(root_pos, root_rot, indices=env_ids)
        self._jetbots.set_velocities(root_vel, indices=env_ids)


        target_pos = self.initial_target_pos[env_ids] + torch.tensor([1.5, 1.5, 0], device=self._device)
        
        self._targets.set_world_poses(target_pos, indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        #self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        #self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
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

    def calculate_metrics(self) -> None:
        #cart_pos = self.obs_buf[:, 0]
        #cart_vel = self.obs_buf[:, 1]
        #pole_angle = self.obs_buf[:, 2]
        #pole_vel = self.obs_buf[:, 3]

        #reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        #reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        #reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)
        reward = 0.0

        self.rew_buf[:] = reward

    def is_done(self) -> None:
        #cart_pos = self.obs_buf[:, 0]
        #pole_pos = self.obs_buf[:, 2]

        #resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        #resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        #resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        
        #self.reset_buf[:] = torch.zeros(self._num_envs)
        resets = torch.where(self.progress_buf >= self._max_episode_length - 1, 1.0, 0.0)
        #print(resets)
        return resets
