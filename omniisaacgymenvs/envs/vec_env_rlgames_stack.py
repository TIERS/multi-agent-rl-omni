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


from omni.isaac.gym.vec_env import VecEnvBase

import torch
import numpy as np

from datetime import datetime
from collections import deque
from gym.spaces import Box


# VecEnv Wrapper for RL training with stacking frames
class VecEnvRLGamesStack(VecEnvBase):

    def _process_data(self):
        self._obs = torch.clamp(self._obs, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
        self._rew = self._rew.to(self._task.rl_device).clone()
        self._states = torch.clamp(self._states, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
        self._resets = self._resets.to(self._task.rl_device).clone()
        self._extras = self._extras.copy()

    def set_task(
        self, task, backend="numpy", sim_params=None, init_sim=True
    ) -> None:
        super().set_task(task, backend, sim_params, init_sim)

        self.num_states = self._task.num_states
        self.state_space = self._task.state_space

        # how many frames to stack
        self.num_stack = 4
        empty_frames = np.zeros((self.num_stack, self.num_envs, self.observation_space.shape[0]))
        self.frames = deque(empty_frames, maxlen=self.num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], self.num_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], self.num_stack, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )
        print("self.observation_space.shape", self.observation_space.shape)
        #self.observation_space.shape = (self.num_stack, self.observation_space.shape[0])

    def step(self, actions):
        actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).to(self._task.device).clone()

        if self._task.randomize_actions:
            actions = self._task._dr_randomizer.apply_actions_randomization(actions=actions, reset_buf=self._task.reset_buf)

        self._task.pre_physics_step(actions)
        
        for _ in range(self._task.control_frequency_inv):
            self._world.step(render=self._render)
            self.sim_frame_count += 1

        self._obs, self._rew, self._resets, self._extras = self._task.post_physics_step()

        if self._task.randomize_observations:
            self._obs = self._task._dr_randomizer.apply_observations_randomization(observations=self._obs, reset_buf=self._task.reset_buf)

        self._states = self._task.get_states()
        self._process_data()


        self.frames.append(self._obs.cpu().numpy())
        
        #print("obs", self._obs, self._obs.shape)
        #print("squeeze", self._obs.squeeze(), self._obs.squeeze().shape)
        #print("frames", np.array(self.frames), np.array(self.frames).shape)
        
        # swap axes and put into tensor
        tens = torch.tensor(np.swapaxes(self.frames, 0, 1), device=self._task.device)

        #print("tens", tens, tens.shape)
        #input()
        
        obs_dict = {"obs": tens, "states": self._states}

        return obs_dict, self._rew, self._resets, self._extras

    def reset(self):
        """ Resets the task and applies default zero actions to recompute observations and states. """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Running RL reset")

        self._task.reset()
        actions = torch.zeros((self.num_envs, self._task.num_actions), device=self._task.device)
        obs_dict, _, _, _ = self.step(actions)

        # TODO need to handle the resets so that when some env resets, only its stack changes, not every envs stack 
        # it works when its done wrong though but may not work later in some cases so would be safer to fix it now
        # work in progress code :
        # frames_np = np.array(self.frames)
        # obs_np = self._obs.cpu().numpy()
        

        # #frames_np[:, resets_np] = obs_np[resets_np]
        # print(frames_np, frames_np.shape)
        # print(obs_np, obs_np.shape)
        #print(resets.cpu().numpy())
        #print(resets_np, resets_np.shape)
        # print(obs_np[resets_np], obs_np[resets_np].shape)
        # print(frames_np[:, resets_np], frames_np[:, resets_np].shape)
        # print(np.repeat(np.expand_dims(obs_np[resets_np], axis=0), 4, axis=0), np.repeat(np.expand_dims(obs_np[resets_np], axis=0), 4, axis=0).shape)
        # # print(self._obs, self._obs.shape)
        
        # input()
        # frames_np[:, resets_np] = np.repeat(np.expand_dims(obs_np[resets_np], axis=0), 4, axis=0)
        # print(frames_np, frames_np.shape)
        # input()


        #frames_np[:, resets_np]
        
        # #[self.frames.append(obs_dict["obs"].squeeze().cpu().numpy()) for _ in range(self.num_stack)]
        # # swap axes and put into tensor
        # if self._num_envs == 1:
        #     frames_np = np.expand_dims(self.frames, 0)
        # else:
        #     frames_np = np.swapaxes(self.frames, 0, 1)
        # tens = torch.tensor(frames_np, device=self._task.device)#.unsqueeze(0)
        
        for _ in range(self.num_stack):
            self.frames.append(self._obs.cpu().numpy())
        
        # swap axes and put into tensor
        tens = torch.tensor(np.swapaxes(self.frames, 0, 1), device=self._task.device)

        #print("tens", tens, tens.shape)

        #input()
        
        obs_dict["obs"] = tens
        return obs_dict
