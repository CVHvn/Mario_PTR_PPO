from PIL import Image
from collections import deque
from datetime import datetime
from pathlib import Path
import copy
import cv2
import imageio
import numpy as np
import random, os
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
#import multiprocessing as mp
from torchvision import transforms as T

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from src.environment import *
from src.memory import *
from src.model import *

class Agent():
    def __init__(self, world, stage, action_type, envs, num_envs, state_dim, action_dim, save_dir, save_model_step,
                 save_figure_step, learn_step, total_step_or_episode, total_step, total_episode, model, old_model,
                 gamma, learning_rate, entropy_coef, V_coef, max_grad_norm,
                 clip_param, batch_size, num_epoch, is_normalize_advantage, V_loss_type, target_kl, gae_lambda,
                 per_buffer_size, per_eps, per_alpha, per_beta, eps_marg, off_epoch,
                 device):
        self.world = world
        self.stage = stage
        self.action_type = action_type

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.learn_step = learn_step
        self.total_step_or_episode = total_step_or_episode
        self.total_step = total_step
        self.total_episode = total_episode

        self.current_step = 0
        self.current_episode = 0

        self.save_model_step = save_model_step
        self.save_figure_step = save_figure_step

        self.device = device
        self.save_dir = save_dir

        self.num_envs = num_envs
        self.envs = envs
        self.model = model.to(self.device)
        self.old_model = old_model.to(self.device)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.V_coef = V_coef
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.off_epoch = off_epoch

        self.per_buffer_size = per_buffer_size
        self.per_eps = per_eps
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.eps_marg = eps_marg
        self.per = PrioritizedReplayBuffer(self.learn_step, self.state_dim, self.action_dim, self.per_buffer_size, eps=self.per_eps, alpha=self.per_alpha, beta=self.per_beta)
        self.memory = Memory(self.num_envs)
        self.is_completed = False

        self.env = None
        self.max_test_score = -1e9
        self.is_normalize_advantage = is_normalize_advantage
        self.V_loss_type = V_loss_type
        self.target_kl = target_kl
        self.gae_lambda = gae_lambda

        # I just log 1000 lastest update and print it to log.
        self.V_loss = np.zeros((1000,)).reshape(-1)
        self.P_loss = np.zeros((1000,)).reshape(-1)
        self.E_loss = np.zeros((1000,)).reshape(-1)
        self.approx_kl_divs = np.zeros((1000,)).reshape(-1)
        self.total_loss = np.zeros((1000,)).reshape(-1)
        self.loss_index = 0
        self.len_loss = 0

    def save_figure(self, is_training = False):
        # test current model and save model/figure if model yield best total rewards.
        # create env for testing, reset test env
        if self.env is None:
            self.env = create_env(self.world, self.stage, self.action_type, True)
        state = self.env.reset()
        done = False

        images = []
        total_reward = 0
        total_step = 0
        num_repeat_action = 0
        old_action = -1

        episode_time = datetime.now()

        # play 1 episode, just get loop action with max probability from model until the episode end.
        while not done:
            with torch.no_grad():
                logit, V = self.model(torch.tensor(state, dtype = torch.float, device = self.device).unsqueeze(0))
            action = logit.argmax(-1).item()
            next_state, reward, done, trunc, info = self.env.step(action)
            state = next_state
            img = Image.fromarray(self.env.current_state)
            images.append(img)
            total_reward += reward
            total_step += 1

            if action == old_action:
                num_repeat_action += 1
            else:
                num_repeat_action = 0
            old_action = action
            if num_repeat_action == 200:
                break

        #logging, if model yield better result, save figure (test_episode.mp4) and model (best_model.pth)
        if is_training:
            f_out = open(f"logging_test.txt", "a")
            f_out.write(f'episode_reward: {total_reward:.4f} episode_step: {total_step} current_step: {self.current_step} loss_p: {(self.P_loss.sum()/self.len_loss):.4f} loss_v: {(self.V_loss.sum()/self.len_loss):.4f} loss_e: {(self.E_loss.sum()/self.len_loss):.4f} loss: {(self.total_loss.sum()/self.len_loss):.4f} approx_kl_div: {(self.approx_kl_divs.sum()/self.len_loss):.4f} episode_time: {datetime.now() - episode_time}\n')
            f_out.close()

        if total_reward > self.max_test_score or info['flag_get']:
            imageio.mimsave('test_episode.mp4', images)
            self.max_test_score = total_reward
            if is_training:
                torch.save(self.model.state_dict(), f"best_model.pth")

        # if model can complete this game, stop training by set self.is_completed to True
        if info['flag_get']:
            self.is_completed = True

    def save_model(self):
        torch.save(self.model.state_dict(), f"model_{self.current_step}.pth")

    def load_model(self, model_path = None):
        if model_path is None:
            model_path = f"model_{self.current_step}.pth"
        self.model.load_state_dict(torch.load(model_path))

    def update_loss_statis(self, loss_p, loss_v, loss_e, loss, approx_kl_div):
        # update loss for logging, just save 1000 latest updates.
        self.V_loss[self.loss_index] = loss_v
        self.P_loss[self.loss_index] = loss_p
        self.E_loss[self.loss_index] = loss_e
        self.total_loss[self.loss_index] = loss
        self.approx_kl_divs[self.loss_index] = approx_kl_div
        self.loss_index = (self.loss_index + 1)%1000
        self.len_loss = min(self.len_loss+1, 1000)

    def select_action(self, states):
        # select action when training, we need use Categorical distribution to make action base on probability from model
        states = torch.tensor(np.array(states), device = self.device)

        with torch.no_grad():
            logits, V = self.model(states)
            policy = F.softmax(logits, dim=1).clamp(max=1 - 1e-20)
            distribution = torch.distributions.Categorical(policy)
            actions = distribution.sample().cpu().numpy().tolist()
        return actions, logits, V

    def calculate_gae_and_td_target(self, states, actions, next_states, rewards, dones, values = None, b_logits = None):
        if values is None:
            values = [None] * len(states)
            values_ = []
            logits = []
        if b_logits is None:
            b_logits = [None] * len(states)
        else:
            pro_dones = []
            pro_done = 1
        # calculate target (td lambda target) and gae advantages
        targets = []
        with torch.no_grad():
            _, next_value = self.old_model(torch.tensor(np.array(next_states[-1]), device = self.device))
        target = next_value
        advantage = 0
        advantages = []

        for state, action, next_state, reward, done, V, b_logit in zip(states[::-1], actions[::-1], next_states[::-1], rewards[::-1], dones[::-1], values[::-1], b_logits[::-1]):
            if V is None:
                with torch.no_grad():
                    logit, V = self.old_model(torch.tensor(np.array(state), device = self.device))
                    values_.append(V)
                    logits.append(logit)
            if b_logit is not None:
                b_logit = torch.tensor(b_logit, device = self.device)
                prob = torch.softmax(logit, -1).clamp(max=1 - 1e-20)
                b_prob = torch.softmax(b_logit, -1).clamp(max=1 - 1e-20)
                action = torch.tensor(np.array(action).reshape(-1)).long()
                index = torch.arange(len(prob))
                action_prob = prob[index, action]
                action_b_prob = b_prob[index, action]
                done_ = torch.tensor(done, device = self.device, dtype = torch.double).reshape(-1)
                pro_done = torch.ones((1, ), device = self.device).double() * (done_ + (1 - done_) * pro_done) * (action_prob.double() / (action_b_prob + 1e-18).double())
                pro_dones.append(pro_done.clone().clamp(max = 1e15).float())
                pro_done = pro_done.clamp(max = 1e100) #fix when pro_done very small or very large

            done = torch.tensor(done, device = self.device, dtype = torch.float).reshape(-1, 1)
            reward = torch.tensor(reward, device = self.device).reshape(-1, 1)

            target = next_value * self.gamma * (1-done) + reward
            advantage = target + self.gamma * advantage * (1-done) * self.gae_lambda
            targets.append(advantage)
            advantage = advantage - V.detach()
            advantages.append(advantage.detach().view(-1))
            next_value = V.detach()

        targets = targets[::-1]
        advantages = advantages[::-1]

        if b_logits[0] is None:
            if values[0] is None:
                values = values_[::-1]
                logits = logits[::-1]
                return targets, advantages, logits, values, _
            return targets, advantages, _, _, _
        else:
            pro_dones = pro_dones[::-1]
            if values[0] is None:
                values = values_[::-1]
                logits = logits[::-1]
                return targets, advantages, logits, values, pro_dones
            return targets, advantages, _, _, pro_dones


    def on_policy_learn(self):
        # get all data
        states, actions, next_states, rewards, dones, old_logits, old_values = self.memory.get_data()

        # calculate target (td lambda target) and gae advantages
        targets, advantages, _, _, _ = self.calculate_gae_and_td_target(states, actions, next_states, rewards, dones, old_values, None)
        backup_data = (states, actions, next_states, rewards, dones, old_logits, advantages)

        # convert all data to tensor
        action_index = torch.flatten(torch.tensor(actions, device = self.device, dtype = torch.int64))
        states = torch.tensor(np.array(states), device = self.device)
        states = states.reshape((-1,  states.shape[2], states.shape[3], states.shape[4]))
        old_values = torch.cat(old_values, 0)
        targets = torch.cat(targets, 0).view(-1, 1)
        advantages = torch.cat(advantages, 0).view(-1)
        old_logits = torch.cat(old_logits, 0)
        old_probs = torch.softmax(old_logits, -1).clamp(max=1 - 1e-20)
        index = torch.arange(0, len(old_probs), device = self.device)
        old_log_probs = (old_probs[index, action_index] + 1e-9).log()
        early_stopping = False

        #train num_epoch time
        for epoch in range(self.num_epoch):
            #shuffle data for each epoch
            shuffle_ids = torch.randperm(len(targets), dtype = torch.int64)
            for i in range(len(old_values)//self.batch_size):
                #train with batch_size data
                self.optimizer.zero_grad()
                start_id = i * self.batch_size
                end_id = min(len(shuffle_ids), (i+1) * self.batch_size)
                batch_ids = shuffle_ids[start_id:end_id]

                #predict logits and values from model
                logits, values = self.model(states[batch_ids])

                #calculate entropy and value loss (using mse or huber based on config)
                probs =  torch.softmax(logits, -1).clamp(max=1 - 1e-20)
                entropy = (- (probs * (probs + 1e-9).log()).sum(-1)).mean()
                if self.V_loss_type == 'huber':
                    loss_V = F.smooth_l1_loss(values, targets[batch_ids])
                else:
                    loss_V = F.mse_loss(values, targets[batch_ids])

                # calculate log_probs
                index = torch.arange(0, len(probs), device = self.device)
                batch_action_index = action_index[batch_ids]

                log_probs = (probs[index, batch_action_index] + 1e-9).log()

                #approx_kl_div copy from https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html#PPO
                #if approx_kl_div larger than 1.5 * target_kl (if target_kl in config is not None), stop training because policy change so much
                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs[batch_ids]
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    early_stopping = True

                #calculate policy loss
                ratio = torch.exp(log_probs - old_log_probs[batch_ids])
                batch_advantages = advantages[batch_ids].detach()
                if self.is_normalize_advantage:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-9)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                loss_P = -torch.min(surr1, surr2).mean()

                # update model
                loss = - entropy * self.entropy_coef + loss_V * self.V_coef + loss_P

                self.update_loss_statis(loss_P.item(), loss_V.item(), entropy.item(), loss.item(), approx_kl_div.item())

                if early_stopping == False:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                else:
                    break
            if early_stopping:
                break

        # calculate priority of each trajectory and save it to per
        states, actions, next_states, rewards, dones, old_logits, advantages = backup_data
        priorities = torch.stack(advantages, 0).abs().mean(0)
        states = np.array(states, dtype = np.uint8).transpose(1, 0, 2, 3, 4)
        next_states = np.array(next_states, dtype = np.uint8).transpose(1, 0, 2, 3, 4)
        actions = np.array(actions).transpose(1, 0)
        rewards = np.array(rewards).transpose(1, 0)
        dones = np.array(dones).transpose(1, 0)
        old_logits = torch.stack(old_logits, 1).cpu().numpy()#.transpose(1, 0, 2)
        priorities = priorities.cpu().numpy()
        for state, action, reward, next_state, done, old_logit, priority in zip(states, actions, rewards, next_states, dones, old_logits, priorities):
            self.per.add((state, action, reward, next_state, done, old_logit), priority)

    def off_policy_learn(self):
        # get all data
        (states, actions, next_states, rewards, dones, b_logits), _, per_index = self.per.sample(self.num_envs)

        # calculate target (td lambda target) and gae advantages
        targets, advantages, old_logits, old_values, pro_dones = self.calculate_gae_and_td_target(states, actions, next_states, rewards, dones, None, b_logits)
        backup_data = (states, actions, next_states, rewards, dones, advantages)

        # convert all data to tensor
        action_index = torch.flatten(torch.tensor(actions, device = self.device, dtype = torch.int64))
        states = torch.tensor(np.array(states), device = self.device)
        states = states.reshape((-1,  states.shape[2], states.shape[3], states.shape[4]))
        old_values = torch.cat(old_values, 0)
        targets = torch.cat(targets, 0).view(-1, 1)
        advantages = torch.cat(advantages, 0).view(-1)
        old_logits = torch.cat(old_logits, 0)
        old_probs = torch.softmax(old_logits, -1).clamp(max=1 - 1e-20)
        index = torch.arange(0, len(old_probs), device = self.device)
        old_log_probs = (old_probs[index, action_index] + 1e-9).log()
        early_stopping = False

        b_logits = torch.tensor(b_logits, device = self.device).reshape(-1, b_logits.shape[-1]).float()
        b_probs = torch.softmax(b_logits, -1).clamp(max=1 - 1e-20)
        b_log_probs = (b_probs[index, action_index] + 1e-9).log()
        old_action_probs = old_probs[index, action_index]
        b_action_probs = b_probs[index, action_index]
        pro_dones = torch.cat(pro_dones, 0)
        pro_marg = torch.min(torch.ones(pro_dones.shape, device = self.device) * (1 - self.eps_marg), pro_dones) + torch.relu((pro_dones - (1 - self.eps_marg)) / (pro_dones + 1e-9))
        pro_marg = pro_marg.float()
        advantages = advantages * pro_marg

        #train 1 time
        shuffle_ids = torch.randperm(len(targets), dtype = torch.int64)
        for i in range(len(old_values)//self.batch_size):
                #train with batch_size data
                self.optimizer.zero_grad()
                start_id = i * self.batch_size
                end_id = min(len(shuffle_ids), (i+1) * self.batch_size)
                batch_ids = shuffle_ids[start_id:end_id]

                #predict logits and values from model
                logits, values = self.model(states[batch_ids])

                #calculate entropy and value loss (using mse or huber based on config)
                probs =  torch.softmax(logits, -1).clamp(max=1 - 1e-20)
                entropy = (- (probs * (probs + 1e-9).log()).sum(-1)).mean()

                index = torch.arange(0, len(probs), device = self.device)
                batch_action_index = action_index[batch_ids]
                batch_b_action_probs = b_action_probs[batch_ids].view(-1)
                action_probs = probs[index, batch_action_index].view(-1)

                if self.V_loss_type == 'huber':
                    loss_V =  F.smooth_l1_loss(values, targets[batch_ids], reduction = 'none').view(-1)
                else:
                    loss_V = F.mse_loss(values, targets[batch_ids], reduction = 'none').view(-1)
                loss_V = (pro_marg[batch_ids] * loss_V).mean()

                # calculate log_probs
                log_probs = (probs[index, batch_action_index] + 1e-9).log()

                #approx_kl_div copy from https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html#PPO
                #if approx_kl_div larger than 1.5 * target_kl (if target_kl in config is not None), stop training because policy change so much
                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs[batch_ids]
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    early_stopping = True

                #calculate policy loss
                ratio = torch.exp(log_probs - b_log_probs[batch_ids])
                batch_advantages = advantages[batch_ids].detach()
                if self.is_normalize_advantage:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-9)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                loss_P = -torch.min(surr1, surr2).mean()

                # update model
                loss = - entropy * self.entropy_coef + loss_V * self.V_coef + loss_P

                self.update_loss_statis(loss_P.item(), loss_V.item(), entropy.item(), loss.item(), approx_kl_div.item())

                if early_stopping == False:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                else:
                    break

        # calculate priority of each trajectory and save it to per
        states, actions, next_states, rewards, dones, advantages = backup_data
        priorities = torch.stack(advantages, 0).abs().mean(0)
        priorities = priorities.cpu().numpy()
        self.per.update_priorities(per_index, priorities)

    def train(self):
        episode_reward = [0] * self.num_envs
        episode_step = [0] * self.num_envs
        max_episode_reward = 0
        max_episode_step = 0
        episode_time = [datetime.now() for _ in range(self.num_envs)]
        total_time = datetime.now()

        last_episode_rewards = []

        #reset envs
        states = self.envs.reset()

        while True:
            # finish training if agent reach total_step or total_episode base on what type of total_step_or_episode is step or episode
            self.current_step += 1

            if self.total_step_or_episode == 'step':
                if self.current_step >= self.total_step:
                    break
            else:
                if self.current_episode >= self.total_episode:
                    break

            actions, logits, values = self.select_action(states)

            next_states, rewards, dones, truncs, infos = self.envs.step(actions)

            # save to memory
            self.memory.save(states, actions, rewards, next_states, dones, logits, values)

            episode_reward = [x + reward for x, reward in zip(episode_reward, rewards)]
            episode_step = [x+1 for x in episode_step]

             # logging after each step, if 1 episode is ending, I will log this to logging.txt
            for i, done in enumerate(dones):
                if done:
                    self.current_episode += 1
                    max_episode_reward = max(max_episode_reward, episode_reward[i])
                    max_episode_step = max(max_episode_step, episode_step[i])
                    last_episode_rewards.append(episode_reward[i])
                    f_out = open(f"logging.txt", "a")
                    f_out.write(f'episode: {self.current_episode} agent: {i} rewards: {episode_reward[i]:.4f} steps: {episode_step[i]} complete: {infos[i]["flag_get"]==True} mean_rewards: {np.array(last_episode_rewards[-min(len(last_episode_rewards), 100):]).mean():.4f} max_rewards: {max_episode_reward:.4f} max_steps: {max_episode_step} current_step: {self.current_step} loss_p: {(self.P_loss.sum()/self.len_loss):.4f} loss_v: {(self.V_loss.sum()/self.len_loss):.4f} loss_e: {(self.E_loss.sum()/self.len_loss):.4f} loss: {(self.total_loss.sum()/self.len_loss):.4f} approx_kl_div: {(self.approx_kl_divs.sum()/self.len_loss):.4f} episode_time: {datetime.now() - episode_time[i]} total_time: {datetime.now() - total_time}\n')
                    f_out.close()
                    episode_reward[i] = 0
                    episode_step[i] = 0
                    episode_time[i] = datetime.now()

            # training agent every learn_step
            if self.current_step % self.learn_step == 0:
                with torch.no_grad():
                    self.old_model.load_state_dict(self.model.state_dict())
                self.on_policy_learn()
                self.memory.reset()
                if self.current_step // self.learn_step >= self.off_epoch:
                    for epoch in range(self.off_epoch):
                        self.off_policy_learn()

            # eval agent every save_figure_step
            if self.current_step % self.save_figure_step == 0:
                self.save_figure(is_training=True)
                if self.is_completed:
                    return

            if self.current_step % self.save_model_step == 0:
                self.save_model()

            states = list(next_states)

        f_out = open(f"logging.txt", "a")
        f_out.write(f' mean_rewards: {np.array(last_episode_rewards[-min(len(last_episode_rewards), 100):]).mean()} max_rewards: {max_episode_reward} max_steps: {max_episode_step} current_step: {self.current_step} total_time: {datetime.now() - total_time}\n')
        f_out.close()