import copy
import math
import os, sys, time
import argparse
import random
from typing import List, Optional

import pandas
from gymnasium.vector.utils import spaces
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR

from hintsClasses import Hint
from myMySql import MySQLconfig
from optimizers.LrFinderOptimizer import LrFinderOptimizer
from training_utils import TrainingProgress, timeSince, load_conf, check_path
from agent import DDPGfDAgent, DATA_RUNTIME, DATA_DEMO, DATE_CHAIN_DEMO
import torch, joblib
import torch.nn as nn
import numpy as np
from logger import logger_setup
import logging
from os.path import join as opj
import gymnasium as gym
import torch.optim as optim

np.set_printoptions(suppress=True, precision=4)

# Used loggers
DEBUG_LLV = 5  # for masked
loggers = ['RLTrainer', 'DDPGfD', 'TP']
# logging.addLevelName(DEBUG_LLV, 'DEBUGLLV')  # Lower level debugging info
logging_level = logging.DEBUG  # logging.DEBUG
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))


def fetch_obs(obs):
    return np.r_[obs['observation'], obs['achieved_goal'], obs['desired_goal']]


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def state_to_tensor(state):
    tensor_list = []
    # 遍历Tuple中的每个元素并进行转换
    for item in state:
        if isinstance(item, np.ndarray):
            tensor_list.extend(item.tolist())
        elif isinstance(item, int):
            tensor_list.append(item)
        else:
            # 处理其他类型的数据
            print("Unsupported data type:", type(item))
            continue
    tensor_list = torch.tensor(tensor_list)
    return tensor_list


class RLTrainer:

    def __init__(self, s_conf_path, env, action_dim, state_dim, hints, performance_default, goal, eval=False):
        self.optimizer_critic = None
        self.optimizer_actor = None
        self.full_conf = load_conf(s_conf_path)
        self.conf = self.full_conf.train_config
        self.action_dim = action_dim

        self.goal = goal
        progress_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progress')
        result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
        self.hint_chosen_result_path = os.path.join(result_dir, 'hint_chosen.txt')
        self.not_obey_result_path = os.path.join(result_dir, 'not_obey.txt')
        self.hint_priority_record_path = os.path.join(result_dir, 'hint_priority.xlsx')

        # Store in xxxx_dir/exp_name+exp_idx/...
        self.tp = TrainingProgress(progress_dir, result_dir, self.conf.exp_name)

        logger_setup(os.path.join(self.tp.result_path, self.conf.exp_name + '-log.txt'), loggers, logging_level)
        self.logger = logging.getLogger('RLTrainer')

        if torch.cuda.is_available():
            torch.cuda.set_device(self.conf.device)  # default 0
            # cudnn.benchmark = True # Faster only for fixed runtime size
            self.logger.info('Use CUDA Device ' + self.conf.device)
            self.device = self.conf.device
        self.device = 'cpu'
        self.logger.info('use CPU for computation')

        if self.conf.seed == -1:
            self.conf.seed = os.getpid() + int.from_bytes(os.urandom(4), byteorder="little") >> 1
            self.logger.info('Random Seed={}'.format(self.conf.seed))
        # Random seed
        torch.manual_seed(self.conf.seed)  # cpu
        np.random.seed(self.conf.seed)  # numpy

        # Backup environment config
        if not eval:
            self.tp.backup_file(s_conf_path, 'training.yaml')

        # Construct Env
        self.env = env  # gym.make('FetchReach-v1')
        self.hints: List[Hint] = hints
        self.logger.info('Environment Loaded')

        self.agent = DDPGfDAgent(self.full_conf.agent_config, self.device, self.env, action_dim, state_dim)
        self.agent.to(self.device)

        # 是否恢复之前的训练结果
        if self.conf.restore:
            self.restore_progress(eval)
        else:
            self.episode = 1
        self.set_optimizer('pretrain')
        # Loss Function setting
        reduction = 'none'
        if self.conf.mse_loss:
            self.q_criterion = nn.MSELoss(reduction=reduction)
        else:
            self.q_criterion = nn.SmoothL1Loss(reduction=reduction)

        # self.HintPriority = []

        self.hints_meeted: list[Hint] = []  # 当下符合条件的hints


        self.lambda3 = 1
        self.lambda4 = 0.00000001
        self.hint_priority_record = [[float(i) for i in range(0, len(self.hints)+1)]]  # 用与记录所有的hint优先级变化
        self.HintPriority = [self.lambda4] * (len(self.hints) + 1)
        self.demo2memory()  # ！！
        self.action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(action_dim),
                                                         self.full_conf.agent_config.action_noise_std)
        self.action_guide = False
        self.total_current_step = 0

    def restore_progress(self, eval=False):
        self.tp.restore_progress(self.conf.tps)  # tps only for restore process from conf
        self.agent.actor_b.load_state_dict(
            self.tp.restore_model_weight(self.conf.tps, self.device, prefix='actor_b'))
        self.agent.actor_t.load_state_dict(
            self.tp.restore_model_weight(self.conf.tps, self.device, prefix='actor_t'))
        self.agent.critic_b.load_state_dict(
            self.tp.restore_model_weight(self.conf.tps, self.device, prefix='critic_b'))
        self.agent.critic_t.load_state_dict(
            self.tp.restore_model_weight(self.conf.tps, self.device, prefix='critic_t'))

        self.episode = self.tp.get_meta('saved_episode') + 1
        np.random.set_state(self.tp.get_meta('np_random_state'))
        torch.random.set_rng_state(self.tp.get_meta('torch_random_state'))
        self.logger.info('Restore Progress,Episode={}'.format(self.episode))

    def summary(self):
        # call Test/Evaluation here
        self.tp.add_meta(
            {'saved_episode': self.episode, 'np_random_state': np.random.get_state(),
             'torch_random_state': torch.random.get_rng_state()})  # , 'validation_loss': self.valid_loss})
        self.save_progress(display=True)

    def save_progress(self, display=False):
        self.tp.save_model_weight(self.agent.actor_b, self.episode, prefix='actor_b')
        self.tp.save_model_weight(self.agent.actor_t, self.episode, prefix='actor_t')
        self.tp.save_model_weight(self.agent.critic_b, self.episode, prefix='critic_b')
        self.tp.save_model_weight(self.agent.critic_t, self.episode, prefix='critic_t')

        self.tp.save_progress(self.episode)
        self.tp.save_conf(self.conf.to_dict())
        if display:
            self.logger.info('Config name ' + self.conf.exp_name)
            self.logger.info('Progress Saved, current episode={}'.format(self.episode))

    def lr_lambda(self, current_step):
        initial_lr = self.conf.lr_rate
        max_lr = 0.0005
        total_steps = self.conf.n_episode  # 总的迭代次数
        return (initial_lr * (max_lr / initial_lr) ** (current_step / total_steps)) / initial_lr

    def set_optimizer(self, type='pretrain'):
        types = ['pretrain', 'action_guide_exp', 'exp', 'LrFinder']
        if type not in types:
            raise KeyError(f'{type} not a valid optimizer')
        # self.optimizer = getattr(optim, self.conf.optim)(
        #     filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.conf.lr_rate,
        #     weight_decay=self.conf.w_decay)  # default Adam
        if type == 'pretrain':
            pretrain_lr_rate = self.conf.pretrain_lr_rate
            self.optimizer_actor = torch.optim.Adam(self.agent.actor_b.parameters(), lr=pretrain_lr_rate,
                                                    weight_decay=self.conf.w_decay)
            self.optimizer_critic = torch.optim.Adam(self.agent.critic_b.parameters(), lr=pretrain_lr_rate,
                                                     weight_decay=self.conf.w_decay)
        elif type in ['action_guide_exp', 'exp']:
            gamma = 1
            initial_lr = self.conf.lr_rate
            final_lr = initial_lr / float(self.conf.lr_decrease_rate)
            if self.conf.action_guide:
                if self.conf.n_episode != self.conf.action_guide_episode:
                    gamma = (final_lr / initial_lr) ** (1 / (self.conf.n_episode - self.conf.action_guide_episode))
            else:
                gamma = (final_lr / initial_lr) ** (1 / self.conf.n_episode)

            self.optimizer_actor = torch.optim.Adam(self.agent.actor_b.parameters(), lr=self.conf.lr_rate,
                                                    weight_decay=self.conf.w_decay)
            self.actor_scheduler = ExponentialLR(self.optimizer_actor, gamma=gamma)

            self.optimizer_critic = torch.optim.Adam(self.agent.critic_b.parameters(), lr=self.conf.lr_rate,
                                                     weight_decay=self.conf.w_decay)
            self.critic_scheduler = ExponentialLR(self.optimizer_critic, gamma=gamma)
        elif type == 'LrFinder':

            self.optimizer_actor = torch.optim.Adam(self.agent.actor_b.parameters(), lr=self.conf.lr_rate,
                                                    weight_decay=self.conf.w_decay)
            self.actor_scheduler = LambdaLR(self.optimizer_actor, self.lr_lambda)
            self.optimizer_critic = torch.optim.Adam(self.agent.critic_b.parameters(), lr=self.conf.lr_rate,
                                                     weight_decay=self.conf.w_decay)
            self.critic_scheduler = LambdaLR(self.optimizer_critic, self.lr_lambda)


    # ！！！！！！！！！
    def demo2memory(self):
        dconf = self.full_conf.demo_config
        if dconf.load_demo_data:
            # for f_idx in range(dconf.load_N):
            self.agent.episode_reset()
            fname = os.path.join(script_dir, "../../../algorithms/RL/DDPGFD/data/demo/demo_1.pkl") # opj(dconf.demo_dir, dconf.prefix + str(f_idx) + '.pkl')
            data = joblib.load(fname)
            # 表示Hint priority index start from 1 ，
            for exp in data:
                s, a, r, s2, done, hint_id = exp
                if self.conf.update_hint_priority:
                    self.HintPriority[
                        hint_id] = r if r >= 0 else 0  # record beginning reward, to represent the priority next

                s_tensor = state_to_tensor(s)  # torch.from_numpy(s).float()
                s2_tensor = state_to_tensor(s2)  # torch.from_numpy(s2).float()
                action = torch.from_numpy(a).float()
                if not done or self.agent.conf.N_step == 0:
                    self.agent.memory.add((s_tensor, action, torch.tensor([r]).float(), s2_tensor,
                                           torch.tensor([self.agent.conf.gamma]),
                                           hint_id))  # Add one-step to memory, last step added in pop with done=True index+1为DATA_DEMO, 大于等于1为demonstration，0为正常transition

                # Add new step to N-step and Pop N-step data to memory
                if self.agent.conf.N_step > 0:
                    self.agent.backup.add_exp(
                        (s_tensor, action, torch.tensor([r]).float(), s2_tensor))  # Push to N-step backup
                    self.agent.add_n_step_experience(hint_id, done)

            self.hint_priority_record.append(copy.deepcopy(self.HintPriority))
            self.logger.info('{}/{} Demo Trajectories Loaded. Total Experience={}'.format(dconf.load_N, dconf.demo_N,
                                                                                          len(self.agent.memory)))
            self.agent.memory.set_protect_size(len(self.agent.memory))
        else:
            self.logger.info('No Demo Trajectory Loaded')

    def update_agent(self, update_step):  # update_step iteration
        # 2. Sample experience and update
        losses_critic = []
        losses_actor = []
        demo_cnt = []
        batch_sz = 0
        if self.agent.memory.ready():
            for _ in range(update_step):
                (batch_s, batch_a, batch_r, batch_s2, batch_gamma,
                 batch_flags), weights, idxes = self.agent.memory.sample(
                    self.conf.batch_size)
                batch_s, batch_a, batch_r, batch_s2, \
                    batch_gamma, weights = batch_s.to(self.device), batch_a.to(self.device), batch_r.to(
                    self.device), batch_s2.to(self.device), batch_gamma.to(self.device), torch.from_numpy(
                    weights.reshape(-1, 1)).float().to(self.device)

                batch_sz += batch_s.shape[0]
                with torch.no_grad():
                    action_tgt = self.agent.actor_t(batch_s)
                    y_tgt = batch_r + batch_gamma * self.agent.critic_t(torch.cat((batch_s, action_tgt), dim=1))

                self.agent.zero_grad()
                # Critic loss
                self.optimizer_critic.zero_grad()
                Q_b = self.agent.critic_b(torch.cat((batch_s, batch_a), dim=1))
                loss_critic = (self.q_criterion(Q_b, y_tgt) * weights).mean()
                # Record Demo count
                d_flags = torch.from_numpy(batch_flags)
                demo_select = d_flags >= DATA_DEMO  # demo_select = d_flags == DATA_DEMO
                N_act = demo_select.sum().item()
                demo_cnt.append(N_act)
                loss_critic.backward()
                self.optimizer_critic.step()

                # Actor loss
                self.optimizer_actor.zero_grad()
                action_b = self.agent.actor_b(batch_s)
                Q_act = self.agent.critic_b(torch.cat((batch_s, action_b), dim=1))
                grad_Q_a = torch.autograd.grad(Q_act, action_b, torch.ones_like(Q_act), retain_graph=True)[0]
                loss_actor = -torch.mean(Q_act)
                loss_actor.backward()
                self.optimizer_actor.step()

                # priority = ((Q_b.detach() - y_tgt).pow(2) + Q_act.detach().pow(
                #     2)).cpu().numpy().ravel() + self.agent.conf.const_min_priority

                grad_Q_a_pow = torch.sum(grad_Q_a ** 2, dim=1).view(-1, 1)
                priority = ((Q_b.detach() - y_tgt).pow(
                    2) + self.lambda3 * grad_Q_a_pow).cpu().numpy().ravel() + self.agent.conf.const_min_priority
                priority[batch_flags >= DATA_DEMO] += self.agent.conf.const_demo_priority  # 单独的演示数据
                priority[batch_flags == DATE_CHAIN_DEMO] += self.agent.conf.const_demo_priority  # gpt 给出的chain的演示数据
                for index, flag in enumerate(batch_flags):
                    if flag >= DATA_DEMO:
                        priority[index] += float(self.agent.conf.transition_priority_multiple) * self.HintPriority[flag]

                if not self.agent.conf.no_per:
                    self.agent.memory.update_priorities(idxes, priority)

                losses_actor.append(loss_actor.item())
                losses_critic.append(loss_critic.item())
        if np.sum(demo_cnt) == 0:
            demo_n = 1e-10
        else:
            demo_n = np.sum(demo_cnt)
        return np.sum(losses_critic), np.sum(losses_actor), demo_n, batch_sz

    # use demonstration to pretrain
    def pretrain(self):

        assert self.full_conf.demo_config.load_demo_data
        self.agent.train()  # 确定这是train
        start_time = time.time()
        self.logger.info('Run Pretrain')
        for step in np.arange(self.conf.pretrain_save_step, self.conf.pretrain_step + 1,
                              self.conf.pretrain_save_step):  # 5 5 5
            losses_critic, losses_actor, demo_n, batch_sz = self.update_agent(self.conf.pretrain_save_step)

            current_lr_actor = self.optimizer_actor.param_groups[0]['lr']
            print(f"pretrain Current actor learning rate: {current_lr_actor}")

            current_lr_critic = self.optimizer_critic.param_groups[0]['lr']
            print(f"pretrain Current critic learning rate: {current_lr_critic}")

            self.logger.info(
                '{}-Pretrain Step {}/{},(Mean):actor_loss={:.8f}, critic_loss={:.8f}, batch_sz={}, Demo_ratio={:.8f}'.format(
                    timeSince(start_time), step, self.conf.pretrain_step, losses_actor / batch_sz,
                                                                          losses_critic / batch_sz, batch_sz,
                                                                          demo_n / batch_sz))
            self.tp.record_step(step, 'pre_train',
                                {'actor_loss_mean': losses_actor / batch_sz,
                                 'critic_loss_mean': losses_critic / batch_sz,
                                 'batch_sz': batch_sz,
                                 'Demo_ratio': demo_n / batch_sz
                                 }, display=False)
            self.episode = 'pre_{}'.format(step)
            self.summary()
            self.tp.plot_data('pre_train', self.conf.pretrain_save_step, step,
                              'result-pretrain-{}.png'.format(self.episode),
                              self.conf.exp_name + '-Pretrain', grid=False,
                              ep_step=self.conf.pretrain_save_step)
        self.episode = 1  # Restore
        # self.actor_scheduler.step()
        # self.critic_scheduler.step()

    def train(self):
        """
        :rtype: object
        """
        self.agent.train()  # 设置模式
        # Define criterion
        start_time = time.time()

        self.set_optimizer(self.conf.lr_optimizer)
        # if not self.conf.action_guide:
        #     self.set_optimizer('exp')
        #self.set_optimizer('LrFinder')
        while self.episode <= self.conf.n_episode:  # self.iter start from 1
            # Episodic statistics
            # if self.conf.action_guide:
            #     if self.episode == self.conf.action_guide_episode + 1:
            #         self.set_optimizer('action_guide_exp')

            eps_since = time.time()
            eps_reward = eps_length = eps_actor_loss = eps_critic_loss = eps_batch_sz = eps_demo_n = 0
            s0, _ = self.env.reset()

            self.agent.episode_reset()  # N step memory
            self.action_noise.reset()

            done = False
            s_tensor = self.agent.obs2tensor(s0)

            while not done:
                # 1. Run environment step
                with torch.no_grad():  # 不要使用梯度
                    # s_tensor = self.agent.obs2tensor(state)

                    action_noise = torch.from_numpy(self.action_noise()).float()
                    action = self.agent.actor_b(s_tensor.to(self.device)[None])[0].cpu() + action_noise
                    action = torch.clamp(action, -1, 1)

                    s_now = s_tensor.tolist()
                    # 每一个step之前会先更新当下state的hints并根据hint进行指导
                    chosen_hint: Optional[Hint] = None
                    not_obey_num = 0
                    space = self.env.get_wrapper_attr('space')
                    self.total_current_step = int(self.env.get_wrapper_attr('total_step')) + 1
                    # 选出所有符合条件的hints
                    self.find_meeted_hints(s_now)
                    # hints_meeted_priorities = [self.HintPriority[hint_meeted.get_hint_id()] for hint_meeted in
                    #                            self.hints_meeted]
                    # reward shaping 找到当前action与hint违背的个数
                    if self.conf.reward_guide and self.hints_meeted is not None:
                        last_conf = self.env.get_wrapper_attr('last_conf')

                        for hint in self.hints_meeted:
                            hint_priority = self.HintPriority[hint.get_hint_id()]
                            if float(hint_priority) < float(self.conf.reward_shaping_threshold):
                                continue
                            conf_name = hint.get_conf_name()
                            min_val, max_val = space[conf_name]
                            _, index = self.get_conf_by_name(conf_name, action)

                            current_value = min_val + (max_val - min_val) * (action[index] + 1) / 2
                            suggest_value = hint.get_tuningValue().get_value(current_value, min_val, max_val, conf_name)
                            # suggest_a = 2 * ((suggest_value - min_val) / (max_val - min_val)) - 1
                            if (suggest_value - last_conf[conf_name] > 0 and (
                                    current_value - last_conf[conf_name]) < 0) or (
                                    suggest_value - last_conf[conf_name] < 0 and (
                                    current_value - last_conf[conf_name]) > 0):
                                not_obey_num += 1
                                with open(self.not_obey_result_path, 'a') as f:
                                    f.write(str(self.total_current_step) + " " + hint.to_string() + "\n")
                            # if (action[index] > 0 and suggest_a < 0) or (action[index] < 0 and suggest_a > 0):

                    # action guide 选出一个去指导action
                    if self.conf.action_guide and int(self.episode) <= int(self.conf.action_guide_episode):
                        # 选出action扰动和reward-guide的hint
                        if self.hints_meeted:
                            # 从所有符合当前环境的hints中根据hint的优先级每次选出一个测试(扰动)
                            hints_meeted_priorities = [self.HintPriority[hint_meeted.get_hint_id()] for hint_meeted in
                                                       self.hints_meeted]
                            if all(weight <= 0 for weight in hints_meeted_priorities):
                                hints_meeted_priorities = [1] * len(hints_meeted_priorities)
                            hints_meeted_priorities_ln = [math.log(priority + 1.1) if priority > 0 else 0 for priority
                                                          in hints_meeted_priorities]
                            chosen_hint = random.choices(self.hints_meeted, weights=hints_meeted_priorities_ln)[0]
                            with open(self.hint_chosen_result_path, 'a') as file:
                                file.write(str(self.total_current_step) + " " + chosen_hint.to_string() + '\n')

                            _, idx = self.get_conf_by_name(chosen_hint.get_conf_name(), action)

                            conf_range = space[chosen_hint.get_conf_name()]
                            min_val, max_val = conf_range
                            action_source_conf = min_val + (max_val - min_val) * (action[idx] + 1) / 2
                            recommend_value = chosen_hint.get_tuningValue().get_value(action_source_conf, min_val,
                                                                                      max_val,
                                                                                      chosen_hint.get_conf_name())
                            suggest_a = 2 * ((recommend_value - min_val) / (max_val - min_val)) - 1
                            action[idx] = suggest_a

                            # if (action[idx] > 0 and recommend_value < 0) or (action[idx] < 0 and recommend_value > 0):
                            #     is_obey = False

                    s2, r, done, _, _ = self.env.step(action.numpy())

                    # reward shaping  更新hint优先级
                    reward_shaping_ratio = float(self.conf.reward_shaping_ratio)
                    if self.conf.reward_guide and self.hints_meeted is not None:
                        if not_obey_num > 0:
                            if not_obey_num < 10:
                                r -= (r / reward_shaping_ratio) * (not_obey_num / 10)
                            else:
                                r -= (r / reward_shaping_ratio)

                    if self.conf.update_hint_priority and self.conf.action_guide and int(self.episode) <= int(self.conf.action_guide_episode):
                        self.HintPriority[chosen_hint.get_hint_id()] = (self.HintPriority[
                                                                            chosen_hint.get_hint_id()] + r) / 2

                    s2 = s2  # fetch_obs(s2)
                    s2_tensor = self.agent.obs2tensor(s2)
                    if not done or self.agent.conf.N_step == 0:  # For last step, not duplicate to the last pop from N_step
                        self.agent.memory.add((s_tensor, action, torch.tensor([r]).float(), s2_tensor,
                                               torch.tensor([self.agent.conf.gamma]),
                                               DATA_RUNTIME))  # Add one-step to memory
                    # 这一step的hint优先级存其来
                    self.hint_priority_record.append(copy.deepcopy(self.HintPriority))
                # Add new step to N-step and Pop N-step data to memory
                if self.agent.conf.N_step > 0:
                    self.agent.backup.add_exp(
                        (s_tensor, action, torch.tensor([r]).float(), s2_tensor))  # Push to N-step backup
                    self.agent.add_n_step_experience(DATA_RUNTIME, done)  # Pop one

                losses_critic, losses_actor, demo_n, batch_sz = self.update_agent(self.conf.update_step)

                # 3. Record episodic statistics
                eps_reward += r
                eps_length += 1
                eps_actor_loss += losses_actor
                eps_critic_loss += losses_critic
                eps_batch_sz += batch_sz
                eps_demo_n += demo_n

                # Next step
                s_tensor = s2_tensor

            self.logger.info(
                '{}: Episode {}-Last:{}: Actor_loss={:.8f}, Critic_loss={:.8f}, Step={}, Reward={}, Demo_ratio={:.8f}'.format(
                    timeSince(start_time),
                    self.episode,
                    timeSince(eps_since),
                    eps_actor_loss / eps_batch_sz,
                    eps_critic_loss / eps_batch_sz,
                    eps_length, eps_reward, eps_demo_n / eps_batch_sz))

            # Update target
            self.agent.update_target(self.agent.actor_b, self.agent.actor_t, self.episode)
            self.agent.update_target(self.agent.critic_b, self.agent.critic_t, self.episode)

            self.tp.record_step(self.episode, 'episode',
                                {'total_reward': eps_reward, 'length': eps_length,
                                 'avg_reward': eps_reward / eps_length,
                                 'elapsed_time': timeSince(eps_since, return_seconds=True),
                                 'actor_loss_mean': eps_actor_loss / eps_batch_sz,
                                 'critic_loss_mean': eps_critic_loss / eps_batch_sz,
                                 'eps_length': eps_length,
                                 'Demo_ratio': eps_demo_n / eps_batch_sz,
                                 }, display=False)
            #

            if self.episode % self.conf.save_every == 0:
                # self.eval()  # Run before summary
                self.summary()
                self.tp.plot_data('episode', 1, self.episode, 'result-train-{}.png'.format(self.episode),
                                  self.conf.exp_name + str(self.conf.exp_idx) + '-Episode', grid=False)

            current_lr_actor = self.optimizer_actor.param_groups[0]['lr']
            print(f"episode{self.episode}: actor learning rate: {current_lr_actor}")

            current_lr_critic = self.optimizer_critic.param_groups[0]['lr']
            print(f"episode{self.episode}: critic learning rate: {current_lr_critic}")

            if self.conf.action_guide:
                if self.episode > self.conf.action_guide_episode:
                    self.actor_scheduler.step()
                    self.critic_scheduler.step()
            else:
                self.actor_scheduler.step()
                self.critic_scheduler.step()

            self.episode += 1
        self.save_files()
    def save_files(self):
        hint_priority_record_path = self.hint_priority_record_path
        pd = pandas.DataFrame(self.hint_priority_record)
        pd.to_excel(hint_priority_record_path)

    def eval(self, save_fig=True):
        self.agent.eval()

        all_length = []
        all_reward = []

        # Backup Environment state

        for eps in range(self.conf.eval_episode):  # self.iter start from 1
            # Episodic statistics
            eps_reward = eps_length = 0
            s0, _ = self.env.reset()  # s0 = fetch_obs(self.env.reset())
            done = False
            s_tensor = self.agent.obs2tensor(s0)
            while not done:
                # 1. Run environment step
                with torch.no_grad():
                    # s_tensor = self.agent.obs2tensor(state)
                    action = self.agent.actor_b(s_tensor.to(self.device)[None])[0].cpu()
                    s2, r, done, _, _ = self.env.step(action.numpy())
                    s2 = s2  # fetch_obs(s2)
                    s2_tensor = self.agent.obs2tensor(s2)
                    self.env.render()

                # 3. Record episodic statistics
                eps_reward += r
                eps_length += 1

                # Next step
                s_tensor = s2_tensor
            all_length.append(eps_length)
            all_reward.append(eps_reward)

        self.tp.record_step(self.episode, 'eval', {'Mean Length': np.mean(all_length), 'Std Length': np.std(all_length),
                                                   'Mean Reward': np.mean(all_reward),
                                                   'Std Reward': np.std(all_reward)})
        self.logger.info(
            'Eval Episode-{}: Mean Reward={:.3f}, Mena Length={:.3f}'.format(self.episode, np.mean(all_reward),
                                                                             np.mean(all_length)))

        if save_fig:
            self.tp.plot_data('eval', self.conf.save_every, self.episode, 'result-eval-{}.png'.format(self.episode),
                              self.conf.exp_name + str(self.conf.exp_idx) + '-Evaluate',
                              self.conf.save_every)  # start from self.conf.save_every
        self.agent.train()

    # 不更新网络 只是收集transition
    def collect_demo(self, n_collect):
        self.agent.eval()

        demo_record = []  # list of tuple (s,a,r,s',Terminal)
        file_idx = 0
        check_path(self.full_conf.demo_config.demo_dir)

        with torch.no_grad():
            for eps in range(n_collect):  # self.iter start from 1
                # Episodic statistics
                s = fetch_obs(self.env.reset())
                done = False
                s_tensor = self.agent.obs2tensor(s)
                while not done:
                    # s_tensor = self.agent.obs2tensor(state)
                    action = self.agent.actor_b(s_tensor.to(self.device)[None])[0].cpu().numpy()
                    s2, r, done, _ = self.env.step(action)
                    s2 = fetch_obs(s2)
                    s2_tensor = self.agent.obs2tensor(s2)
                    self.env.render()

                    demo_record.append((s, action, r, s2, done))
                    if done:
                        save_name = opj(self.full_conf.demo_config.demo_dir,
                                        self.full_conf.demo_config.prefix + str(eps) + '.pkl')
                        joblib.dump(demo_record, save_name)
                        self.logger.info('Terminate: Record {} saved'.format(save_name))
                        demo_record = []
                        file_idx += 1

                    # Next step
                    s_tensor = s2_tensor
                    s = s2
        print('Collect {} Demo'.format(n_collect))

    def find_meeted_hints(self, s_now):
        if self.conf.action_guide or self.conf.reward_guide:
            for hint in self.hints:
                if self.meet_condition(hint.get_conditions(), s_now):
                    if hint not in self.hints_meeted:
                        # hint.condition == s:
                        print('ADD Hint ' + hint.to_string())
                        self.hints_meeted.append(hint)  # 加入当前state meet的hint列表中，Env step中需要reward导向
                else:
                    if hint in self.hints_meeted:
                        self.hints_meeted.remove(hint)
                        print('DROP Hint ' + hint.to_string())
            self.env.unwrapped.update_hints(self.hints_meeted)

    def trans_conf_to_action(self, conf, conf_space):
        conf_keys_list = self.env.get_wrapper_attr('conf_keys_list')
        action = np.zeros(len(conf), dtype=np.float32)
        for idx, conf_name in enumerate(conf_keys_list):
            conf_min = conf_space[conf_name][0]
            conf_max = conf_space[conf_name][1]
            action[idx] = ((conf[idx] - conf_min) / (conf_max - conf_min)) * 2 - 1
        return action

    def collect_demo_use_demonstration(self, n_collect):
        self.agent.eval()
        demo_record = []  # list of tuple (s,a,r,s',Terminal)
        check_path(self.full_conf.demo_config.demo_dir)

        with torch.no_grad():
            # 恢复为默认配置 获得当前负载默认配置下的状态
            s, _ = self.env.reset()
            conf_default = self.env.get_wrapper_attr('conf_default')
            conf_default_copy = copy.deepcopy(conf_default)
            conf_space = self.env.get_wrapper_attr('space')
            # 当前负载下默认配置的系统状态  每一条匹配hint并 并sample
            for hint in self.hints:
                # conf = np.zeros(self.action_dim)  # 全是0表示所有参数都是默认的
                # self.dbms.reset_config()
                # self.dbms.make_conf_effect()
                # self.dbms.restart_dbms()
                conf = [v for v in conf_default_copy.values()]
                if self.meet_condition(hint.get_conditions(), s):  # hint.condition == s:
                    self.hints_meeted.append(hint)
                    print(hint.to_string() + 'is included')
                    conf_name = hint.get_conf_name()
                    conf_min = conf_space[conf_name][0]
                    conf_max = conf_space[conf_name][1]
                    tuning_value = hint.get_tuningValue().get_value(conf_default[conf_name], conf_min, conf_max,
                                                                    conf_name)
                    conf_name = hint.config_name
                    idx = -1
                    for k, v in conf_default.items():
                        idx += 1
                        if str(k) == conf_name:
                            conf[idx] = tuning_value

                    a = self.trans_conf_to_action(conf, conf_space)
                    s2, r, done, _, _ = self.env.step(a)
                    demo_record.append(
                        (s, a, r, s2, done, hint.get_hint_id()))  # demo_record.append((s, a, r, s2, done))
                else:
                    if hint in self.hints_meeted:
                        self.hints_meeted.remove(hint)
                        print(hint.to_string() + 'is removed')

            # for eps in range(free_search_use_demonstration):
            #     done = False
            #     s_tensor = self.agent.obs2tensor(s)
            #     while not done:  # 走一个episode的steps
            #         action = self.agent.actor_b(s_tensor.to(self.device)[None])[0].cpu().numpy()
            #         s2, r, done, _ = self.env.step(action)
            #         s2 = fetch_obs(s2) == == == == == == == == == == == == == == == == == == 加上扰动
            #         s2_tensor = self.agent.obs2tensor(s2)
            #
            #         demo_record.append((s, action, r, s2, done))
            # save_name = opj(self.full_conf.demo_config.demo_dir, self.full_conf.demo_config.prefix + '.pkl')

            save_name = os.path.join(project_dir, 'src/algorithms/RL/DDPGFD/data/demo/demo_1.pkl')

            joblib.dump(demo_record, save_name)
            meeted_hint_save_name = os.path.join(project_dir, 'src/algorithms/RL/DDPGFD/data/demo/meeted_hints.txt')
                # "/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/src/algorithms/RL/DDPGFD/data/demo/meeted_hints.txt"
            with open(meeted_hint_save_name, 'w') as file:
                # 遍历数组中的每个字符串
                for hint_meeted in self.hints_meeted:
                    # 写入字符串到文件，并添加换行符
                    file.write(hint_meeted.to_string() + '\n')
            self.logger.info('Terminate: Record {} saved'.format(save_name))

    def meet_condition(self, conditions, obs):
        # return True
        is_meet = True
        if not conditions:
            is_meet = True
        dbms = self.env.get_wrapper_attr('dbms')
        dbms_name = dbms.dbms_name
        if dbms_name == 'mysql':
            for condition in conditions:
                if condition == 'memory available':
                    if not (float(self.get_ob_by_name('mem_usage', obs)) < 50):
                        is_meet = False
                if condition == 'low buffer ratio' and float(
                        self.get_ob_by_name('Innodb_buffer_pool_read_requests', obs)) != 0:
                    if not ((1 - (float(self.get_ob_by_name('Innodb_buffer_pool_reads', obs)) / float(
                            self.get_ob_by_name('Innodb_buffer_pool_read_requests', obs)))) * 100) > 99:
                        is_meet = False
                if condition == 'many connections':
                    if not int(self.get_ob_by_name('Threads_connected', obs)) > 3:
                        is_meet = False
                if condition == 'write heavy':
                    if not int(self.get_ob_by_name('Com_select', obs)) / (
                            int(self.get_ob_by_name('Com_select', obs)) + int(
                        self.get_ob_by_name('Com_insert', obs)) + int(self.get_ob_by_name('Com_update', obs)) + int(
                        self.get_ob_by_name('Com_delete', obs))) < 0.3:
                        is_meet = False
                if condition == 'read heavy':
                    if not int(self.get_ob_by_name('Com_select', obs)) / (
                            int(self.get_ob_by_name('Com_select', obs)) + int(
                        self.get_ob_by_name('Com_insert', obs)) + int(self.get_ob_by_name('Com_update', obs)) + int(
                        self.get_ob_by_name('Com_delete', obs))) > 0.6:
                        is_meet = False
                if condition == 'large dataset':
                    if not int(self.get_ob_by_name('data_set_num', obs)) > 30000:
                        is_meet = False
                if condition == 'large queries':
                    if not (int(self.get_ob_by_name('Com_select', obs)) + int(
                            self.get_ob_by_name('Com_insert', obs)) + int(self.get_ob_by_name('Com_update', obs)) + int(
                        self.get_ob_by_name('Com_delete', obs))) > 50000:
                        is_meet = False
                if condition == 'dirty data in buffer pool':
                    if float(self.get_ob_by_name('Innodb_buffer_pool_pages_total', obs)) == 0:
                        is_meet = False
                    elif not (float(self.get_ob_by_name('Innodb_buffer_pool_pages_dirty', obs)) / float(
                            self.get_ob_by_name('Innodb_buffer_pool_pages_total', obs))) > 0.25:
                        is_meet = False
                if condition == 'HDD':
                    if not False:
                        is_meet = False
                if condition == 'SSD':
                    pass
                if condition == 'log wait':
                    if not (float(self.get_ob_by_name('Innodb_log_waits', obs)) > 0):
                        is_meet = False
                if condition == 'purge operation':
                    if not (int(self.get_ob_by_name('Com_update', obs)) + int(
                            self.get_ob_by_name('Com_delete', obs)) > 20000):
                        is_meet = False
                if condition == 'lacking index':
                    if not int(self.get_ob_by_name('Handler_read_rnd_next', obs)) > 10000:
                        is_meet = False
                if condition == 'no lacking index':
                    if not int(self.get_ob_by_name('Handler_read_rnd_next', obs)) < 1000:
                        is_meet = False
                if condition == 'join operation':
                    if not int(self.get_ob_by_name('Handler_read_rnd_next', obs)) > 10000:
                        is_meet = False
                if condition == 'heap table operations':
                    if not int(self.get_ob_by_name('Created_tmp_disk_tables', obs)) / int(
                            self.get_ob_by_name('Created_tmp_tables', obs)) > 0.2:
                        is_meet = False
                if condition == 'sequential scan':
                    if not int(self.get_ob_by_name('Handler_read_rnd_next', obs)) > 10000:
                        is_meet = False
                if condition == 'sort':
                    if not int(self.get_ob_by_name('Sort_merge_passes', obs)) > 1:
                        is_meet = False
                if condition == 'insert':
                    if not int(self.get_ob_by_name('Com_insert', obs)) > 10000:
                        is_meet = False
                if condition == 'update':
                    if not int(self.get_ob_by_name('Com_update', obs)) > 10000:
                        is_meet = False
                if condition == 'WO_noHIntUpdate tables':
                    if not int(self.get_ob_by_name('Created_tmp_disk_tables', obs)) / int(
                            self.get_ob_by_name('Created_tmp_tables', obs)) > 0.2:
                        is_meet = False
            return is_meet
        elif dbms_name == 'pg':
            for condition in conditions:
                if condition == 'memory available':
                    if not (float(self.get_ob_by_name('mem_usage', obs)) < 50):
                        is_meet = False
                elif condition == 'not memory available':
                    if not (float(self.get_ob_by_name('mem_usage', obs)) > 90):
                        is_meet = False
                elif condition == 'more than 1GB mem':
                    if not (float(self.get_ob_by_name('mem_usage', obs)) < 10):
                        is_meet = False
                elif condition == 'less than 1GB mem':
                    if not (float(self.get_ob_by_name('mem_usage', obs)) > 95):
                        is_meet = False
                elif condition == 'many cpus':
                    if not (float(self.get_ob_by_name('cpu_usage', obs)) <= 50):
                        is_meet = False
                elif condition == 'not many cpus':
                    if not (float(self.get_ob_by_name('cpu_usage', obs)) >= 80):
                        is_meet = False
                elif condition == 'IO available':
                    if not (float(self.get_ob_by_name('io_latency_changed', obs)) < 0.5 and float(
                            self.get_ob_by_name('write_speed_changed', obs)) > 2000000):
                        is_meet = False
                elif condition == 'not IO available':
                    if not (float(self.get_ob_by_name('io_latency_changed', obs)) > 0.5 and float(
                            self.get_ob_by_name('write_speed_changed', obs)) < 2000000):
                        is_meet = False
                elif condition == 'read heavy':
                    if not (float(self.get_ob_by_name('readproportion', obs)) >= 0.7):
                        is_meet = False
                elif condition == 'not read heavy':
                    if not (float(self.get_ob_by_name('readproportion', obs)) < 0.5):
                        is_meet = False
                elif condition == 'write heavy':
                    if not (float(self.get_ob_by_name('updateproportion', obs)) + float(
                            self.get_ob_by_name('insertproportion', obs)) >= 0.8):
                        is_meet = False
                elif condition == 'not write heavy':
                    if not (float(self.get_ob_by_name('updateproportion', obs)) + float(
                            self.get_ob_by_name('insertproportion', obs)) < 0.5):
                        is_meet = False

                elif condition == 'delay in wal data to disk':
                    if not (float(self.get_ob_by_name('wal_write_time', obs)) > 100):
                        is_meet = False
                elif condition == 'issues with caching data':
                    if float(self.get_ob_by_name('blks_hit', obs)) + float(self.get_ob_by_name('blks_read', obs)) != 0:
                        if not (float(self.get_ob_by_name('blks_hit', obs)) / (
                                float(self.get_ob_by_name('blks_hit', obs)) + float(
                            self.get_ob_by_name('blks_read', obs))) < 0.85):
                            is_meet = False
                    else:
                        is_meet = False
                elif condition == 'many concurrent transactions':
                    if not (float(self.get_ob_by_name('xact_commit', obs)) > 60000):
                        is_meet = False
                elif condition == 'dirty data in kernel page':
                    if not (int(self.get_ob_by_name('node_memory_Dirty_bytes', obs)) > 10000000):
                        is_meet = False
                elif condition == 'not dirty data in kernel page':
                    if not (int(self.get_ob_by_name('node_memory_Dirty_bytes', obs)) <= 10000000):
                        is_meet = False
                elif condition == 'frequent dirty buffers':
                    if not (int(self.get_ob_by_name('buffers_backend', obs)) > 1200):
                        is_meet = False
                elif condition == 'not frequent dirty buffers':
                    if not (int(self.get_ob_by_name('buffers_checkpoint', obs)) <= 1200):
                        is_meet = False
                elif condition == 'frequent checkpoint':
                    if not (int(self.get_ob_by_name('checkpoints_req', obs)) > 5):
                        is_meet = False

                # 负载设置相关
                elif condition == 'many connections':
                    if not (float(self.get_ob_by_name('threadcount', obs)) > 2):
                        is_meet = False
                elif condition == 'parallel operations':
                    if not (int(self.get_ob_by_name('threadcount', obs)) >= 2):
                        is_meet = False
                elif condition == 'large dataset':
                    if not (int(self.get_ob_by_name('recordcount', obs)) >= 10000):
                        is_meet = False
                elif condition == 'small table':
                    if not (int(self.get_ob_by_name('fieldcount', obs)) < 20):
                        is_meet = False
                elif condition == 'large table':
                    if not (int(self.get_ob_by_name('fieldcount', obs)) >= 100):
                        is_meet = False
                elif condition == 'large queries':
                    if not (int(self.get_ob_by_name('operationcount', obs)) >= 30000):
                        is_meet = False
                elif condition == 'not large queries':
                    if not (int(self.get_ob_by_name('operationcount', obs)) <= 10000):
                        is_meet = False
                elif condition == 'insert or update':
                    if not (float(self.get_ob_by_name('insertproportion', obs)) + float(
                            self.get_ob_by_name('updateproportion', obs)) >= 0.8):
                        is_meet = False
                elif condition == 'analytics':
                    if not (float(self.get_ob_by_name('readproportion', obs)) >= 0.7):
                        is_meet = False

                elif condition == 'wal operation':
                    if not (int(self.get_ob_by_name('wal_records', obs)) > 30000):
                        is_meet = False
                elif condition == 'index':
                    if not (int(self.get_ob_by_name('idx_scan', obs)) > 100):
                        is_meet = False
                elif condition == 'no index':
                    if not (int(self.get_ob_by_name('idx_scan', obs)) <= 100):
                        is_meet = False
                elif condition == 'sort':
                    if not (int(self.get_ob_by_name('temp_bytes', obs)) >= 100):
                        is_meet = False
                elif condition == 'hash':
                    if not (int(self.get_ob_by_name('temp_bytes', obs)) >= 100):
                        is_meet = False
                elif condition == 'complex query':
                    if not (int(self.get_ob_by_name('operationcount', obs)) >= 10000):
                        is_meet = False
                elif condition == 'maintenance operations':
                    if not (int(self.get_ob_by_name('vacuum_count', obs)) >= 10):
                        is_meet = False
                elif condition == 'frequent access temporary table':
                    if not (int(self.get_ob_by_name('temp_bytes', obs)) >= 100):
                        is_meet = False

                # 还未达到
                elif condition == 'not Dynamic Shared Memory available':
                    if not False:
                        is_meet = False
                elif condition == 'large join':
                    if not False:
                        is_meet = False
                elif condition == 'not large join':
                    if not False:
                        is_meet = False
                else:
                    print('condition: {} is not included in all the conditions'.format(str(condition)))
                    # raise KeyError('condition: {} is not included in all the conditions'.format(str(condition)))
            return is_meet
        else:
            raise KeyError(f'{dbms_name} is not in mysql or pg')

    def get_ob_by_name(self, obs_name, obs):
        obs_keys_list = self.env.get_wrapper_attr('obs_keys_list')
        idx = obs_keys_list.index(obs_name)

        parsed_obs = []
        for item in obs:
            if isinstance(item, np.ndarray):
                parsed_obs.extend(item)
            else:
                parsed_obs.append(item)
        return parsed_obs[idx]

    # 通过conf的名称从action中获得值和idx
    def get_conf_by_name(self, conf_name, action):
        conf_keys_list = self.env.get_wrapper_attr('conf_keys_list')
        idx = conf_keys_list.index(conf_name)
        return action[idx], idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', help='Training Configuration', type=str)
    parser.add_argument('--eval', help='Evaluation mode', action='store_true', default=False)
    parser.add_argument('--collect', help='Collect Demonstration Data', action='store_true', default=False)
    parser.add_argument('-n_collect', help='Number of episode for demo collection', type=int, default=100)

    args = parser.parse_args()
    conf_path = args.conf

    trainer = RLTrainer(conf_path, args.eval)
    if args.eval:
        trainer.eval(save_fig=False)
    elif args.collect:
        trainer.collect_demo(args.n_collect)
    else:
        if trainer.conf.pretrain_demo:
            trainer.pretrain()
        trainer.train()


def analysis():
    # from numba import njit
    import matplotlib.pyplot as plt
    # @njit
    def calc_ewma_reward(reward):
        reward_new = np.zeros(len(reward) + 1)
        reward_new[0] = -50  # Min reward of the env
        ewma_reward = -50  # Min reward of the env
        idx = 1
        for r in reward:
            ewma_reward = 0.05 * r + (1 - 0.05) * ewma_reward
            reward_new[idx] = ewma_reward
            idx += 1
        return reward_new

    from matplotlib import colors as cl
    global_colors = [cl.cnames['aqua'], cl.cnames['orange']]

    configs = [
        's0.yaml',
        's1.yaml',
    ]
    show_names = [
        'No Demo (s0.yaml)',
        'With Demo (s1.yaml)',
    ]
    conf_base = './config'

    data_plot = {}
    for c, name in zip(configs, show_names):
        full_conf = load_conf(opj(conf_base, c))
        conf = full_conf.train_config
        progress_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progress')
        result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
        tp = TrainingProgress(progress_dir, result_dir, conf.exp_name)
        tp.restore_progress(1800)
        reward = tp.get_step_data('total_reward', 'episode', 1, 1801)
        ewma_reward = calc_ewma_reward(np.asarray(reward))
        data_plot[name] = np.asarray([0] + reward)
        data_plot[name + '-ewma'] = ewma_reward
        print('Done Processing {},avg_step={}'.format(name, tp.get_step_data('Mean Length', 'eval', 1800, 1801, 1)))

    fig = plt.figure(dpi=300, figsize=(6, 3))
    fig.suptitle('Total Reward-{}'.format('FetchReach-v1'))
    x_ticks = list(range(0, 1800 + 1, 1))
    # for i, (k, v) in enumerate(append_dict.items()):
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True)
    # ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelsize=4)
    ax.yaxis.set_tick_params(labelsize=4)

    c_idx = 0
    for name in show_names:
        color = global_colors[c_idx]
        v1 = data_plot[name]
        ax.plot(x_ticks, v1, linewidth=1, color=color, alpha=0.2)
        v2 = data_plot[name + '-ewma']
        ax.plot(x_ticks, v2, label=name, linewidth=1, color=color)
        c_idx += 1
        ax.legend(fontsize='x-small', loc='lower right')
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('./plot_-{}.jpg'.format('FetchReach-v1'))
    plt.clf()
    plt.close(fig)


if __name__ == '__main__':
    os.putenv('DISPLAY', ':0')
    main()
    # analysis()
