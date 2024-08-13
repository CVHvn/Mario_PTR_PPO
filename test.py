import argparse
import torch

from src.environment import *
from src.memory import *
from src.model import *
from src.agent import *

def get_args():
    parser = argparse.ArgumentParser(
        """PPO implement to playing Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument('--num_envs', type=int, default=16, help='Number of environment')
    parser.add_argument('--learn_step', type=int, default=512, help='Number of steps between training model')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_epoch', type=int, default=2, help='epoch')
    parser.add_argument('--off_epoch', type=int, default=8, help='epoch')

    parser.add_argument('--learning_rate', type=float, default=7e-5)
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor for rewards')
    parser.add_argument('--V_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy loss coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda coefficient')
    parser.add_argument('--clip_param', type=float, default=0.2, help='Clip coefficient for policy loss')
    parser.add_argument('--target_kl', type=float, default=0.05, help='Set target_kl = None if you do not use this. Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping.')

    parser.add_argument('--per_buffer_size', type=int, default=256, help='PER buffer size')
    parser.add_argument('--per_eps', type=float, default=1e-2, help='Eps in PER')
    parser.add_argument('--per_alpha', type=float, default=0.7, help='Alpha in PER')
    parser.add_argument('--per_beta', type=float, default=0.4, help='Beta in PER')
    parser.add_argument('--eps_marg', type=float, default=0.2, help='eps marg use to clip importance ratio weight')

    parser.add_argument("--is_normalize_advantage", type=bool, default=False, help = "Use normalize advantage or not")
    parser.add_argument("--V_loss_type", type=str, default="huber", help = "Use mse or huber loss for value network")

    parser.add_argument('--total_step', type=int, default=int(5e6), help='Total step for training')
    parser.add_argument('--save_model_step', type=int, default=int(1e5), help='Number of steps between saving model')
    parser.add_argument('--save_figure_step', type=int, default=int(1e3), help='Number of steps between testing model')
    parser.add_argument('--total_step_or_episode', type=str, default='step', help='choice stop training base on total step or total episode')
    parser.add_argument('--total_episode', type=int, default=None, help='Total episodes for training')

    parser.add_argument("--action_dim", type=int, default=12, help='12 if set action_type to complex else 7')
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--state_dim", type=tuple, default=(4, 84, 84))

    parser.add_argument("--save_dir", type=str, default="")
    args = parser.parse_args()
    return args


def test(config):
    model = Model(config.state_dim, config.action_dim)
    agent = Agent(world = config.world, stage = config.stage, action_type = config.action_type, envs = None, num_envs = config.num_envs,
              state_dim = config.state_dim, action_dim = config.action_dim, save_dir = config.save_dir,
              save_model_step = config.save_model_step, save_figure_step = config.save_figure_step, learn_step = config.learn_step,
              total_step_or_episode = config.total_step_or_episode, total_step = config.total_step, total_episode = config.total_episode,
              model = model, old_model = None, gamma = config.gamma, learning_rate = config.learning_rate,
              entropy_coef = config.entropy_coef, V_coef = config.V_coef,
              max_grad_norm = config.max_grad_norm, clip_param = config.clip_param, batch_size = config.batch_size,
              num_epoch = config.num_epoch, is_normalize_advantage = config.is_normalize_advantage, V_loss_type = config.V_loss_type,
              target_kl = config.target_kl, gae_lambda = config.gae_lambda,  per_buffer_size = config.per_buffer_size,
              per_eps = config.per_eps, per_alpha = config.per_alpha, per_beta = config.per_beta, eps_marg = config.eps_marg, off_epoch = config.off_epoch,
              device = "cuda" if torch.cuda.is_available() else "cpu")
    agent.load_model(config.pretrained_model)
    agent.save_figure()

if __name__ == "__main__":
    config = get_args()
    test(config)
