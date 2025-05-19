import sys
from collections import deque
current_dir = '/The location of this file in your system'
sys.path.insert(0, current_dir)

import json
import os
import time
from copy import deepcopy
import uuid
import torch.optim as optim
import numpy as np
import pprint

import gym
import torch
import d4rl
import argparse
import gin

import absl.app
import absl.flags

import torch.nn.functional as F
from typing import Union, Tuple

from SimpleCQL.conservative_sac import ConservativeSAC
from SimpleCQL.replay_buffer import batch_to_torch, get_d4rl_dataset, subsample_batch, subsample_batch_diffusion
from SimpleCQL.model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from SimpleCQL.sampler import StepSampler, TrajSampler
from SimpleCQL.utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from SimpleCQL.utils import WandBLogger
from viskit.logging import logger, setup_logger


from diffusion.trainer import REDQTrainer
from diffusion.train_diffuser import SimpleDiffusionGenerator
from diffusion.utils import construct_diffusion_model
from diffusion.denoiser_network import ResidualMLPDenoiser
import torch
import torch.nn as nn
import torchvision



FLAGS_DEF = define_flags_with_default(
    env='walker2d-medium-expert-v2',#hopper-medium-expert-v2
    max_traj_length=1000,
    seed=42,
    device='cuda',
    save_model=False,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=1000,
    bc_epochs=0,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=10,

    cql=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),

    normalize= True,

    #diffusion
    energy_hidden_layers = 3,
    ebm_activation = str("relu"),
    ebm_layer_type = "MLP",
    ebm_spectral_norm = True,
    ebm_lr = float(1e-3),
    model_terminals =False,
    energy_train_epoch = 20,
    log_name = "gd_cql",
    num_samples = 100000,
    num_negative_sample = 10,
    grad_clip = float(1),
    batch_size_diffusion = int(256),
    offline_mixing_ratio = 0.5,
    policy_guide=True,
    state_guide=True,
    test_divergence=False,
    ope_clip=0.1,
    te_clip=0.1,
    pe_clip=0.1,
    normalizer_type = 'standard'
)

class BasicClassifier(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        input_dim = obs_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, sa):

        # 打印模型权重的设备信息
        return self.net(sa)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, action_dim, output_dim, num_heads, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.output_dim = output_dim

        # 输入嵌入层
        self.state_embedding = nn.Linear(input_dim, dim_feedforward)
        self.action_embedding = nn.Linear(action_dim, dim_feedforward)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.next_state_head = nn.Linear(dim_feedforward, output_dim)
        self.reward_head = nn.Linear(dim_feedforward, 1)
        self.done_head = nn.Linear(dim_feedforward, 1)

    def forward(self, state, action):
        # 嵌入输入
        state_emb = self.state_embedding(state)
        action_emb = self.action_embedding(action)

        # 合并状态和动作嵌入
        input_seq = state_emb + action_emb

        # Transformer 编码器
        input_seq = input_seq.unsqueeze(1)  # 添加序列维度
        transformer_output = self.transformer_encoder(input_seq)

        # 取出编码器的输出
        transformer_output = transformer_output.squeeze(1)

        # 预测下一个状态、奖励和是否完成标志
        next_state = self.next_state_head(transformer_output)
        reward = self.reward_head(transformer_output)
        done = self.done_head(transformer_output)

        return next_state, reward, done

def test_authenticity(diffusion,env):
    env = gym.make(env)
    difference=[]
    # 打开文件进行写入
    with open('authenticity_dip.txt', 'w') as f:
        for i in range(len(diffusion['observations'])):
            env.reset()
            initial_state = diffusion['observations'][i]
            # 将diffusion生成的obs装载到环境
            env.data.qpos[1:] = initial_state[:8]
            env.data.qvel[:] = initial_state[8:]
            # 进行将diffusion生成的actions
            action = diffusion['actions'][i]
            next_state, reward, done, _ = env.step(action)
            #tranfomer生成
            #k1=torch.tensor([diffusion['observations'][i]], dtype=torch.float32)
            #k2=torch.tensor([diffusion['actions'][i]], dtype = torch.float32)
            #t_n_state,t_reward, t_done=trans(k1,k2)
            #判断生成差距
            next_state = torch.tensor(next_state, dtype=torch.float32)
            next_state1 = torch.tensor(diffusion['next_observations'][i], dtype=torch.float32)
            distance1 = torch.sqrt(torch.sum((next_state - next_state1) ** 2))
            distance2 = abs(reward - diffusion['rewards'][i])
            distance3 = abs(done - diffusion['dones'][i])
            #distance4 = torch.sqrt(torch.sum((next_state - t_n_state) ** 2))
            #distance5 = abs(reward - t_reward)
            #distance6 = abs(done - t_done)
            total = distance1 + distance2 + distance3
            #total2 = distance4 + distance5 #+ distance6
            difference.append([distance1, distance2, distance3, total])
            f.write("%s\n" % [distance1, distance2, distance3, total])
    return difference

def gen_acc(diffusion,env,classifier):
    env = gym.make(env)
    difference=[]
    # 打开文件进行写入
    total=0
    total1 = 0
    ns=0
    r=0
    ob=[]
    act=[]
    for i in range(len(diffusion['observations'])):
        env.reset()
        initial_state = diffusion['observations'][i]
        # 将diffusion生成的obs装载到环境
        env.data.qpos[1:] = initial_state[:8]
        env.data.qvel[:] = initial_state[8:]
        # 进行将diffusion生成的actions
        action = diffusion['actions'][i]
        next_state, reward, done, _ = env.step(action)
        # tranfomer生成
        # k1=torch.tensor([diffusion['observations'][i]], dtype=torch.float32)
        # k2=torch.tensor([diffusion['actions'][i]], dtype = torch.float32)
        # t_n_state,t_reward, t_done=trans(k1,k2)
        # 判断生成差距
        next_state = torch.tensor(next_state, dtype=torch.float32)
        next_state1 = torch.tensor(diffusion['next_observations'][i], dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        done_tensor = torch.tensor(done, dtype=torch.float32)
        reward1_tensor = torch.tensor(diffusion['rewards'][i], dtype=torch.float32)
        done1_tensor = torch.tensor(diffusion['dones'][i], dtype=torch.float32)
        state_error_squared = (next_state - next_state1) ** 2
        reward_error_squared = (reward_tensor - reward1_tensor) ** 2
        #done_error_squared = (done_tensor - done1_tensor) ** 2
        ns+=torch.sqrt(torch.sum((next_state - next_state1) ** 2))/next_state
        r+=torch.sqrt(torch.sum((reward_tensor - reward1_tensor) ** 2))/next_state

        mse = torch.mean(
            torch.cat([state_error_squared.view(-1), reward_error_squared.view(-1)]))#, done_error_squared.view(-1)
        print("mse = ", mse)

        rmse = torch.sqrt(mse)
        true_values = torch.cat([next_state.view(-1), reward_tensor.view(-1)])#, done_tensor.view(-1)
        true_abs_mean = torch.mean(torch.abs(true_values))
        rrmse = rmse / true_abs_mean
        if rrmse <= 0.15:
            total += 1
        if rrmse <= 0.3:
            total1 += 1
        if rrmse > 0.3:
            ob.append(diffusion['observations'][i])
            act.append(diffusion['actions'][i])
    buffer={'observations':ob,'actions':act}
    pri=label_classifier(buffer,classifier)
    pri_total=label_classifier(diffusion,classifier)
    acc=total/len(diffusion['observations'])
    acc1 = total1 / len(diffusion['observations'])
    print('acc=',acc,acc1)
    print(ns/len(diffusion['observations']))
    print(r / len(diffusion['observations']))
    print(torch.mean(pri))
    print(torch.mean(pri_total))
    return acc,acc1,pri,pri_total

def label_classifier(buffer,classifier):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    states = [torch.tensor(state, dtype=torch.float32).to(device) for state in buffer['observations']]
    actions = [torch.tensor(action, dtype=torch.long).to(device) for action in buffer['actions']]
    combined_tensors = [torch.cat((t1, t2), dim=0) for t1, t2 in zip(states, actions)]
    final_tensor = torch.stack(combined_tensors, dim=0)
    # 将最终张量输入到分类器中
    with torch.no_grad():  # 禁用梯度计算，节省计算资源
        classifier.eval()  # 将分类器设置为评估模式
        predictions = classifier(final_tensor)
    # 返回预测结果
    return predictions

def draw_distribution(diffusion,input_name):
    # 打开文件进行写入
    with open(f"{input_name}.txt", 'w') as f:
        for i in range(len(diffusion['observations'])):
            if input_name=='guide':
                f.write(
                    "%s\n" % [diffusion['observations'][i], diffusion['actions'][i]])
            else:
                f.write(
                    "%s\n" % [diffusion['observations'][i], diffusion['actions'][i]])
    return 1




def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def main(argv):
    #print(torch.version.cuda)  # PyTorch 使用的 CUDA 版本
    #print(torch.backends.cudnn.version())  # cuDNN 版本
    #print(111111111111111111111111111111111111111)
    FLAGS = absl.flags.FLAGS
    acc_w=[]
    acc_w1= []

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    set_random_seed(FLAGS.seed)
    average_target_q_list=[]
    average_return_list=[]

    env=gym.make(FLAGS.env)
    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)
    dataset = get_d4rl_dataset(eval_sampler.env)
    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)
    print(eval_sampler.env.observation_space.shape[0],eval_sampler.env.action_space.shape[0])
    #print(1)

    #draw_distribution(dataset, 'offline')
    #print(2)

    batch_size_offline = int(FLAGS.batch_size_diffusion * FLAGS.offline_mixing_ratio)

    if FLAGS.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    policy = TanhGaussianPolicy(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
        orthogonal_init=FLAGS.orthogonal_init,
    )

    qf1 = FullyConnectedQFunction(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunction(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf2 = deepcopy(qf2)


    diff_dims = eval_sampler.env.observation_space.shape[0] + eval_sampler.env.action_space.shape[0] + 1 + \
                eval_sampler.env.observation_space.shape[0]
    if FLAGS.model_terminals:
        diff_dims += 1

    inputs = torch.zeros((128, diff_dims)).float()
    rew_model = None
    rew_model_optim = None

    #这边初始化一个引导池（存放策略探索的状态）
    guide_buffer={}
    diffusion_replay_buffer = get_d4rl_dataset(eval_sampler.env)#每10个epoch生成一次数据，每一次生成清空diffusion_replay_buffer ps:最初的10个epoch没有diffusion数据，因此用离线数据代替


    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    names = FLAGS.env.split('-')
    env_name = names[0]
    other_name = ''
    for name in names[1:-1]:
        other_name = other_name + name + '-'
    other_name = other_name[:-1]

    if env_name == "hopper":
        s_dim=11
        a_dim=3
    elif env_name == "halfcheetah":
        s_dim=17
        a_dim=6
    elif env_name == "walker2d":
        s_dim = 17
        a_dim = 6

    gin_config_files = '/The location of this file in your system/configs/ugd/' + env_name + '/' + other_name + '.gin'

    gin.parse_config_files_and_bindings([gin_config_files], [])


    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = ConservativeSAC(FLAGS.cql, policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    viskit_metrics = {}

    # Train diffusion model
    diffusion_trainer = REDQTrainer(
        construct_diffusion_model(
            inputs=inputs,
            skip_dims=[
                eval_sampler.env.observation_space.shape[0] + eval_sampler.env.action_space.shape[0]],
            disable_terminal_norm=FLAGS.model_terminals,
        ),
        results_folder=os.path.join("/The location of this file in your system/logs", FLAGS.log_name),
        model_terminals=FLAGS.model_terminals,
        rew_model=rew_model,
        rew_model_optim=rew_model_optim
    )
    diffusion_trainer.update_normalizer(dataset, device=FLAGS.device)
    diffusion_trainer.train_from_redq_buffer(dataset)  # 训练diffusion


    for epoch in range(FLAGS.n_epochs):
        metrics = {'epoch': epoch}
        print(guide_buffer)

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                #guide_buffer = {}  # 清空引导池（仅初始CQL）
                batch = subsample_batch_diffusion(dataset,diffusion_replay_buffer,batch_size_offline,FLAGS.batch_size-batch_size_offline)
                #batch =subsample_batch(dataset,FLAGS.batch_size)
                batch = batch_to_torch(batch, FLAGS.device)
                output_metrics, guide_buffer=sac.train(batch, guide_buffer, epoch, bc=epoch < FLAGS.bc_epochs)
                metrics.update(prefix_metrics(output_metrics, 'sac'))

        average_target_q_list.append(metrics['sac/average_target_q'])

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:#每10个epoch
                #eval
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                metrics['average_normalizd_return'] = np.mean(
                    [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                )
                if FLAGS.save_model:
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, 'model.pkl')

                average_return_list.append(metrics['average_return'])
                #diffusion
                if epoch !=0:
                    if epoch !=9:
                        batch = subsample_batch(diffusion_replay_buffer,10000)#随机选择部分diffusion生成的数据添加到离线数据集，以避免后续CQL对这些分布过度惩罚
                        for i in batch.keys():
                            dataset[i]=np.concatenate((dataset[i],batch[i]), axis=0)
                    # 使用示例
                    obs_dim = eval_sampler.env.observation_space.shape[0]
                    action_dim = eval_sampler.env.action_space.shape[0]
                    classifier = BasicClassifier(obs_dim, action_dim)
                    classifier.to(FLAGS.device)
                    classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)



                    fl=diffusion_trainer.train_classifier(guide_buffer,dataset,classifier,classifier_optimizer)#训练分类器
                    #fl=1
                    draw_distribution(guide_buffer, 'guide')
                    guide_buffer = {}  # 清空引导池
                    if rew_model is not None:
                        diffusion_trainer.train_rew_model(dataset)

                    diffusion_replay_buffer = {}

                    # Add samples to agent replay buffer
                    generator = SimpleDiffusionGenerator(env=env, ema_model=diffusion_trainer.ema.ema_model,
                                                         rew_model=rew_model)
                    if fl==1:
                        diffusion_batch = generator.sample(clip=FLAGS.grad_clip,
                                                           num_samples=FLAGS.num_samples)
                    else:
                        diffusion_batch = generator.sample(clip=FLAGS.grad_clip,
                                                           num_samples=FLAGS.num_samples,
                                                           classifier=classifier,s_dim=s_dim,
                                                           a_dim=a_dim)
                    dict_index = 0
                    for i in ['observations','actions','rewards','next_observations','dones']:
                        diffusion_replay_buffer.update({i: diffusion_batch[dict_index]})
                        dict_index = dict_index + 1

                    acc,acc1,pri,pri_total=gen_acc(diffusion_replay_buffer,FLAGS.env,classifier)
                    #torch.set_printoptions(threshold=float('inf'))
                    #print(pri)
                    #time.sleep(10000000)
                    #print(pri_total)
                    #torch.set_printoptions(threshold=1000)  # 默认阈值通常是 1000
                    #acc_w.append(acc)
                    #acc_w1.append(acc1)
                    draw_distribution(diffusion_replay_buffer,'diffusion')
                    #draw_distribution(dataset, 'offline')


                    #batch = replay_buffer.combine_replay_buffer(diffusion_replay_buffer, batch_size_offline,
                                                                #batch_size_online, FLAGS.device)#####################################

        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)



    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')

    # 打开文件进行写入
    with open('Average_Return_1.txt', 'w') as f:
        for item in average_return_list:
            f.write("%s\n" % item)


    # 打开文件进行写入
    with open('average_target_q_1.txt', 'w') as f:
        for item in average_target_q_list:
            f.write("%s\n" % item)

if __name__ == '__main__':
    absl.app.run(main)
