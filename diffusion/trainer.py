import time

import gin
from typing import Optional, Sequence, Tuple
import torch
from multiprocessing import cpu_count
import pathlib
import tqdm
import torch.nn as nn
import numpy as np

from accelerate import Accelerator
from ema_pytorch import EMA
import wandb
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from diffusion.train_diffuser import SimpleDiffusionGenerator
from SimpleCQL.replay_buffer import sample
from SimpleCQL.replay_buffer import batch_to_torch, get_d4rl_dataset, subsample_batch, subsample_batch_diffusion

def info_nce_loss(z1, z2,device):
    batch_size = z1.size(0)

    # 计算相似度矩阵
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / 0.1

    # 构建正样本对
    sim_ij = torch.diag(sim, batch_size)
    sim_ji = torch.diag(sim, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    # 构建负样本掩码
    mask = (~torch.eye(2 * batch_size, dtype=torch.bool)).float().to(device)
    negatives = sim * mask

    # 计算损失
    numerator = torch.exp(positives)
    denominator = negatives.exp().sum(dim=1)
    loss = -torch.log(numerator / denominator).mean()
    return loss


def cycle(dl):
    while True:
        for data in dl:
            yield data


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def sample_schedule(num_sample_steps=None, device=None,
            epsilon = 1e-20,
            rho = 7,
            sigma_min: float = 0.002,  # min noise level
            sigma_max: float = 80,  # max noise level
):
    num_sample_steps = default(num_sample_steps, num_sample_steps)

    N = num_sample_steps
    inv_rho = 1 / rho

    steps = torch.arange(num_sample_steps, device=device, dtype=torch.float32)
    sigmas = (sigma_max ** inv_rho + steps / (N - 1 + epsilon) * (
            sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho

    sigmas = F.pad(sigmas, (0, 1), value=0.)  # last step is sigma value of 0.
    return sigmas


@gin.configurable
class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            dataset: Optional[torch.utils.data.Dataset] = None,
            train_batch_size: int = 16,
            small_batch_size: int = 16,
            gradient_accumulate_every: int = 1,
            train_lr: float = 1e-4,
            lr_scheduler: Optional[str] = None,
            train_num_steps: int = 100000,
            ema_update_every: int = 10,
            ema_decay: float = 0.995,
            adam_betas: Tuple[float, float] = (0.9, 0.99),
            save_and_sample_every: int = 10000,
            weight_decay: float = 0.,
            results_folder: str = './results',
            amp: bool = False,
            fp16: bool = False,
            split_batches: bool = True,
    ):
        super().__init__()
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp
        self.model = diffusion_model

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Number of trainable parameters: {num_params}.')

        self.save_and_sample_every = save_and_sample_every
        self.train_num_steps = train_num_steps
        self.gradient_accumulate_every = gradient_accumulate_every

        if dataset is not None:
            # If dataset size is less than 800K use the small batch size
            if len(dataset) < int(8e5):
                self.batch_size = small_batch_size
            else:
                self.batch_size = train_batch_size
            print(f'Using batch size: {self.batch_size}')
            # dataset and dataloader
            dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())
            dl = self.accelerator.prepare(dl)
            self.dl = cycle(dl)
        else:
            # No dataloader, train batch by batch
            self.batch_size = train_batch_size
            self.dl = None

        # optimizer, make sure that the bias and layer-norm weights are not decayed
        no_decay = ['bias', 'LayerNorm.weight', 'norm.weight', '.g']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        self.opt = torch.optim.AdamW(optimizer_grouped_parameters, lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.results_folder = pathlib.Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        if lr_scheduler == 'linear':
            print('using linear learning rate scheduler')
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt,
                lambda step: max(0, 1 - step / train_num_steps)
            )
        elif lr_scheduler == 'cosine':
            print('using cosine learning rate scheduler')
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt,
                train_num_steps
            )
        else:
            self.lr_scheduler = None

        self.model.normalizer.to(self.accelerator.device)
        self.ema.ema_model.normalizer.to(self.accelerator.device)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone: int):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    # Train for the full number of steps.
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = (next(self.dl)[0]).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')
                wandb.log({
                    'step': self.step,
                    'loss': total_loss,
                    'lr': self.opt.param_groups[0]['lr']
                })

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.save(self.step)

                pbar.update(1)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        accelerator.print('training complete')

    # Allow user to pass in external data.
    def train_on_batch(
            self,
            data: torch.Tensor,
            use_wandb=True,
            splits=1,  # number of splits to split the batch into
            **kwargs,
    ):
        accelerator = self.accelerator
        device = accelerator.device
        data = data.to(device)

        total_loss = 0.
        if splits == 1:
            with self.accelerator.autocast():
                loss = self.model(data, **kwargs)
                total_loss += loss.item()
            self.accelerator.backward(loss)
        else:
            assert splits > 1 and data.shape[0] % splits == 0
            split_data = torch.split(data, data.shape[0] // splits)

            for idx, d in enumerate(split_data):
                with self.accelerator.autocast():
                    # Split condition as well
                    new_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor):
                            new_kwargs[k] = torch.split(v, v.shape[0] // splits)[idx]
                        else:
                            new_kwargs[k] = v

                    loss = self.model(d, **new_kwargs)
                    loss = loss / splits
                    total_loss += loss.item()
                self.accelerator.backward(loss)

        accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        if use_wandb:
            wandb.log({
                'step': self.step,
                'loss': total_loss,
                'lr': self.opt.param_groups[0]['lr'],
            })

        accelerator.wait_for_everyone()

        self.opt.step()
        self.opt.zero_grad()

        accelerator.wait_for_everyone()

        self.step += 1
        if accelerator.is_main_process:
            self.ema.to(device)
            self.ema.update()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self.save(self.step)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return total_loss

@gin.configurable
class REDQTrainer(Trainer):
    def __init__(
            self,
            diffusion_model,
            ebm_batch_size: int = 256,
            train_batch_size: int = 16,
            gradient_accumulate_every: int = 1,
            train_lr: float = 1e-4,
            lr_scheduler: Optional[str] = None,
            train_num_steps: int = 10000,
            ema_update_every: int = 10,
            ema_decay: float = 0.995,
            adam_betas: Tuple[float, float] = (0.9, 0.99),
            save_and_sample_every: int = 10000,
            weight_decay: float = 0.,
            results_folder: str = './results',
            amp: bool = False,
            fp16: bool = False,
            split_batches: bool = True,
            model_terminals: bool = False,
            args=None,
            rew_model=None,
            rew_model_optim=None
    ):
        super().__init__(
            diffusion_model,
            #dataset=None,
            train_batch_size=train_batch_size,
            gradient_accumulate_every=gradient_accumulate_every,
            train_lr=train_lr,
            lr_scheduler=lr_scheduler,
            train_num_steps=train_num_steps,
            ema_update_every=ema_update_every,
            ema_decay=ema_decay,
            adam_betas=adam_betas,
            save_and_sample_every=save_and_sample_every,
            weight_decay=weight_decay,
            results_folder=results_folder,
            amp=amp,
            fp16=fp16,
            split_batches=split_batches,
        )

        self.model_terminals = model_terminals
        self.ebm_batch_size = ebm_batch_size

        #self.args = args
        self.rew_model = rew_model
        self.rew_model_optim = rew_model_optim
        self.device='cuda'


    def train_rew_model(self, buffer, num_steps: Optional[int] = None):
        num_steps = num_steps or self.train_num_steps
        for j in range(num_steps):
            states, actions, rewards, next_states, dones, mc_returns = buffer.sample(self.batch_size)
            data = [states, actions]
            accelerator = self.accelerator
            device = accelerator.device
            data = torch.cat(data, dim=1).to(device)

            pred_rew = self.rew_model(data)
            rewards = rewards.to(device)
            assert (pred_rew.shape == rewards.shape)
            loss = F.mse_loss(pred_rew, rewards)

            self.rew_model_optim.zero_grad()
            loss.backward()
            self.rew_model_optim.step()
            if j % 1000 == 0:
                print(f'[{j}/{num_steps}] loss: {loss:.4f}')


    def train_classifier(self, guide_buffer,dataset,classifier,c_optim):
        print("###################################################Classifier Trainning###################################################")
        dataset_batch=subsample_batch(dataset,100000)
        criterion = nn.BCELoss()  # 二分类交叉熵损失
        l2_lambda = 0.001  # 正则化强度
        # 从guide_buffer和dataset中提取states和actions
        if guide_buffer['observations'].shape[0]>0:
            guide_states = [torch.tensor(state, dtype=torch.float32) for state in guide_buffer['observations']]
            guide_actions = [torch.tensor(action, dtype=torch.long) for action in guide_buffer['actions']]
            guide_labels = torch.ones(len(guide_buffer['observations']), dtype=torch.float)  # 标签为1

            dataset_states = [torch.tensor(state, dtype=torch.float32) for state in dataset_batch['observations']]
            dataset_actions = [torch.tensor(action, dtype=torch.long) for action in dataset_batch['actions']]
            dataset_labels = torch.zeros(len(dataset_batch['observations']), dtype=torch.float)  # 标签为0

            combined_states = guide_states + dataset_states
            combined_actions = guide_actions + dataset_actions
            # 拼接每个张量对
            combined_tensors = [torch.cat((t1, t2), dim=0) for t1, t2 in zip(combined_states, combined_actions)]
            # 合并数据
            final_tensor = torch.stack(combined_tensors, dim=0)

            labels = torch.cat((guide_labels, dataset_labels))

            # 创建数据加载器
            dataset_tensor = TensorDataset(final_tensor, labels)

            dataloader = DataLoader(dataset_tensor, batch_size=5000, shuffle=True)

            # 训练模型
            for epoch in range(20):
                classifier.train()
                total_loss = 0
                for input, labels in dataloader:
                    # 将数据转移到设备上（如GPU）
                    input, labels = input.to(self.device), labels.to(self.device)

                    # 清零梯度
                    c_optim.zero_grad()

                    # 前向传播
                    outputs = classifier(input)

                    # 计算损失
                    loss = criterion(outputs, labels.unsqueeze(1))
                    l2_norm = sum(torch.sum(param ** 2) for param in classifier.parameters())
                    loss = loss + l2_lambda * l2_norm

                    # 反向传播
                    loss.backward()

                    # 更新权重
                    c_optim.step()

                    # 累加损失
                    total_loss += loss.item()

                # 打印每个epoch的损失
                print(f'Epoch {epoch + 1}/{20}, Loss: {total_loss / len(dataloader)}')
        else:
            return 1
        return 0

    def train_classifier(self, guide_buffer,dataset,classifier,c_optim):
        print("###################################################Classifier Trainning###################################################")
        dataset_batch=subsample_batch(dataset,100000)
        criterion = nn.BCELoss()  # 二分类交叉熵损失
        l2_lambda = 0.001  # 正则化强度
        # 从guide_buffer和dataset中提取states和actions
        if guide_buffer['observations'].shape[0]>0:
            guide_states = [torch.tensor(state, dtype=torch.float32) for state in guide_buffer['observations']]
            guide_actions = [torch.tensor(action, dtype=torch.long) for action in guide_buffer['actions']]
            guide_labels = torch.ones(len(guide_buffer['observations']), dtype=torch.float)  # 标签为1

            dataset_states = [torch.tensor(state, dtype=torch.float32) for state in dataset_batch['observations']]
            dataset_actions = [torch.tensor(action, dtype=torch.long) for action in dataset_batch['actions']]
            dataset_labels = torch.zeros(len(dataset_batch['observations']), dtype=torch.float)  # 标签为0

            combined_states = guide_states + dataset_states
            combined_actions = guide_actions + dataset_actions
            # 拼接每个张量对
            combined_tensors = [torch.cat((t1, t2), dim=0) for t1, t2 in zip(combined_states, combined_actions)]
            # 合并数据
            final_tensor = torch.stack(combined_tensors, dim=0)

            labels = torch.cat((guide_labels, dataset_labels))

            # 创建数据加载器
            dataset_tensor = TensorDataset(final_tensor, labels)

            dataloader = DataLoader(dataset_tensor, batch_size=500, shuffle=True)

            # 训练模型
            for epoch in range(20):
                classifier.train()
                total_loss = 0
                for input, labels in dataloader:
                    # 将数据转移到设备上（如GPU）
                    input, labels = input.to(self.device), labels.to(self.device)

                    # 清零梯度
                    c_optim.zero_grad()

                    # 前向传播
                    outputs = classifier(input)

                    # 计算损失
                    loss = criterion(outputs, labels.unsqueeze(1))
                    l2_norm = sum(torch.sum(param ** 2) for param in classifier.parameters())
                    loss = loss + l2_lambda * l2_norm

                    # 反向传播
                    loss.backward()

                    # 更新权重
                    c_optim.step()

                    # 累加损失
                    total_loss += loss.item()

                # 打印每个epoch的损失
                print(f'Epoch {epoch + 1}/{20}, Loss: {total_loss / len(dataloader)}')
        else:
            return 1
        return 0

    def train_from_redq_buffer(self, buffer, online_buffer = None, num_steps: Optional[int] = None):
        num_steps = num_steps or self.train_num_steps
        for j in range(num_steps):
            if online_buffer is not None:
                states, actions, rewards, next_states, dones, mc_returns = online_buffer.sample(int(0.5*self.batch_size))
                states_off, actions_off, rewards_off, next_states_off, dones_off, mc_returns_off = buffer.sample(int(0.5*self.batch_size))
                states = torch.cat([states, states_off], dim=0)
                actions = torch.cat([actions, actions_off], dim=0)
                rewards = torch.cat([rewards, rewards_off], dim=0)
                next_states = torch.cat([next_states, next_states_off], dim=0)
            else:
                #states, actions, rewards, next_states, dones, mc_returns = buffer.sample(self.batch_size)
                sample_batch=sample(buffer,self.batch_size)
                states, actions, rewards, next_states, dones =torch.from_numpy(sample_batch['observations']),torch.from_numpy(sample_batch['actions']),torch.from_numpy(sample_batch['rewards'].reshape(-1, 1)),torch.from_numpy(sample_batch['next_observations']),torch.from_numpy(sample_batch['dones'])
            data = [states, actions, rewards, next_states]
            if self.model_terminals:
                data.append(dones)
            data = torch.cat(data, dim=1)

            loss = self.train_on_batch(data, use_wandb=False)
            if j % 1000 == 0:
                print(f'[{j}/{num_steps}] loss: {loss:.4f}')

    def train_transition_from_redq_buffer(self, buffer, num_steps: Optional[int] = None):
        num_steps = num_steps or self.train_num_steps
        for j in range(num_steps):
            states, actions, rewards, next_states, dones, mc_returns = buffer.sample(self.batch_size)
            state_actions = torch.cat([states, actions], dim=1)
            data = [rewards, next_states]
            if self.model_terminals:
                data.append(dones)
            data = torch.cat(data, dim=1)
            loss = self.train_on_batch(data, use_wandb=False, cond=state_actions)
            if j % 1000 == 0:
                print(f'[{j}/{num_steps}] loss: {loss:.4f}')

    def update_normalizer(self, buffer, device=None):
        data = make_inputs_from_replay_buffer(buffer, self.model_terminals)
        data = torch.from_numpy(data).float()
        self.model.normalizer.reset(data)
        self.ema.ema_model.normalizer.reset(data)
        if device:
            self.model.normalizer.to(device)
            self.ema.ema_model.normalizer.to(device)


# Make transition dataset from REDQ replay buffer.
def make_inputs_from_replay_buffer(
        replay_buffer,
        model_terminals: bool = False,
) -> np.ndarray:
    obs=replay_buffer['observations']
    actions = replay_buffer['actions']
    next_obs = replay_buffer['next_observations']
    rewards = replay_buffer['rewards'].reshape(-1, 1)
    '''ptr_location = replay_buffer._pointer
    obs = replay_buffer._states[:ptr_location].cpu().detach().numpy()
    actions = replay_buffer._actions[:ptr_location].cpu().detach().numpy()
    next_obs = replay_buffer._next_states[:ptr_location].cpu().detach().numpy()
    rewards = replay_buffer._rewards[:ptr_location].cpu().detach().numpy()'''
    inputs = [obs, actions, rewards, next_obs]
    '''if model_terminals:
        terminals = replay_buffer._dones[:ptr_location].astype(np.float32)
        inputs.append(terminals[:, None].cpu().detach().numpy())'''
    return np.concatenate(inputs, axis=1)
