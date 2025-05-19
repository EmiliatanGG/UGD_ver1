import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision


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
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
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