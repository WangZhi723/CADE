"""
Multi-Scale Feature Fusion Network
多尺度特征融合网络：结合CNN和LSTM/Attention提取时空特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import sys
sys.path.append('/home/zhi/Risk')
from config import Config


class CausalConv1d(nn.Module):
    """
    因果卷积层：确保 t 时刻的输出仅依赖于 t 及之前的输入
    防止信息泄露
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=self.padding,
            dilation=dilation
        )
    
    def forward(self, x):
        # x: (batch, channels, time)
        x = self.conv(x)
        # 移除右侧的未来信息
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class ShortTermCNN(nn.Module):
    """
    短期特征提取器（CNN）
    输入: 气象数据的短期窗口 (batch, channels, time_window)
    输出: 短期特征向量 (batch, feature_dim)
    
    数学形式:
        h_s^(l) = ReLU(BatchNorm(CausalConv1d(h_s^(l-1))))
        h_s = GlobalMaxPool(h_s^(L))
    """
    def __init__(self, 
                 input_channels=Config.CNN_INPUT_CHANNELS, 
                 filters=Config.CNN_FILTERS,
                 kernel_sizes=Config.CNN_KERNEL_SIZES,
                 dropout=Config.CNN_DROPOUT):
        super(ShortTermCNN, self).__init__()
        
        layers = []
        in_ch = input_channels
        
        for i, (out_ch, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            # 因果卷积 + BatchNorm + ReLU + Dropout
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        
        self.conv_layers = nn.Sequential(*layers)
        self.output_dim = filters[-1]
        
    def forward(self, x):
        """
        x: (batch, channels, time_window)
           channels: [GHI, temp, humidity, wind_speed, wind_dir_sin, wind_dir_cos]
        Returns: (batch, feature_dim)
        """
        # 卷积提取局部特征
        x = self.conv_layers(x)  # (batch, filters[-1], time_window')
        
        # 全局最大池化和平均池化
        x_max = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (batch, filters[-1])
        x_avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (batch, filters[-1])
        
        # 拼接两种池化结果
        x = torch.cat([x_max, x_avg], dim=1)  # (batch, 2*filters[-1])
        
        return x


class LongTermLSTM(nn.Module):
    """
    长期时序特征提取器（LSTM）
    输入: 历史光伏和系统状态的长期窗口 (batch, time_window, features)
    输出: 长期特征向量 (batch, feature_dim)
    
    数学形式 (LSTM 单元):
        f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # 遗忘门
        i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # 输入门
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ tanh(W_c · [h_{t-1}, x_t] + b_c)  # 细胞状态
        o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # 输出门
        h_t = o_t ⊙ tanh(c_t)  # 隐藏状态
    """
    def __init__(self, 
                 input_dim=Config.LSTM_INPUT_DIM,
                 hidden_dim=Config.LSTM_HIDDEN_DIM,
                 num_layers=Config.LSTM_NUM_LAYERS,
                 dropout=Config.LSTM_DROPOUT):
        super(LongTermLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # 单向，保持因果性
        )
        
        # Layer Normalization 稳定训练
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim
        
    def forward(self, x):
        """
        x: (batch, time_window, input_dim)
           input_dim: [P_pv, SoC, P_battery, load]
        Returns: (batch, hidden_dim)
        """
        # LSTM 前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, time_window, hidden_dim)
        # h_n: (num_layers, batch, hidden_dim)
        
        # 取最后一个时间步的输出
        x = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Layer Normalization
        x = self.layer_norm(x)
        
        return x


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制：动态融合多尺度特征
    
    数学形式:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    def __init__(self, 
                 embed_dim=Config.ATTENTION_DIM,
                 num_heads=Config.ATTENTION_HEADS):
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, key, value, mask=None):
        """
        query, key, value: (batch, seq_len, embed_dim)
        mask: (batch, seq_len, seq_len) or None
        Returns: (batch, seq_len, embed_dim)
        """
        batch_size = query.size(0)
        
        # 线性投影并分成多头
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: (batch, num_heads, seq_len, head_dim)
        
        # 计算注意力分数: QK^T / √d_k
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # attn_scores: (batch, num_heads, seq_len, seq_len)
        
        # 应用 mask（可选）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和: Attention(Q,K,V) = softmax(QK^T/√d_k) V
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: (batch, num_heads, seq_len, head_dim)
        
        # 拼接多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class MultiScaleFeatureFusion(nn.Module):
    """
    多尺度特征融合网络（核心模块）
    
    架构:
        1. 短期特征: CNN 提取气象数据的局部模式
        2. 长期特征: LSTM 提取历史状态的时序依赖
        3. 特征融合: Multi-Head Attention 动态加权融合
        4. 输出: 融合后的特征向量，用于策略网络
    
    数学形式:
        h_s = CNN(X_weather[:, t-k:t])        # 短期特征
        h_l = LSTM(X_history[:, t-T:t])       # 长期特征
        [h_s', h_l'] = Attention([h_s, h_l])  # 注意力融合
        z = MLP([h_s', h_l'])                 # 最终特征
    """
    def __init__(self, 
                 cnn_input_channels=Config.CNN_INPUT_CHANNELS,
                 lstm_input_dim=Config.LSTM_INPUT_DIM,
                 fusion_dim=Config.FUSION_DIM):
        super(MultiScaleFeatureFusion, self).__init__()
        
        # 1. 短期特征提取器 (CNN)
        self.short_term_cnn = ShortTermCNN(input_channels=cnn_input_channels)
        cnn_output_dim = self.short_term_cnn.output_dim * 2  # max + avg pooling
        
        # 2. 长期特征提取器 (LSTM)
        self.long_term_lstm = LongTermLSTM(input_dim=lstm_input_dim)
        lstm_output_dim = self.long_term_lstm.output_dim
        
        # 3. 投影到相同维度（为 Attention 做准备）
        self.cnn_projection = nn.Linear(cnn_output_dim, Config.ATTENTION_DIM)
        self.lstm_projection = nn.Linear(lstm_output_dim, Config.ATTENTION_DIM)
        
        # 4. 多头注意力融合
        self.attention = MultiHeadAttention(
            embed_dim=Config.ATTENTION_DIM,
            num_heads=Config.ATTENTION_HEADS
        )
        
        # 5. 融合后的全连接层
        self.fusion_mlp = nn.Sequential(
            nn.Linear(Config.ATTENTION_DIM * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        self.output_dim = fusion_dim
        
    def forward(self, short_term_input, long_term_input):
        """
        short_term_input: (batch, cnn_channels, short_window)
                         气象数据：[GHI, temp, humidity, wind_speed, ...]
        long_term_input: (batch, long_window, lstm_features)
                        历史状态：[P_pv, SoC, P_battery, load]
        
        Returns: 
            fusion_features: (batch, fusion_dim)
            attention_weights: (batch, num_heads, 2, 2) - 用于可视化
        """
        # 1. 提取短期特征
        h_short = self.short_term_cnn(short_term_input)  # (batch, cnn_output_dim)
        h_short = self.cnn_projection(h_short)  # (batch, attention_dim)
        
        # 2. 提取长期特征
        h_long = self.long_term_lstm(long_term_input)  # (batch, lstm_output_dim)
        h_long = self.lstm_projection(h_long)  # (batch, attention_dim)
        
        # 3. 拼接为序列用于 Attention
        # (batch, 2, attention_dim): [h_short, h_long]
        features = torch.stack([h_short, h_long], dim=1)
        
        # 4. Self-Attention 融合
        fused_features, attn_weights = self.attention(features, features, features)
        # fused_features: (batch, 2, attention_dim)
        
        # 5. 展平并通过 MLP
        fused_features = fused_features.view(fused_features.size(0), -1)  # (batch, 2*attention_dim)
        output = self.fusion_mlp(fused_features)  # (batch, fusion_dim)
        
        return output, attn_weights


class CustomFeatureExtractor(nn.Module):
    """
    自定义特征提取器：包装 MultiScaleFeatureFusion 以适配 Stable-Baselines3
    
    用于替换 SB3 中的默认 MLP 特征提取器
    """
    def __init__(self, observation_space, features_dim=Config.FUSION_DIM):
        super(CustomFeatureExtractor, self).__init__()
        
        self.features_dim = features_dim
        
        # 多尺度特征融合网络
        self.feature_fusion = MultiScaleFeatureFusion(fusion_dim=features_dim)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: (batch, obs_dim) 
                     需要包含短期气象数据和长期历史数据
        
        注意: 这里假设 observation 已经被正确处理为两部分：
              - 前 N 维：短期气象数据
              - 后 M 维：长期历史数据
        """
        # 这里需要根据实际的 observation 结构进行解析
        # 简化示例（实际使用时需要根据环境的 observation_space 调整）
        
        batch_size = observations.size(0)
        
        # 示例：假设前 6*12=72 维是短期气象，后 4*288=1152 维是长期历史
        short_term_len = Config.CNN_INPUT_CHANNELS * Config.SHORT_TERM_WINDOW
        long_term_len = Config.LSTM_INPUT_DIM * Config.LONG_TERM_WINDOW
        
        # 提取并重塑
        short_term = observations[:, :short_term_len].view(
            batch_size, Config.CNN_INPUT_CHANNELS, Config.SHORT_TERM_WINDOW
        )
        long_term = observations[:, short_term_len:short_term_len+long_term_len].view(
            batch_size, Config.LONG_TERM_WINDOW, Config.LSTM_INPUT_DIM
        )
        
        # 特征融合
        features, _ = self.feature_fusion(short_term, long_term)
        
        return features


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 80)
    print("测试多尺度特征融合网络")
    print("=" * 80)
    
    # 设置设备
    device = Config.DEVICE
    print(f"使用设备: {device}")
    
    # 创建模型
    model = MultiScaleFeatureFusion().to(device)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试输入
    batch_size = 8
    short_term_input = torch.randn(
        batch_size, 
        Config.CNN_INPUT_CHANNELS, 
        Config.SHORT_TERM_WINDOW
    ).to(device)
    long_term_input = torch.randn(
        batch_size,
        Config.LONG_TERM_WINDOW,
        Config.LSTM_INPUT_DIM
    ).to(device)
    
    print(f"\n输入形状:")
    print(f"  短期气象数据: {short_term_input.shape}")
    print(f"  长期历史数据: {long_term_input.shape}")
    
    # 前向传播
    with torch.no_grad():
        output, attn_weights = model(short_term_input, long_term_input)
    
    print(f"\n输出形状:")
    print(f"  融合特征: {output.shape}")
    print(f"  注意力权重: {attn_weights.shape}")
    
    # 打印注意力权重（第一个样本）
    print(f"\n第一个样本的注意力权重 (头0):")
    print(attn_weights[0, 0].cpu().numpy())
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
