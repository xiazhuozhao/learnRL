import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 定义 Actor-Critic 网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # 输入层到隐藏层的全连接层
        
        # Actor 头部
        self.actor = nn.Linear(128, action_dim)  # 隐藏层到动作输出层的全连接层
        
        # Critic 头部
        self.critic = nn.Linear(128, 1)  # 隐藏层到状态值输出层的全连接层
        
        self.softmax = nn.Softmax(dim=-1)  # 对动作概率进行 softmax 处理

    def forward(self, state):
        x = torch.relu(self.fc1(state))  # 通过隐藏层并使用 ReLU 激活函数
        action_probs = self.softmax(self.actor(x))  # 计算动作概率 π(a|s)
        value = self.critic(x)  # 计算状态值 V(s)
        return action_probs, value

# 超参数
gamma = 0.99  # 折扣因子
lr = 0.001  # 学习率
num_episodes = 1000  # 训练的总回合数
env_name = "CartPole-v1"  # 环境名称

# 创建环境并获取状态和动作的维度
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 初始化网络和优化器
policy = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=lr)

# 训练循环
for episode in range(num_episodes):
    state = env.reset()[0]  # 重置环境并获取初始状态
    done = False
    total_reward = 0  # 记录总奖励

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)  # 将状态转换为张量
        action_probs, value = policy(state_tensor)  # 前向传播获取动作概率和状态值（分别为actor和critic的输出）
        action = np.random.choice(len(action_probs.detach().numpy()), p=action_probs.detach().numpy())  # 根据概率选择动作
        
        next_state, reward, done, _, _ = env.step(action)  # 执行动作并获取下一个状态和奖励

        # 使用 1 步 TD 误差计算优势
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        _, next_value = policy(next_state_tensor) if not done else (None, torch.tensor(0.0))  # 如果未结束，计算下一个状态值，否则为 0
        
        td_target = reward + gamma * next_value.item()  # 计算 TD 目标
        advantage = td_target - value.item()  # 计算优势

        # 计算损失
        log_prob = torch.log(action_probs[action])  # 动作的对数概率
        policy_loss = -log_prob * advantage  # 策略梯度损失
        value_loss = nn.functional.mse_loss(value, torch.tensor(td_target))  # Critic 损失

        # 执行梯度更新
        optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        optimizer.step()

        state = next_state  # 更新状态
        total_reward += reward  # 累加奖励

    # 监控训练进度
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}, Total Reward: {total_reward}")

env.close()  # 关闭环境