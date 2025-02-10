import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 创建CartPole环境
env = gym.make('CartPole-v1')

# 策略网络，1层MLP
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # 定义第一层全连接层，输入维度为状态维度，输出维度为128
        self.fc1 = nn.Linear(state_dim, 128)
        # 定义第二层全连接层，输入维度为128，输出维度为动作维度
        self.fc2 = nn.Linear(128, action_dim)
        # 定义softmax层，用于输出动作概率
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        # 前向传播，使用ReLU激活函数
        x = torch.relu(self.fc1(state))
        # 计算动作概率
        action_probs = self.softmax(self.fc2(x))
        return action_probs

# 采样轨迹
def sample_trajectory(env, policy, max_timesteps=500):
    # 重置环境，获取初始状态
    state = env.reset()[0]
    trajectory = []
    
    for t in range(max_timesteps):
        # 将状态转换为tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # 通过策略网络计算动作概率
        action_probs = policy(state_tensor)
        # 根据动作概率选择动作
        action = np.random.choice(len(action_probs.detach().numpy()), p=action_probs.detach().numpy())

        # 执行动作，获取下一个状态、奖励和是否结束标志
        next_state, reward, done, _, _ = env.step(action)
        # 将当前状态、动作和奖励存储到轨迹中
        trajectory.append((state, action, reward))

        # 更新状态
        state = next_state
        if done:
            break

    return trajectory

# 计算回报
def compute_returns(trajectory, gamma=0.99):
    returns = []
    G = 0
    # 反向遍历轨迹，计算每个时间步的回报
    for _, _, reward in reversed(trajectory):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns

# 训练REINFORCE算法
def train_reinforce(env, policy, optimizer, num_episodes=1000, gamma=0.99):
    for episode in range(num_episodes):
        # 采样一个轨迹
        trajectory = sample_trajectory(env, policy)
        # 计算轨迹的回报
        returns = compute_returns(trajectory, gamma)

        policy_loss = []
        # 遍历轨迹中的每个状态、动作和回报
        for (state, action, _), G_t in zip(trajectory, returns):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            # 计算动作概率
            action_probs = policy(state_tensor)
            # 计算动作的对数概率
            log_prob = torch.log(action_probs[action])
            # 计算策略损失
            policy_loss.append(-log_prob * G_t)

        # 清零梯度
        optimizer.zero_grad()
        # 计算总损失
        loss = torch.stack(policy_loss).sum()
        # 反向传播
        loss.backward()
        # 更新策略网络参数
        optimizer.step()

        # 每50个episode打印一次总奖励
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}, Total Reward: {sum([r for _, _, r in trajectory])}")

# 获取状态和动作的维度
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
# 初始化策略网络
policy = PolicyNetwork(state_dim, action_dim)
# 初始化优化器
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# 训练策略网络
train_reinforce(env, policy, optimizer)