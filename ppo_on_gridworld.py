import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import matplotlib.pyplot as plt

# 定义 GridWorld 环境
class GridWorldEnv:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        self.max_steps = 50  # 每一局最多步数
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        # 为简化起见，将状态表示为归一化后的二维坐标 [x/(N-1), y/(N-1)]
        return np.array([self.agent_pos[0] / (self.grid_size - 1),
                         self.agent_pos[1] / (self.grid_size - 1)], dtype=np.float32)

    def step(self, action):
        # action: 0:上, 1:下, 2:左, 3:右
        x, y = self.agent_pos
        if action == 0:  # 上
            x -= 1
        elif action == 1:  # 下
            x += 1
        elif action == 2:  # 左
            y -= 1
        elif action == 3:  # 右
            y += 1
        # 保证边界内移动
        x = np.clip(x, 0, self.grid_size - 1)
        y = np.clip(y, 0, self.grid_size - 1)
        self.agent_pos = (x, y)
        self.steps += 1
        done = (self.agent_pos == self.goal) or (self.steps >= self.max_steps)
        reward = 1.0 if self.agent_pos == self.goal else -0.1
        return self._get_state(), reward, done, {}

    def render(self):
        """
        直接在命令行中打印当前环境状态，
        用 'A' 表示当前位置，'G' 表示目标点，其他位置用 '.' 表示
        """
        grid = np.full((self.grid_size, self.grid_size), '.')
        grid[self.goal] = 'G'
        grid[self.agent_pos] = 'A'
        for row in grid:
            print(' '.join(row))
        print("")

    def render_route(self, route):
        """
        在命令行中可视化代理的路径，使用 '*' 标记经过的路线
        参数:
            route: 记录了每一步 agent_pos 的列表，每个元素为 (x, y) 坐标
        """
        grid = np.full((self.grid_size, self.grid_size), '.')
        # 用 '*' 标记经过的所有位置
        for pos in route:
            grid[pos] = '*'
        # 特别标记起点和终点
        grid[self.start] = 'S'  # 起点用 S 表示
        grid[self.goal] = 'G'   # 终点用 G 表示
        # 如果最后一步未达到终点，则用 A 表示终止时的位置
        if route[-1] != self.goal:
            grid[route[-1]] = 'A'
        for row in grid:
            print(' '.join(row))
        print("")

# 定义演员-评论家网络（Actor-Critic Network）
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # 策略网络（Actor）和价值网络（Critic）共享的全连接层（MLP）
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),  # 引入非线性
        )
        # 策略头，用于输出动作概率分布
        self.policy_head = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)  # 将输出转换为概率分布
        )
        # 价值头，用于输出状态值
        self.value_head = nn.Linear(64, 1)  # 输出标量值

    def forward(self, state):
        x = self.fc(state)  # 共享的全连接层
        action_probs = self.policy_head(x)  # 策略头
        state_value = self.value_head(x)  # 价值头
        return action_probs, state_value

# PPO 算法实现
class PPOAgent:
    def __init__(self, state_dim, action_dim, clip_epsilon=0.2, gamma=0.99, lr=1e-3):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # [1, state_dim]
        action_probs, state_value = self.policy(state_tensor)
        # action_probs 是策略网络（Actor）输出的动作概率分布
        dist = distributions.Categorical(action_probs)
        # dist是一个 Categorical 分布对象，可以方便地进行采样
        # distributions 用于处理离散概率分布
        action = dist.sample()
        # dist.sample() 会根据这些概率随机选择一个动作，并返回该动作的索引
        return action.item(), dist.log_prob(action), state_value

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        # 从后往前计算折扣累计回报；遇到 done 则 R 重置为 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            # 计算折扣累计回报
            # R_t = r_t + gamma * R_{t+1}
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """
        计算广义优势估计（GAE）
        参数：
            rewards: 每一步的奖励列表
            values: 每一步对应的价值估计列表（长度应与 rewards 相同）
            dones: 每一步是否终止的标志（布尔值列表）
            gamma: 折扣因子
            lam: GAE 的 lambda 参数
        返回：
            advantages: 每个时间步的优势估计列表
        """
        advantages = []
        gae = 0
        # 为方便计算，在最后增加一个 bootstrap 值 0
        values = values + [0]
        for t in reversed(range(len(rewards))):
            # 计算 TD 残差（Temporal Difference Error）
            # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            # 计算 GAE
            # GAE_t = delta_t + gamma * lambda * (1 - done) * GAE_{t+1}
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, trajectories, ppo_epochs=4, batch_size=32):
        # trajectories 为一个列表，每个元素为 (state, action, log_prob, advantages, returns, state_value)
        states = torch.FloatTensor([t[0] for t in trajectories])
        actions = torch.LongTensor([t[1] for t in trajectories]).unsqueeze(1)
        old_log_probs = torch.cat([t[2] for t in trajectories]).detach()
        advantages = torch.FloatTensor([t[3] for t in trajectories]).unsqueeze(1)
        returns = torch.FloatTensor([t[4] for t in trajectories]).unsqueeze(1)
        values = torch.cat([t[5] for t in trajectories]).detach()
        # 优势归一化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(ppo_epochs):
            for b_states, b_actions, b_old_log_probs, b_returns, b_advantages in loader:
                action_probs, state_values = self.policy(b_states)
                dist = distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(b_actions.squeeze())
                # 计算概率比率
                ratio = (new_log_probs - b_old_log_probs.squeeze()).exp()
                # PPO 剪切目标
                surr1 = ratio * b_advantages.squeeze()
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages.squeeze()
                # torch.clamp 函数用于将 ratio 限制在 [1 - self.clip_epsilon, 1 + self.clip_epsilon] 范围内
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(state_values, b_returns)
                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

# 训练 PPO 在 GridWorld 环境下
def train_ppo_on_gridworld():
    env = GridWorldEnv(grid_size=5)
    state_dim = 2       # 状态为归一化后的 (x, y)
    action_dim = 4      # 上、下、左、右
    agent = PPOAgent(state_dim, action_dim, clip_epsilon=0.2, gamma=0.99, lr=1e-3)

    num_updates = 1000
    all_rewards = []
    last_successful_route = None  # 用于保存最后一次成功到达终点的路径

    for update in range(num_updates):
        trajectories = []  # 存储每个时刻的信息
        ep_rewards = []
        dones = []
        state = env.reset()
        done = False

        # 记录从起点开始的路径（保存整数坐标）
        route = []
        route.append(env.agent_pos)

        while not done:
            action, log_prob, state_value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            trajectories.append((state, action, log_prob, reward, state_value))
            ep_rewards.append(reward)
            dones.append(done)
            state = next_state
            route.append(env.agent_pos)
            if done:
                break

        # 计算回报和优势
        advantages = agent.compute_gae([t[3] for t in trajectories],
                                       [t[4] for t in trajectories],
                                       dones, gamma=0.99, lam=0.95)
        returns = agent.compute_returns([t[3] for t in trajectories], dones)
        # 组合轨迹数据：(state, action, log_prob, advantages, returns, state_value)
        trajectories = [(traj[0], traj[1], traj[2], adv, ret, traj[4])
                        for traj, adv, ret in zip(trajectories, advantages, returns)]
        agent.update(trajectories)

        total_reward = sum(ep_rewards)
        all_rewards.append(total_reward)
        
        # 如果本局训练成功达到终点，则保存这条路径
        if env.agent_pos == env.goal:
            last_successful_route = route

        if update % 50 == 0:
            print(f"更新 {update} 次，当前回合总奖励: {total_reward:.2f}")
            # print("当前环境状态:")
            # env.render()  # 打印当前环境状态
            if last_successful_route is not None:
                print("最后一次成功到达终点的路线:")
                env.render_route(last_successful_route)
            else:
                print("尚未有成功到达终点的路线。\n")

    # 绘制训练过程中每次更新的奖励曲线
    plt.plot(all_rewards)
    plt.xlabel("Number of Updates")
    plt.ylabel("Episode Reward")
    plt.title("PPO Training Curve in GridWorld Environment")
    plt.savefig("ppo_on_gridworld.png")
    plt.show()

if __name__ == "__main__":
    train_ppo_on_gridworld()