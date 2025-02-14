import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import matplotlib.pyplot as plt

# GridWorld 环境
class GridWorldEnv:
    def __init__(self, grid_size=5, obstacle_positions=None):
        self.grid_size = grid_size
        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        self.obstacle_positions = obstacle_positions if obstacle_positions is not None else [(2, 2)]  # 默认障碍物位置
        self.max_steps = 50
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        # 状态仍为归一化后的坐标（不包含障碍物信息）
        return np.array([self.agent_pos[0] / (self.grid_size - 1),
                         self.agent_pos[1] / (self.grid_size - 1)], dtype=np.float32)

    def step(self, action):
        x, y = self.agent_pos
        # 计算移动后的新位置
        if action == 0:  # 上
            x -= 1
        elif action == 1:  # 下
            x += 1
        elif action == 2:  # 左
            y -= 1
        elif action == 3:  # 右
            y += 1
        
        # 边界约束
        x = np.clip(x, 0, self.grid_size - 1)
        y = np.clip(y, 0, self.grid_size - 1)
        new_pos = (x, y)
        
        # 障碍物检测
        if new_pos in self.obstacle_positions:
            # 碰到障碍物，保持原位，给予较大惩罚
            reward = -0.5
            done = False
        else:
            # 正常移动
            self.agent_pos = new_pos
            # 判断是否到达终点
            reward = 1.0 if self.agent_pos == self.goal else -0.1
            done = (self.agent_pos == self.goal) or (self.steps >= self.max_steps)
        
        self.steps += 1
        return self._get_state(), reward, done, {}

    def render(self):
        """显示环境，障碍物用X表示"""
        grid = np.full((self.grid_size, self.grid_size), '.')
        # 标记障碍物
        for obs in self.obstacle_positions:
            grid[obs] = 'X'
        grid[self.goal] = 'G'
        grid[self.agent_pos] = 'A'
        for row in grid:
            print(' '.join(row))
        print("")

    def render_route(self, route):
        """显示路径，障碍物用X表示"""
        grid = np.full((self.grid_size, self.grid_size), '.')
        # 标记障碍物
        for obs in self.obstacle_positions:
            grid[obs] = 'X'
        # 标记路径
        for pos in route:
            if grid[pos] == '.':
                grid[pos] = '*'
        grid[self.start] = 'S'
        grid[self.goal] = 'G'
        if route[-1] != self.goal:
            grid[route[-1]] = 'A'
        for row in grid:
            print(' '.join(row))
        print("")

# 演员-评论家网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Linear(64, 1)

    def forward(self, state):
        x = self.fc(state)
        return self.policy_head(x), self.value_head(x)

# PPO算法实现
class PPOAgent:
    def __init__(self, state_dim, action_dim, clip_epsilon=0.2, gamma=0.99, lr=1e-3):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, state_value = self.policy(state_tensor)
        dist = distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), state_value

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R if not done else r
            returns.insert(0, R)
        return returns

    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0
        values = values + [0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, trajectories, ppo_epochs=4, batch_size=32):
        states = torch.FloatTensor([t[0] for t in trajectories])
        actions = torch.LongTensor([t[1] for t in trajectories]).unsqueeze(1)
        old_log_probs = torch.cat([t[2] for t in trajectories]).detach()
        advantages = torch.FloatTensor([t[3] for t in trajectories]).unsqueeze(1)
        returns = torch.FloatTensor([t[4] for t in trajectories]).unsqueeze(1)
        values = torch.cat([t[5] for t in trajectories]).detach()
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(ppo_epochs):
            for b_states, b_actions, b_old_log_probs, b_returns, b_advantages in loader:
                action_probs, state_values = self.policy(b_states)
                dist = distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(b_actions.squeeze())
                
                ratio = (new_log_probs - b_old_log_probs).exp()
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(state_values, b_returns)
                loss = policy_loss + 0.5 * value_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

# 训练函数（修改障碍物配置）
def train_ppo_on_gridworld():
    # 创建包含障碍物的环境（示例障碍物布局）
    obstacle_positions = [
        (1, 1), (1, 2), (1, 3),
        (2, 1),          (2, 3),
    ]  # 中心区域障碍物
    env = GridWorldEnv(grid_size=5, obstacle_positions=obstacle_positions)
    
    state_dim = 2
    action_dim = 4
    agent = PPOAgent(state_dim, action_dim, clip_epsilon=0.2, gamma=0.99, lr=1e-3)

    num_updates = 1000
    all_rewards = []
    last_successful_route = None

    for update in range(num_updates):
        trajectories = []
        ep_rewards = []
        dones = []
        state = env.reset()
        done = False
        route = [env.agent_pos]

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

        # 计算GAE和回报
        advantages = agent.compute_gae(
            [t[3] for t in trajectories],
            [t[4].item() for t in trajectories],  # 修正values的获取方式
            dones
        )
        returns = agent.compute_returns([t[3] for t in trajectories], dones)
        trajectories = [
            (t[0], t[1], t[2], adv, ret, t[4])
            for t, adv, ret in zip(trajectories, advantages, returns)
        ]
        agent.update(trajectories)

        total_reward = sum(ep_rewards)
        all_rewards.append(total_reward)
        
        # 记录成功路径
        if env.agent_pos == env.goal:
            last_successful_route = route

        if update % 50 == 0:
            print(f"Update {update}, Total Reward: {total_reward:.2f}")
            if last_successful_route:
                print("Last Successful Path:")
                env.render_route(last_successful_route)
            else:
                print("No successful path yet.\n")

    # 绘制训练曲线
    plt.plot(all_rewards)
    plt.xlabel("Training Epochs")
    plt.ylabel("Episode Reward")
    plt.title("PPO Training Performance with Obstacles")
    plt.savefig("ppo_obstacles.png")
    plt.show()

if __name__ == "__main__":
    train_ppo_on_gridworld()