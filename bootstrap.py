import numpy as np
import matplotlib.pyplot as plt

# 定义 GridWorld 环境
class GridWorld:
    def __init__(self, size=5, terminal_states={(4, 4)}, rewards={(4, 4): 10}):
        self.size = size  # 网格大小
        self.terminal_states = terminal_states  # 终止状态
        self.rewards = rewards  # 奖励
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右动作
        self.state_space = [(i, j) for i in range(size) for j in range(size)]  # 状态空间

    def get_next_state(self, state, action):
        """返回应用动作后的下一个状态。"""
        if state in self.terminal_states:
            return state  # 终止状态保持不变

        next_state = (state[0] + action[0], state[1] + action[1])

        # 边界条件
        if next_state not in self.state_space:
            return state  # 如果越界，保持原地不动

        return next_state

    def get_reward(self, state):
        return self.rewards.get(state, -0.01)  # 非目标状态的小惩罚

# 定义 n 步 TD 策略评估
def n_step_td_policy_evaluation(env, policy, alpha=0.1, gamma=0.9, n=3, episodes=1000):
    """n 步时序差分策略评估，用于估计 V_π。"""
    V = np.zeros((env.size, env.size))  # 初始化价值函数
    deltas = []

    for episode in range(episodes):
        state = (np.random.randint(env.size), np.random.randint(env.size))  # 从随机状态开始
        if state in env.terminal_states:
            continue

        trajectory = []
        T = 1000
        t = 0

        while True:
            if t < T:
                action = policy(state)
                next_state = env.get_next_state(state, action)
                reward = env.get_reward(next_state)

                trajectory.append((state, reward))
                state = next_state

                if next_state in env.terminal_states:
                    T = t  # 设置终止时间

            tau = t - n  # 第一个可更新的时间步

            if tau >= 0:
                # 计算从 tau 到 tau+n 的回报 G
                G = sum([gamma**(i - tau) * trajectory[i][1] for i in range(tau, min(tau + n, T)+1)])
                
                if tau + n < T:  # 用估计的 V(S_tau+n) 进行引导
                    G += gamma**n * V[trajectory[tau + n][0]]

                state_tau = trajectory[tau][0]
                delta = abs(G - V[state_tau])
                V[state_tau] += alpha * (G - V[state_tau])
                deltas.append(delta)

            if tau >= T-1:
                break
            t += 1  # 下一个时间步
    return V, deltas

# 定义随机策略
def random_policy(state):
    return env.actions[np.random.choice(len(env.actions))]

# 运行 n 步 TD 评估
env = GridWorld()
gamma_values = [0.99]
n_values = [1, 3, 5]

# 存储绘图结果
results = {}

def plot_value_function(V, title="Value Function"):
    plt.figure(figsize=(6, 6))
    plt.imshow(V, cmap='coolwarm', origin='upper')
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            plt.text(j, i, f"{V[i, j]:.2f}", ha='center', va='center', color='black')
    plt.colorbar()
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.show()

for gamma in gamma_values:
    for n in n_values:
        V, deltas = n_step_td_policy_evaluation(env, random_policy, gamma=gamma, n=n)
        results[(gamma, n)] = (V,deltas)
        plot_value_function(V, title=f"Value Function (γ={gamma}, n={n})")

# # 绘制收敛图
# plt.figure(figsize=(10, 6))
# for (gamma, n), deltas in results.items():
#     plt.plot(deltas, label=f"γ={gamma}, n={n}")

# plt.xlabel("迭代次数")
# plt.ylabel("Δ (价值函数变化)")
# plt.title("n 步 TD 策略评估的收敛性")
# plt.yscale("log")  # 使用对数刻度更好地可视化
# plt.legend()
# plt.grid()
# plt.savefig("n_step_td_convergence.png")
# plt.show()