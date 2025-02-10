import numpy as np
import matplotlib.pyplot as plt

# 定义 GridWorld 环境
class GridWorld:
    def __init__(self, size=5, terminal_states={(4, 4)}, rewards={(4, 4): 10}):
        self.size = size  # 网格大小，状态空间 S 的维度
        self.terminal_states = terminal_states  # 终止状态集合 S_terminal
        self.rewards = rewards  # 奖励函数 R(s)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 动作空间 A = {上, 下, 左, 右}
        self.state_space = [(i, j) for i in range(size) for j in range(size)]  # 状态空间 S = {(i, j) | 0 ≤ i, j < size}

    def get_next_state(self, state, action):
        """返回应用动作后的下一个状态 s'。
        参数:
        state -- 当前状态 s = (i, j)
        action -- 动作 a ∈ A
        返回值:
        next_state -- 下一个状态 s'
        """
        if state in self.terminal_states:
            return state  # 终止状态 s ∈ S_terminal 保持不变

        next_state = (state[0] + action[0], state[1] + action[1])  # 计算 s' = s + a

        # 边界条件：如果 s' 越界，保持原地不动
        if next_state not in self.state_space:
            return state

        return next_state

    def get_reward(self, state):
        """返回状态 s 的奖励 R(s)。
        参数:
        state -- 当前状态 s = (i, j)
        返回值:
        reward -- 奖励值 R(s)
        """
        return self.rewards.get(state, -0.01)  # 非目标状态的小惩罚 R(s) = -0.01


# 定义 n 步 TD 策略评估
def n_step_td_policy_evaluation(env, policy, alpha=0.1, gamma=0.9, n=3, episodes=1000):
    """n 步时序差分策略评估，用于估计 V_π。
    参数:
    env -- 环境实例
    policy -- 策略函数 π(a|s)
    alpha -- 学习率 α
    gamma -- 折扣因子 γ
    n -- n 步 TD 的步数
    episodes -- 迭代次数
    返回值:
    V -- 价值函数 V_π(s)
    deltas -- 每次更新的变化量 Δ
    """
    V = np.zeros((env.size, env.size))  # 初始化价值函数 V(s) = 0
    deltas = []  # 用于记录每次更新的变化量 Δ

    for episode in range(episodes):
        state = (np.random.randint(env.size), np.random.randint(env.size))  # 从随机状态 s_0 开始
        if state in env.terminal_states:
            continue  # 如果初始状态是终止状态 s ∈ S_terminal，则跳过该回合

        trajectory = []  # 记录状态和奖励的轨迹 τ = [(s_0, r_1), (s_1, r_2), ...]
        T = 1000  # 终止时间步，初始设置为一个大值
        t = 0  # 当前时间步 t

        while True:
            if t < T:
                action = policy(state)  # 根据策略 π(a|s) 选择动作 a
                next_state = env.get_next_state(state, action)  # 获取下一个状态 s'
                reward = env.get_reward(next_state)  # 获取奖励 r = R(s')

                trajectory.append((state, reward))  # 将当前状态和奖励添加到轨迹 τ 中
                state = next_state  # 更新当前状态 s = s'

                if next_state in env.terminal_states:
                    T = t  # 如果到达终止状态 s ∈ S_terminal，设置终止时间步 T

            tau = t - n  # 第一个可更新的时间步 τ = t - n

            if tau >= 0:
                # 计算从 τ 到 τ+n 的回报 G_{τ:τ+n}
                G = sum([gamma**(i - tau) * trajectory[i][1] for i in range(tau, min(tau + n, T)+1)])
                
                if tau + n < T:  # 如果 τ+n < T，用估计的 V(s_{τ+n}) 进行引导
                    G += gamma**n * V[trajectory[tau + n][0]]

                state_tau = trajectory[tau][0]  # 获取 τ 时刻的状态 s_τ
                delta = abs(G - V[state_tau])  # 计算价值函数的变化量 Δ = |G - V(s_τ)|
                V[state_tau] += alpha * (G - V[state_tau])  # 更新价值函数 V(s_τ) ← V(s_τ) + α(G - V(s_τ))
                deltas.append(delta)  # 记录变化量 Δ

            if tau >= T-1:
                break  # 如果 τ 达到终止时间步 T-1，结束循环
            t += 1  # 增加时间步 t = t + 1
    return V, deltas  # 返回价值函数 V(s) 和变化量记录 Δ


# 定义随机策略
def random_policy(state):
    """随机策略 π(a|s)，从动作空间 A 中随机选择一个动作 a。
    参数:
    state -- 当前状态 s = (i, j)
    返回值:
    action -- 随机选择的动作 a ∈ A
    """
    return env.actions[np.random.choice(len(env.actions))]  # 随机选择一个动作 a


# 运行 n 步 TD 评估
env = GridWorld()  # 创建 GridWorld 环境实例
gamma_values = [0.99]  # 定义不同的折扣因子 γ
n_values = [1, 3, 5]  # 定义不同的 n 步数

# 存储绘图结果
results = {}

def plot_value_function(V, title="Value Function"):
    """绘制价值函数图 V(s)。
    参数:
    V -- 价值函数 V(s)
    title -- 图像标题
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(V, cmap='coolwarm', origin='upper')  # 显示价值函数矩阵 V(s)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            plt.text(j, i, f"{V[i, j]:.2f}", ha='center', va='center', color='black')  # 在每个格子中显示价值 V(s)
    plt.colorbar()  # 添加颜色条
    plt.title(title)  # 设置标题
    plt.savefig(f"{title}.png")  # 保存图像
    plt.show()  # 显示图像

for gamma in gamma_values:
    for n in n_values:
        V, deltas = n_step_td_policy_evaluation(env, random_policy, gamma=gamma, n=n)  # 运行 n 步 TD 策略评估
        results[(gamma, n)] = (V, deltas)  # 存储结果
        plot_value_function(V, title=f"Value Function (γ={gamma}, n={n})")  # 绘制价值函数图 V(s)