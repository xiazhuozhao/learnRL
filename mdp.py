import numpy as np

# 定义网格世界的参数
GRID_SIZE = 4  # 网格的大小
N_STATES = GRID_SIZE * GRID_SIZE  # 状态总数
N_ACTIONS = 4  # 动作数：0:上 1:下 2:左 3:右
GAMMA = 0.9  # 折扣因子

# 定义障碍物和终止状态
OBSTACLES = {(1, 1)}  # 障碍物的位置
TERMINAL = (GRID_SIZE-1, GRID_SIZE-1)  # 终止状态的位置

# 初始化状态转移矩阵 P[s][a] = [(prob, next_state, reward, done)]
P = {s: {a: [] for a in range(N_ACTIONS)} for s in range(N_STATES)}

# 当前状态和总奖励
current_state = 0
total_reward = 0

# 将坐标转换为状态
def coord_to_state(row, col):
    return row * GRID_SIZE + col

# 将状态转换为坐标
def state_to_coord(s):
    return (s // GRID_SIZE, s % GRID_SIZE)

# 检查移动是否有效
def is_valid(row, col):
    if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
        return (row, col) not in OBSTACLES
    return False

# 计算曼哈顿距离（用于奖励函数）
def manhattan_distance(s1, s2):
    row1, col1 = state_to_coord(s1)
    row2, col2 = state_to_coord(s2)
    return abs(row1 - row2) + abs(col1 - col2)

# 获取奖励（改进后的奖励函数）
def get_reward(next_s):
    nr, nc = state_to_coord(next_s)
    if (nr, nc) == TERMINAL:
        return 10  # 到达终止状态奖励
    # 增加距离惩罚：距离终点越远，惩罚越大
    distance = manhattan_distance(next_s, coord_to_state(TERMINAL[0], TERMINAL[1]))
    return -0.1 * distance  # 惩罚与距离成正比

# 构建状态转移矩阵
def build_transition_matrix():
    for s in range(N_STATES):
        row, col = state_to_coord(s)
        if (row, col) == TERMINAL:
            continue  # 终止状态没有转移
        
        for a in range(N_ACTIONS):
            new_row, new_col = row, col
            if a == 0:    # 上
                new_row = max(row-1, 0)
            elif a == 1:  # 下
                new_row = min(row+1, GRID_SIZE-1)
            elif a == 2:  # 左
                new_col = max(col-1, 0)
            elif a == 3:  # 右
                new_col = min(col+1, GRID_SIZE-1)
            
            # 检查是否有效移动
            if not is_valid(new_row, new_col):
                prob = 0.0
                next_s = s  # 保持原位
            else:
                prob = 1.0
                next_s = coord_to_state(new_row, new_col)
            
            reward = get_reward(next_s)
            done = (new_row, new_col) == TERMINAL
            
            # 确定性的状态转移
            P[s][a].append((prob, next_s, reward, done))

# 重置环境
def reset():
    global current_state, total_reward, step_count
    current_state = 0  # 初始状态在(0,0)
    total_reward = 0  # 初始化总奖励
    step_count = 0  # 初始化步骤计数器
    return current_state

# 执行动作
def step(action):
    global current_state, total_reward, step_count
    transitions = P[current_state][action]
    prob, next_s, reward, done = transitions[0]
    current_state = next_s
    discounted_reward = reward * (GAMMA ** step_count)  # 计算折扣后的奖励
    distance_penalty = -0.1 * manhattan_distance(next_s, coord_to_state(TERMINAL[0], TERMINAL[1]))  # 计算距离惩罚
    print(f"基础奖励: {reward:.4f}, 距离惩罚: {distance_penalty:.4f}, 折扣因子：{GAMMA ** step_count:.4f}, 折扣奖励: {discounted_reward:.4f}")
    total_reward += discounted_reward  # 累加折扣后的奖励
    step_count += 1  # 增加步骤计数器
    print(f"当前状态: {current_state}, 动作: {action}, 状态转移概率: {transitions}")
    return next_s, discounted_reward, done, {}

# 渲染当前网格状态
def render():
    grid = []
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            s = coord_to_state(r, c)
            if (r, c) == TERMINAL:
                row.append('T')
            elif (r, c) in OBSTACLES:
                row.append('X')
            elif s == current_state:
                row.append('A')
            else:
                row.append('.')
        grid.append(' '.join(row))
    print('\n'.join(grid))
    print()

# 值迭代算法
def value_iteration(theta=1e-6):
    V = np.zeros(N_STATES)  # 初始化价值函数
    iteration = 0
    while True:
        delta = 0
        for s in range(N_STATES):
            row, col = state_to_coord(s)
            if (row, col) == TERMINAL:
                continue  # 终止状态价值为0
            v_old = V[s]
            max_value = -np.inf
            for a in range(N_ACTIONS):
                total = 0
                for (prob, next_s, reward, _) in P[s][a]:
                    total += prob * (reward + GAMMA * V[next_s])
                if total > max_value:
                    max_value = total
            V[s] = max_value
            delta = max(delta, abs(v_old - V[s]))
        iteration += 1
        print(f"第 {iteration} 次迭代后的价值函数：")
        print(V.reshape(GRID_SIZE, GRID_SIZE))
        if delta < theta:
            break
    return V

# 提取最优策略
def extract_policy(V):
    policy = np.zeros(N_STATES, dtype=int)  # 初始化策略
    for s in range(N_STATES):
        row, col = state_to_coord(s)
        if (row, col) == TERMINAL:
            continue  # 终止状态没有策略
        max_value = -np.inf
        best_action = 0
        for a in range(N_ACTIONS):
            total = 0
            for (prob, next_s, reward, _) in P[s][a]:
                total += prob * (reward + GAMMA * V[next_s])
            if total > max_value:
                max_value = total
                best_action = a
        policy[s] = best_action
    return policy

# 打印最优价值函数
def print_optimal_value(V):
    print("最优价值函数:")
    print(V.reshape(GRID_SIZE, GRID_SIZE))

# 打印最优策略
def print_optimal_policy(policy):
    print("最优策略:")
    action_symbols = ['↑', '↓', '←', '→']
    policy_grid = []
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            s = coord_to_state(r, c)
            if (r, c) == TERMINAL:
                row.append('T')
            elif (r, c) in OBSTACLES:
                row.append('X')
            else:
                row.append(action_symbols[policy[s]])
        policy_grid.append(' '.join(row))
    print('\n'.join(policy_grid))

if __name__ == "__main__":
    build_transition_matrix()  # 构建状态转移矩阵
    reset()  # 重置环境
    print("初始网格：")
    render()
    
    # 执行值迭代算法
    optimal_V = value_iteration()
    print_optimal_value(optimal_V)
    
    # 提取最优策略
    optimal_policy = extract_policy(optimal_V)
    print_optimal_policy(optimal_policy)
    
    # 使用最优策略规划路径
    print("\n使用最优策略规划路径：")
    reset()
    while True:
        action = optimal_policy[current_state]
        next_state, reward, done, _ = step(action)
        render()
        print(f"执行动作: {['↑', '↓', '←', '→'][action]}, 奖励: {reward:.4f}, 终止状态: {done}")
        if done:
            print("到达终止状态。")
            break
    
    print(f"总奖励: {total_reward}")