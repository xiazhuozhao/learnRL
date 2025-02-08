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

# 获取奖励
def get_reward(next_s):
    nr, nc = state_to_coord(next_s)
    if (nr, nc) == TERMINAL:
        return 10
    return -0.1

current_state = 0
total_reward = 0
step_count = 0

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
    print(f"基础奖励: {reward}, 折扣因子：{GAMMA ** step_count}, 折扣奖励: {discounted_reward}")
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

# 打印MDP信息
def print_mdp():
    # print(f"状态集（S）: {N_STATES}")
    # print(f"动作集（A）: {N_ACTIONS}")
    # print(f"状态转移概率（P）(prob, next_state, reward, done): {P}")
    print(f"奖励函数（R）: {total_reward}")
    print(f"折扣因子（γ）: {GAMMA}")

if __name__ == "__main__":
    build_transition_matrix()  # 构建状态转移矩阵
    reset()  # 重置环境
    print("初始网格：")
    render()
    print(f"状态转移概率（P）(prob, next_state, reward, done): {P}")
    print(f"状态集大小（S）: {N_STATES}")
    print(f"动作集大小（A）: {N_ACTIONS}")
    print(f"折扣因子（γ）: {GAMMA}")
    
    # 定义键与动作的映射
    action_map = {
        'w': 0,  # 上
        's': 1,  # 下
        'a': 2,  # 左
        'd': 3   # 右
    }
    
    while True:
        print('-'*20)
        action = input("请输入动作 (w:上 s:下 a:左 d:右, q:退出): ")
        if action == 'q':
            break
        if action not in action_map:
            print("无效输入。")
            continue
        
        action = action_map[action]
        next_state, reward, done, _ = step(action)
        print(f"执行动作 {action}:")
        render()
        print(f"奖励: {reward}, 终止状态: {done}")
        if done:
            print("到达终止状态。")
            break
    
    print(f"总奖励: {total_reward}")


# 主函数
# if __name__ == "__main__":
#     build_transition_matrix()  # 构建状态转移矩阵
#     reset()  # 重置环境
#     print("初始网格：")
#     render()
    
#     actions = [3, 1, 3, 1, 3, 1, 3, 1]  # 向右->向下循环 (0:上 1:下 2:左 3:右)
    
#     for action in actions:
#         next_state, reward, done, _ = step(action)
#         print(f"执行动作 {action}:")
#         render()
#         print(f"奖励: {reward}, 终止状态: {done}\n")
#         if done:
#             break
    
#     print(f"总奖励: {total_reward}")
#     print_mdp()