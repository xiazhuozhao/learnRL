### 价值函数（Value Function）详解

价值函数（Value Function）是强化学习中的一个核心概念，用于评估在某个状态下，智能体未来可能获得的累积奖励。价值函数通常分为两种：**状态价值函数**（State Value Function）和**动作价值函数**（Action Value Function）。

#### 1. 状态价值函数（V(s)）

状态价值函数表示在状态$s$下，智能体遵循某个策略时，未来可能获得的累积奖励的期望值。公式如下：

$ V^{\pi}(s) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0 = s, \pi \right] $

其中：
- $ V^{\pi}(s) $：在状态$s$下，遵循策略$π$的状态价值函数。
- $ \mathbb{E} $：期望值。
- $ \gamma $：折扣因子，取值范围为[0, 1]，用于平衡当前奖励和未来奖励的重要性。
- $ R_{t+1} $：在时间步$t+1$获得的即时奖励。
- $ S_0 = s $：初始状态为$s$。
- $ \pi $：策略，表示在状态$s$下选择动作的规则。

#### 2. 动作价值函数（Q(s, a)）

动作价值函数表示在状态$s$下，智能体执行动作$a$后，未来可能获得的累积奖励的期望值。公式如下：

$ Q^{\pi}(s, a) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0 = s, A_0 = a, \pi \right] $

其中：
- $ Q^{\pi}(s, a) $：在状态$s$下，执行动作$a$后，遵循策略$π$的动作价值函数。
- $ A_0 = a $：初始动作为$a$。

#### 3. 贝尔曼方程（Bellman Equation）

价值函数可以通过贝尔曼方程进行递归定义。对于状态价值函数，贝尔曼方程如下：

$ V^{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V^{\pi}(s') \right] $

其中：
- $ \pi(a|s) $：在状态$s$下选择动作$a$的概率。
- $ P(s'|s, a) $：在状态$s$下执行动作$a$后，转移到状态$s'$的概率。
- $ R(s, a, s') $：在状态$s$下执行动作$a$后，转移到状态$s'$获得的即时奖励。

对于动作价值函数，贝尔曼方程如下：

$ Q^{\pi}(s, a) = \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma \sum_{a'} \pi(a'|s') Q^{\pi}(s', a') \right] $

#### 4. 最优价值函数

最优价值函数表示在状态$s$下，智能体遵循最优策略时，未来可能获得的累积奖励的最大值。最优状态价值函数和最优动作价值函数分别表示为：

$ V^*(s) = \max_{\pi} V^{\pi}(s) $
$ Q^*(s, a) = \max_{\pi} Q^{\pi}(s, a) $
$
V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]
$  

其中：
- $ V^*(s) $：状态 $ s $ 的最优价值函数。
- $ \max_a $：对所有可能的动作 $ a $ 取最大值。
- $ P(s' \mid s, a) $：在状态 $ s $ 下执行动作 $ a $ 后转移到状态 $ s' $ 的概率。
- $ R(s, a, s') $：在状态 $ s $ 下执行动作 $ a $ 后转移到状态 $ s' $ 获得的即时奖励。
- $ \gamma $：折扣因子，用于平衡当前奖励和未来奖励的重要性。

##### 推导过程

###### （1）从贝尔曼方程出发

首先，回顾贝尔曼方程（Bellman Equation）的定义。对于任意策略 $ \pi $，状态价值函数 $ V^\pi(s) $ 满足：

$
V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]
$

其中：
- $ \pi(a \mid s) $：在状态 $ s $ 下选择动作 $ a $ 的概率。

###### （2）最优策略的定义

最优策略 $ \pi^* $ 是指在所有策略中，能够使状态价值函数最大化的策略。因此，最优状态价值函数 $ V^*(s) $ 定义为：

$
V^*(s) = \max_\pi V^\pi(s)
$

###### （3）将贝尔曼方程推广到最优策略

对于最优策略 $ \pi^* $，状态价值函数 $ V^*(s) $ 满足：

$
V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]
$

这是因为最优策略会选择使未来累积奖励最大化的动作 $ a $，因此需要对所有可能的动作 $ a $ 取最大值。

###### （4）直观理解

- **$ \max_a $**：最优策略会选择使未来累积奖励最大化的动作。
- **$ \sum_{s'} P(s' \mid s, a) $**：考虑所有可能的下一个状态 $ s' $。
- **$ R(s, a, s') + \gamma V^*(s') $**：当前奖励加上未来奖励的折扣值。




#### 5. 值迭代算法

值迭代算法是一种动态规划方法，用于求解最优价值函数。其基本思想是通过迭代更新价值函数，直到收敛到最优值。值迭代算法的步骤如下：

1. 初始化价值函数 $ V(s) $ 为任意值（通常为0）。
2. 对于每个状态$s$，计算新的价值函数：
   $ V(s) = \max_{a} \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V(s') \right] $
3. 重复步骤2，直到价值函数的变化小于某个阈值$θ$。

#### 6. 代码中的价值函数实现

在代码中，价值函数通过 `value_iteration` 函数实现。该函数通过迭代更新每个状态的价值函数，直到收敛到最优值。最终，最优价值函数用于提取最优策略。

```python
def value_iteration(theta=1e-6):
    V = np.zeros(N_STATES)  # 初始化价值函数
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
        if delta < theta:
            break
    return V
```