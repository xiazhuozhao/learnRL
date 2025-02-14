# PPO算法梳理

## 基本算法

![PPO算法流程图](pic/PPO算法流程图.drawio.svg)

## 要点

### 基本框架：Advantaged Actor-Critic

策略梯度算法是一类使用策略网络来预测最优策略，并且通过策略的梯度进行优化的算法。策略梯度的表达式为
$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{s\sim\eta,a\sim\pi} \left[ \nabla_{\theta} \ln \pi(a_t , s_t|\theta) q_\pi(s,a) \right]
$$
Actor-critic方法是一种策略梯度方法，通过价值网络预测策略梯度表达式中的$q_\pi(s,a)$，并使用TD error更新价值网络。

为了减小$q_\pi(s,a)$的方差，可以使用带baseline的策略梯度方法
$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{s\sim\eta,a\sim\pi} \left[ \nabla_{\theta} \ln \pi(a , s|\theta) \left[q_\pi(s,a)-b(s)\right] \right]
$$
这里$b(s)$是任意函数，通常取做$V_\pi(s)$的某种近似。

将baseline结合于Actor-critic方法，我们可以让critic网络预测$V_\pi(s)$，然后用$V_\pi(s')$来计算$Q_\pi(a,s)$，即$Q_\pi(a,s)=r(a,s)+\gamma V_\pi(s')$。这样改写之后的Actor-critic方法称作Advantaged actor-critic方法。

### PPO的核心思想：限制策略更新大小

策略梯度方法中涉及到优势函数的期望，这个期望在实际计算中是通过采样计算的。如果在一条轨迹中优化很多次，采样的随机性就逐渐累积，实验上的结果是梯度可能会很大，每个episode都会带来过大的策略更新。TRPO算法以及PPO算法的关键创新是限制策略更新大小。

#### TRPO：固定惩罚函数

TRPO算法把优化$L^{PG}(\theta)$的问题替换为了优化
$$
L^{PG}(\theta)=\mathbb E_t\left[\frac{\pi(a_t,s_t)}{\pi_{old}(a_t,s_t)}A_t\right]
$$
使得
$$
\mathbb E_t\left[\mathrm{KL}\left({\pi(a_t,s_t)}{\pi_{old}(a_t,s_t)}\right)\right]\le\delta
$$
这里$\pi_{old}$作为固定参数（不参与求梯度），起到的作用是$\nabla\ln\pi$产生的$1/\pi$。这个限制相当于限制策略更新的大小。

实际上计算时可以把限制替换为罚函数
$$
L^{PG}(\theta)=\mathbb E_t\left[\frac{\pi(a_t,s_t)}{\pi_{old}(a_t,s_t)}A_t-\beta\mathrm{KL}\left({\pi(a_t,s_t)}{\pi_{old}(a_t,s_t)}\right)\right]
$$
TRPO算法使用固定的$\beta$，但实际往往需要根据$\mathrm{KL}\left({\pi(a_t,s_t)}{\pi_{old}(a_t,s_t)}\right)$的数值进行调节。

#### Adaptive KL Penalty

在TRPO的基础上，引入自动调整$\beta$的策略：设定一个目标KL divergence，如果更新的KL divergence较小，就减少$\beta$，反之增大$\beta$。

#### 裁剪目标函数

另一限制更新大小的方式是把$r_\theta=\frac{\pi(a_t,s_t)}{\pi_{old}(a_t,s_t)}$限制在一个给定的区间内
$$
L^{CLIP}=\mathbb E_t\left[\min\left(r_\theta A_t,\mathrm{clip}(r_\theta,1-\epsilon,1+\epsilon)A_t\right)\right]
$$
相当于$A>0$时，限制$r$不大于$1+\epsilon$；$A<0$时，限制$r$不小于$1-\epsilon$，可以防止策略更新过大。

$\epsilon$的选择：原论文中，0.2达到了最佳结果。

### GAE：平衡方差和偏差自举

定义$A(a,s)=Q_\pi(a,s)-b(s)$为优势函数，$b(s)$可以取任意形式，但是由于目的是减少$A(a,s)$的方差，一般取$b(s)=\mathbb E\left[Q(a,s)\right]=V(s)$。

由贝尔曼方程的递推，可以得到
$$
A^k(a,s)=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots+\gamma^{k-1}r_{t+k-1}+\gamma^{k}V_{t+k}-V_t
$$
定义$\delta_t=r_t+\gamma V_{t+1}-V_t$，上式也能写作
$$
A^k(a,s)=\sum_{i=0}^{i=k-1}\gamma^i\delta_{t+i}
$$
实际中，可以取不同的k。$k$越小，则$A^k$受更新前的估计值$V_{t+k}$的影响就越大，自举现象可能会导致估计持续产生偏差；$k$越大，$A^k$受$V_{t+k}$的影响就越小，而受新的观测$r_t,\cdots,r_{t+k-1}$影响就越大，由于采样的随机性，这样产生的优势函数方差较大。

为了综合各种$k$值得影响，可以定义
$$
\begin{align}
A_t^{GAE}&=(1-\lambda)\left(A^1+A^2\lambda+A^3\lambda^2+\cdots\right)\\
&=(1-\lambda)\left(\delta_{t}+\lambda(\delta_t+\gamma\delta_{t+1})+\lambda^2(\delta_t+\gamma\delta_{t+1}+\gamma^2\delta_{t+2})+\cdots\right)\\
&=(1-\lambda)\left(\sum_{i=0}^{+\infty}\lambda^i\delta_{t}+\sum_{i=1}^{+\infty}\gamma\lambda^i\delta_{t+1}+\sum_{i=2}^{+\infty}\gamma^2\lambda^i\delta_{t+2}+\cdots\right)\\
&=(1-\lambda)\sum_{k=0}^{+\infty}\left(\frac{\gamma^k\lambda^k}{1-\lambda}\delta_{t+k}\right)\\
&=\sum_{k=0}^{+\infty}\gamma^k\lambda^k\delta_{t+k}
\end{align}
$$
$\lambda=0$时，即是TD-error；$\lambda=1$时，即是$A^{\infty}$。

$\lambda$的选择（gpt意见）：
- 论文中（例如Schulman等人的论文）常用的 λ 值大约是0.95，这个值在很多任务上表现较好。在一些噪声较大或长期依赖性较弱的环境中，可以尝试降低 λ（例如0.90），以降低方差；而在需要更长远考虑的任务中，可以考虑提高 λ（例如0.98甚至0.99）。
- 折扣因子 γ 通常设为0.99或0.995（取决于环境的任务和奖励稀疏程度）。
- 有时候调高 λ 会带来更平滑的优势估计，但也可能使训练更新的“步伐”变得缓慢，进而需要调整学习率。

$\gamma$的选择：

- 对于连续控制问题或需要考虑长期回报的问题，通常设为0.99。
- 如果环境噪声过大或者更需要考虑即时回报，则可以设为0.95甚至0.9。
- 通常都在0.95-0.99之间