import numpy as np
import matplotlib.pyplot as plt

# Define GridWorld environment
class GridWorld:
    def __init__(self, size=5, terminal_states={(4, 4)}, rewards={(4, 4): 10}):
        self.size = size
        self.terminal_states = terminal_states
        self.rewards = rewards
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        self.state_space = [(i, j) for i in range(size) for j in range(size)]

    def get_next_state(self, state, action):
        """Returns next state after applying action."""
        if state in self.terminal_states:
            return state  # Terminal states stay the same

        next_state = (state[0] + action[0], state[1] + action[1])

        # Boundary conditions
        if next_state not in self.state_space:
            return state  # If out of bounds, stay in place

        return next_state

    def get_reward(self, state):
        return self.rewards.get(state, -0.01)  # Small penalty for non-goal states

# Define n-step TD Policy Evaluation
def n_step_td_policy_evaluation(env, policy, alpha=0.1, gamma=0.9, n=3, episodes=1000):
    """n-step Temporal Difference policy evaluation for estimating V_π."""
    V = np.zeros((env.size, env.size))  # Initialize value function
    deltas = []

    for episode in range(episodes):
        state = (np.random.randint(env.size), np.random.randint(env.size))  # Start at random state
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
                    T = t  # Set termination time

            tau = t - n  # First updateable time step

            if tau >= 0:
                # Compute return G from tau to tau+n
                G = sum([gamma**(i - tau) * trajectory[i][1] for i in range(tau, min(tau + n, T)+1)])
                
                if tau + n < T:  # Bootstrap with estimated V(S_tau+n)
                    G += gamma**n * V[trajectory[tau + n][0]]

                state_tau = trajectory[tau][0]
                delta = abs(G - V[state_tau])
                V[state_tau] += alpha * (G - V[state_tau])
                deltas.append(delta)

            if tau >= T-1:
                break
            t += 1  # Next time step
    return V, deltas

# Define a random policy
def random_policy(state):
    return env.actions[np.random.choice(len(env.actions))]

# Run n-step TD evaluation
env = GridWorld()
gamma_values = [0.99]
n_values = [1, 3, 5]

# Store results for plotting
results = {}

def plot_value_function(V, title="Value Function"):
    plt.figure(figsize=(6, 6))
    plt.imshow(V, cmap='coolwarm', origin='upper')
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            plt.text(j, i, f"{V[i, j]:.2f}", ha='center', va='center', color='black')
    plt.colorbar()
    plt.title(title)
    plt.show()

for gamma in gamma_values:
    for n in n_values:
        V, deltas = n_step_td_policy_evaluation(env, random_policy, gamma=gamma, n=n)
        results[(gamma, n)] = (V,deltas)
        plot_value_function(V, title=f"Value Function (γ={gamma}, n={n})")

# Plot Convergence
# plt.figure(figsize=(10, 6))
# for (gamma, n), deltas in results.items():
#     plt.plot(deltas, label=f"γ={gamma}, n={n}")

# plt.xlabel("Iterations")
# plt.ylabel("Δ (Value Function Change)")
# plt.title("Convergence of n-Step TD Policy Evaluation")
# plt.yscale("log")  # Log scale for better visualization
# plt.legend()
# plt.grid()
# plt.show()
