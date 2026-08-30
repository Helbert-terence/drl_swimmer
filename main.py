import torch
import torch.nn.functional as F
import torch.optim as optim
from physics.simu import SwimmerEnv
from learner.agents import Actor, Critic

# Hyperparameters
num_episodes = 10000
gamma = 0.99
gae_lambda = 0.95
lr = 3e-4
clip_eps = 0.2
k_epochs = 4
entropy_coef = 0.01

# Initialization
env = SwimmerEnv(n_segments=5, length=40)
state = env.reset()

state_dim = len(state)
action_dim = len(env.motors)
max_action = 5.0

actor = Actor(state_dim, action_dim, max_action)
critic = Critic(state_dim)
actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=lr)


for episode in range(num_episodes):
    state = env.reset()

    states = []
    actions = []
    old_log_probs = []
    rewards = []
    values = []

    # Collect one episode
    truncated = False
    while not truncated:
        state_tensor = torch.FloatTensor(state)
        action, log_prob = actor.act(state_tensor)
        value = critic(state_tensor).item()

        action_numpy = action.detach().numpy()
        next_state, reward, truncated = env.step(action_numpy)

        states.append(state_tensor)
        actions.append(action.detach())
        old_log_probs.append(log_prob.detach())
        rewards.append(reward)
        values.append(value)

        state = next_state

    # Bootstrap value for the last state
    with torch.no_grad():
        last_value = critic(torch.FloatTensor(state)).item()

    # GAE (Generalized Advantage Estimation)
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_val = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma*next_val - values[t]
        gae = delta + gamma*gae_lambda*gae
        advantages.insert(0, gae)

    advantages_tensor = torch.FloatTensor(advantages)
    returns_tensor = advantages_tensor + torch.FloatTensor(values)

    states_tensor = torch.stack(states)
    actions_tensor = torch.stack(actions)
    old_log_probs_tensor = torch.stack(old_log_probs)

    # Normalize the advantages
    advantages_tensor = (advantages_tensor - advantages_tensor.mean())/(advantages_tensor.std() + 1e-8)

    # PPO: k epochs on the same batch
    for _ in range(k_epochs):
        predicted_values = critic(states_tensor).squeeze()
        log_probs, entropy = actor.evaluate(states_tensor, actions_tensor)
        ratio = (log_probs - old_log_probs_tensor).exp()

        surr1 = ratio*advantages_tensor
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps)*advantages_tensor

        actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef*entropy.mean()
        critic_loss = F.mse_loss(predicted_values, returns_tensor)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        critic_optimizer.step()

    # Monitoring
    total_reward = sum(rewards)
    if episode%10 == 0:
        print(f"Episode {episode} | Reward: {total_reward:.1f} | Actor loss: {actor_loss.item():.4f} | Critic loss: {critic_loss.item():.4f}")
    if episode%100 == 0:
        torch.save(actor.state_dict(), "actor.pth")
        torch.save(critic.state_dict(), "critic.pth")

# Save the weights
torch.save(actor.state_dict(), "actor.pth")
torch.save(critic.state_dict(), "critic.pth")