import torch
import torch.nn.functional as F
import torch.optim as optim
from physics.simu import SwimmerEnv
from learner.agents import Actor, Critic

# Hyperparamètres 
NUM_EPISODES = 10000
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 3e-4
CLIP_EPS = 0.2
K_EPOCHS = 4
ENTROPY_COEF = 0.01 
RENDER_EVERY = 10**9

# Initialisation
env = SwimmerEnv(n_segments=10, length=20)
state = env.reset()

state_dim = len(state)
action_dim = len(env.motors)
max_action = 5.0

actor = Actor(state_dim, action_dim, max_action)
critic = Critic(state_dim)
actor_optimizer = optim.Adam(actor.parameters(), lr=LR)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR)


for episode in range(NUM_EPISODES):
    show = (episode % RENDER_EVERY == 0)
    state = env.reset()

    states = []
    actions = []
    old_log_probs = []
    rewards = []
    values = []

    # Collecte d'un épisode
    truncated = False
    while not truncated:
        state_tensor = torch.FloatTensor(state)
        action, log_prob = actor.act(state_tensor)
        value = critic(state_tensor).item()

        action_numpy = action.detach().numpy()
        next_state, reward, truncated = env.step(action_numpy)
        if show:
            env.render()

        states.append(state_tensor)
        actions.append(action.detach())
        old_log_probs.append(log_prob.detach())
        rewards.append(reward)
        values.append(value)

        state = next_state

    # Bootstrap value pour le dernier état
    with torch.no_grad():
        last_value = critic(torch.FloatTensor(state)).item()

    # GAE (Generalized Advantage Estimation)
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_val = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + GAMMA * next_val - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * gae
        advantages.insert(0, gae)

    advantages_tensor = torch.FloatTensor(advantages)
    returns_tensor = advantages_tensor + torch.FloatTensor(values)

    states_tensor = torch.stack(states)
    actions_tensor = torch.stack(actions)
    old_log_probs_tensor = torch.stack(old_log_probs)

    # Normaliser les advantages
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

    # PPO : K epochs sur le même batch
    for _ in range(K_EPOCHS):
        values = critic(states_tensor).squeeze()
        log_probs, entropy = actor.evaluate(states_tensor, actions_tensor)
        ratio = (log_probs - old_log_probs_tensor).exp()

        surr1 = ratio * advantages_tensor
        surr2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * advantages_tensor

        actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy.mean()
        critic_loss = F.mse_loss(values, returns_tensor)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        critic_optimizer.step()

    # Suivi
    total_reward = sum(rewards)
    if episode % 10 == 0:
        print(f"Episode {episode} | Reward: {total_reward:.1f} | Actor loss: {actor_loss.item():.4f} | Critic loss: {critic_loss.item():.4f}")
    if episode % 100 == 0:
        torch.save(actor.state_dict(), "actor.pth")
        torch.save(critic.state_dict(), "critic.pth")

# Sauvegarde des poids
torch.save(actor.state_dict(), "actor.pth")
torch.save(critic.state_dict(), "critic.pth")
print("Poids sauvegardés dans actor.pth et critic.pth")