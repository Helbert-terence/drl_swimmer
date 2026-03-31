import torch
from physics.simu import SwimmerEnv
from learner.agents import Actor

env = SwimmerEnv()
state = env.reset()

state_dim = len(state)
action_dim = len(env.motors)
max_action = 5.0

actor = Actor(state_dim, action_dim, max_action)
actor.load_state_dict(torch.load("actor.pth", weights_only=True))
actor.eval()

while True:
    state = env.reset()
    truncated = False
    total_reward = 0

    while not truncated:
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action = actor(state_tensor)
        state, reward, truncated = env.step(action.numpy())
        total_reward += reward
        env.render()
        if env.screen is None:  # fenêtre fermée
            exit()

    print(f"Episode fini | Reward: {total_reward:.1f}")
