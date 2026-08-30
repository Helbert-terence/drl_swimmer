import os
from collections import deque

import pygame
import torch
from PIL import Image

from physics.simu import SwimmerEnv
from learner.agents import Actor

model_path = "actor.pth"
gif_path = "assets/swimmer.gif"
record = True           # set to False for a replay without recording
gif_fps = 30            # frames per second in the gif
gif_scale = 0.5         # gif size compared to the window
pre_seconds = 5         # recorded time before the capture
post_seconds = 10     # recorded time after the capture
frame_skip = 60//gif_fps

env = SwimmerEnv(n_segments=5, length=40)
state = env.reset()

state_dim = len(state)
action_dim = len(env.motors)
max_action = 5.0

actor = Actor(state_dim, action_dim, max_action)
actor.load_state_dict(torch.load(model_path, weights_only=True))
actor.eval()

pre_frames = pre_seconds*gif_fps
post_frames = post_seconds*gif_fps
frames = deque(maxlen=pre_frames + post_frames)
post_count = None


def capture(screen):
    # Copy the window content into a PIL image
    size = (int(screen.get_width()*gif_scale), int(screen.get_height()*gif_scale))
    small = pygame.transform.smoothscale(screen, size)
    array = pygame.surfarray.array3d(small).swapaxes(0, 1)
    return Image.fromarray(array)


def save_gif(images):
    folder = os.path.dirname(gif_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=int(1000/gif_fps), loop=0)
    print(f"Gif saved in {gif_path}")


while True:
    state = env.reset()
    truncated = False
    total_reward = 0
    step = 0

    while not truncated:
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action = actor(state_tensor)
        state, reward, truncated = env.step(action.numpy())
        total_reward += reward
        env.render()
        if env.screen is None:  # window closed
            exit()

        if record:
            # A reward above 1 means that a prey was eaten
            if reward > 1.0 and post_count is None:
                post_count = post_frames
            if step%frame_skip == 0:
                frames.append(capture(env.screen))
                if post_count is not None:
                    post_count -= 1
                    if post_count <= 0:
                        save_gif(list(frames))
                        record = False

        step += 1

    print(f"Episode done | Reward: {total_reward:.1f}")