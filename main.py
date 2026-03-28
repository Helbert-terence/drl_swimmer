import pygame
import pymunk.pygame_util
from physics.simu import SwimmerEnv

env = SwimmerEnv(n_segments=4, length=30)
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)

obs = env.reset()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = [1, -1, 2]

    obs, reward, truncated = env.step(action)
    screen.fill((200, 220, 255))
    env.space.debug_draw(draw_options)
    pygame.display.flip()
    clock.tick(60)