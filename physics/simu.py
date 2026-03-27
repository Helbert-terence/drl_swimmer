import pymunk
import pymunk.pygame_util
import pygame
import math


def create_space():
    space = pymunk.Space()
    space.gravity = (0, 0)
    return space


def create_segment(space, position, length=80, mass=1, radius=3):
    half = length / 2
    moment = pymunk.moment_for_segment(mass, (-half, 0), (half, 0), radius)
    body = pymunk.Body(mass, moment)
    body.position = position
    shape = pymunk.Segment(body, (-half, 0), (half, 0), radius)
    space.add(body, shape)
    return body


def main(nbody):
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    space = create_space()
    length = 80
    body = []
    for i in range(nbody):
        body.append(create_segment(space, (WIDTH / 2 - length*(nbody//2 - i) , HEIGHT / 2), length))
        if i != 0:
            pivot = pymunk.PivotJoint(body[i-1], body[i], (length/2, 0), (-length/2, 0))
            space.add(pivot)

    draw_options = pymunk.pygame_util.DrawOptions(screen)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((200, 220, 255)) 
        space.debug_draw(draw_options) 
        pygame.display.flip() 
        space.step(1/60.0) 
        clock.tick(60)     
    pygame.quit()


if __name__ == "__main__":
    NBODY = 4
    main(NBODY)