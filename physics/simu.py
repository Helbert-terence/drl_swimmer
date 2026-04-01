import math
import random
import numpy as np
import pymunk
import pygame
import pymunk.pygame_util


class SwimmerEnv:
    def __init__(self, n_segments=8, length=20, max_steps=2000):
        self.n_segments = n_segments
        self.length = length
        self.max_steps = max_steps
        self.dt = 1 / 60
        self.step_count = 0
        self.space = None
        self.bodies = []
        self.motors = []
        self.prey = None
        self.screen = None
        self._build()

    def _build(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.bodies = []
        self.motors = []

        for i in range(self.n_segments):
            x = 400 - self.length * (self.n_segments // 2 - i)
            body = self._create_segment((x, 300))
            self.bodies.append(body)

            if i > 0:
                prev = self.bodies[i - 1]
                half = self.length / 2

                pivot = pymunk.PivotJoint(prev, body, (half, 0), (-half, 0))
                motor = pymunk.SimpleMotor(prev, body, 0)
                motor.max_force = 1e6
                limit = pymunk.RotaryLimitJoint(prev, body, -math.radians(135), math.radians(135))

                self.space.add(pivot, motor, limit)
                self.motors.append(motor)

        self.prey = self._create_prey(self._random_position())

    def _create_segment(self, position):
        half = self.length / 2
        mass = 1
        radius = 3
        moment = pymunk.moment_for_segment(mass, (-half, 0), (half, 0), radius)

        body = pymunk.Body(mass, moment)
        body.position = position

        shape = pymunk.Segment(body, (-half, 0), (half, 0), radius)
        shape.filter = pymunk.ShapeFilter(group=1)

        self.space.add(body, shape)
        return body

    def _create_prey(self, position, radius=10):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = position

        shape = pymunk.Circle(body, radius)
        shape.sensor = True
        shape.color = pygame.Color("red")

        self.space.add(body, shape)
        return body

    def _apply_water_physics(self, body):
        C_PAR = 0.5
        C_PERP = 5.0
        C_ANGULAR = 1000.0

        direction = pymunk.Vec2d(1, 0).rotated(body.angle)
        normal = direction.perpendicular()
        v = body.velocity

        force_par = direction * (-C_PAR * v.dot(direction))
        force_perp = normal * (-C_PERP * v.dot(normal))

        body.apply_force_at_world_point(force_par + force_perp, body.position)
        body.torque = -C_ANGULAR * body.angular_velocity

    def get_state(self):
        state = []

        # Angles relatifs entre segments consécutifs
        for i in range(1, len(self.bodies)):
            rel_angle = self.bodies[i].angle - self.bodies[i - 1].angle
            state.append(rel_angle)
        # Vitesse angulaire de chaque segment
        for body in self.bodies:
            state.append(body.angular_velocity)

        head = self.bodies[0]
        dx = self.prey.position.x - head.position.x
        dy = self.prey.position.y - head.position.y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-8

        # Direction normalisée vers la proie
        state.append(dx / dist)
        state.append(dy / dist)
        # Distance normalisée
        state.append(dist / 800)
        # Cos(angle entre heading tête et direction proie)
        head_dir = pymunk.Vec2d(1, 0).rotated(head.angle)
        state.append(head_dir.x * dx / dist + head_dir.y * dy / dist)
        # Vitesse de la tête
        state.append(head.velocity.x / 100)
        state.append(head.velocity.y / 100)

        return np.array(state, dtype=np.float32)

    def reset(self):
        self.step_count = 0
        self._build()
        return self.get_state()

    def step(self, torques):
        self.step_count += 1

        for i, motor in enumerate(self.motors):
            motor.rate = np.clip(torques[i], -5.0, 5.0)

        for body in self.bodies:
            self._apply_water_physics(body)

        self.space.step(self.dt)

        # Vitesse de la tête projetée vers la proie
        head = self.bodies[0]
        to_prey = pymunk.Vec2d(self.prey.position.x - head.position.x,
                               self.prey.position.y - head.position.y)
        dist = to_prey.length + 1e-8
        vel_toward = head.velocity.dot(to_prey / dist)

        reward = vel_toward / 50.0      # approche = positif, fuite = négatif, immobile = 0
        reward -= 0.02                   # pénalité temporelle : rester immobile coûte
        if dist < 15:
            reward += 5.0
            self._respawn_prey()

        truncated = self.step_count >= self.max_steps
        return self.get_state(), reward, truncated

    def _distance_to_prey(self):
        head = self.bodies[0].position
        prey = self.prey.position
        return math.sqrt((head.x - prey.x) ** 2 + (head.y - prey.y) ** 2)

    def _respawn_prey(self):
        for shape in self.prey.shapes:
            self.space.remove(shape)
        self.space.remove(self.prey)
        self.prey = self._create_prey(self._random_position())

    def _random_position(self):
        return (random.uniform(50, 750), random.uniform(50, 550))
    
    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.screen = None          # désactive le rendu
                pygame.display.quit()       # ferme juste la fenêtre
                return

        self.screen.fill((20, 60, 120))       # fond bleu eau
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(60)