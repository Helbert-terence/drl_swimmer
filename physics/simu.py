import pymunk
import pymunk.pygame_util
import pygame
import math
import numpy as np 
import random
    
class SwimmerEnv:
    def __init__(self, n_segments, length, max_step = 2000):
        self.n_segments = n_segments
        self.length = length
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.step_count = 0
        self.max_steps = max_step
        self.bodies = []
        self.motors = [] 
        self.dt = 1/60
        
        for i in range(n_segments):
            x_pos = 400 - length * (n_segments // 2 - i)
            body = self._create_segment((x_pos, 300), length)
            self.bodies.append(body)
            
            if i != 0:
                pivot = pymunk.PivotJoint(self.bodies[i-1], self.bodies[i], (length/2, 0), (-length/2, 0))
                self.space.add(pivot)

                motor = pymunk.SimpleMotor(self.bodies[i-1], self.bodies[i], 0)
                motor.max_force = 1e6 
                self.space.add(motor)
                self.motors.append(motor)
                limit = pymunk.RotaryLimitJoint(self.bodies[i-1], self.bodies[i],-135*np.pi/180, 135*np.pi/180)
                self.space.add(limit)

        x = random.uniform(50, 750)
        y = random.uniform(50, 550)
        self.prey = self._create_prey((x,y))

    def _create_prey(self, position, radius=10):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = position   
        shape = pymunk.Circle(body, radius)
        shape.sensor = True 
        shape.color = pygame.Color("red") 
        self.space.add(body, shape)
        return body
    
    def move_prey_randomly(self):
        x = random.uniform(50, 750)
        y = random.uniform(50, 550)
        self.prey.position = (x, y)

    def _create_segment(self, position, length):
        mass = 1
        moment = pymunk.moment_for_segment(mass, (-length/2, 0), (length/2, 0), 3)
        body = pymunk.Body(mass, moment)
        body.position = position
        shape = pymunk.Segment(body, (-length/2, 0), (length/2, 0), 3)
        shape.filter = pymunk.ShapeFilter(group=1) 
        self.space.add(body, shape)
        return body

    def _apply_water_physics(self, body):
        C_PAR = 0.5    # Résistance quand le segment avance tout droit 
        C_PERP = 5.0   # Résistance quand le segment se déplace sur le côté 
        C_ANGULAR = 1000.0 # Résistance à la rotation sur lui-même dans l'eau

        direction = pymunk.Vec2d(1, 0).rotated(body.angle)
        normal = direction.perpendicular()

        v = body.velocity
        v_par = v.dot(direction)  
        v_perp = v.dot(normal)   

        force_par = direction * (-C_PAR * v_par)
        force_perp = normal * (-C_PERP * v_perp)
        force_totale = force_par + force_perp

        body.apply_force_at_world_point(force_totale, body.position)   
        body.torque = -C_ANGULAR * body.angular_velocity

    def get_state(self):
        state = []
        for body in self.bodies:
            state.append(body.angle)
            state.append(body.angular_velocity)   
        head = self.bodies[0]
        dx = self.prey.position.x - head.position.x
        dy = self.prey.position.y - head.position.y
        state.extend([dx/800, dy/600])
        return np.array(state, dtype=np.float32)
    
    def reset(self):
        self.__init__(self.n_segments, self.length)
        return self.get_state()

    def step(self, torques):
        self.step_count += 1
        for i in range(len(self.motors)):
            vitesse_cible = np.clip(torques[i], -5.0, 5.0) 
            self.motors[i].rate = vitesse_cible

        for body in self.bodies:
            self._apply_water_physics(body)
        head = self.bodies[0]
        prey = self.prey
        dist_before = np.sqrt((head.position.x - prey.position.x)**2 + (head.position.y - prey.position.y)**2)
        self.space.step(self.dt)
        dist_after = np.sqrt((head.position.x - prey.position.x)**2 + (head.position.y - prey.position.y)**2)
        reward = dist_before - dist_after 
        if dist_after < 10 :
            reward += 100
            self.move_prey_randomly()
        Flag = self.step_count > self.max_steps
        next_state = self.get_state()
        return next_state, reward, Flag