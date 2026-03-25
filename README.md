# drl_swimmer

A 2D physics-based environment where articulated creatures made of line segments learn to "swim" using Deep Reinforcement Learning.

Unlike standard rigid-body simulations, flailing in a vacuum doesn't generate forward momentum here. This project implements anisotropic drag (water resistance).

## Key Features

* **Custom Water Physics:** Simulates fluid dynamics through anisotropic friction (segments face high resistance when moving laterally, low resistance longitudinally).
* **Configurable Morphology:** Easily define the number of segments, their lengths, and the joints connecting them before launching a training session.


