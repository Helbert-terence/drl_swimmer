# drl_swimmer

A 2D physics-based environment where articulated creatures made of line segments learn to swim and  hunt randomly placed prey using Deep Reinforcement Learning.

/physics contains simu.py with the class SwimmerEnv that create all the simulation, the rendering and the rewarding function.

/learner contains agents.py that have class Actor and Critic to create both neural network.

main.py launch the PPO training loop with chosen hyperparameters and creature's parameters (number of segments and length of segments), it automatically save .pth files for actors' and critics' neural network.

play.py replays a trained actor from a .pth file.
