# Learning-Latent-Dynamics-for-Robotic-Pushing-Task
This is a control-from-pixel (visuomotor policy) project that utilizes a VAE (Variational Autoencoder) to perform a robotic pushing task by learning latent space dynamics.

# Planar Pushing Learning From Images

This project involves learning the dynamics of a planar pushing task directly from images.

## State Space
- The **state space** is a **32 × 32 grayscale image** captured from an **overhead camera**.  
- It encodes the block's position and orientation on a planar surface.

<div align="center">
    <img src="Img/state_space.png" alt="State Space and Action Space" width="500px">
</div>

*Figure: The robot action space for the planar pushing task.*

## Action Space
The action space is parameterized as \(\mathbf{u} = [p, \phi, \ell]^\top \in \mathbb{R}^3\):
1. **\(p \in [-1, 1]\):** Pushing location along the block's lower edge.
2. **\(\phi \in [-\frac{\pi}{2}, \frac{\pi}{2}]\):** Pushing angle relative to the block.
3. **\(\ell \in [0, 1]\):** Pushing length, as a fraction of the max length (\(0.1 \, \text{m}\)).

## Overview
- The robot's pusher interacts with the block along its lower edge.
- Actions control the pushing location, direction, and distance.
- The goal is to learn the relationship between actions (\(\mathbf{u}\)) and the resulting state changes.
