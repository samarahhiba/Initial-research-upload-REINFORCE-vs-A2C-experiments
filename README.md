# Zero-Sum Policy Gradient Methods for Crash-Cost Markov Games
Car-Bus Research Game CS298 (ongoing) 

**Author:** Samarah Hiba  
**Institution:** University of Wisconsin–Madison  
**Research Project:** CS298 (AI Research)

---
## Research Questions
1. Do policy gradient methods converge in zero-sum Markov games?
2. Does REINFORCE exhibit policy oscillation?
3. Does A2C stabilize training compared to vanilla policy gradients?
4. How does crash-cost magnitude affect equilibrium behavior?


## Overview
This project implements:
- REINFORCE
- Advantage Actor-Critic (A2C)
- Deep Minimax-Q

on a grid-based zero-sum crash-cost driving environment.

## Motivation
This work explores convergence behavior in zero-sum Markov games and policy oscillation under policy gradient methods.

## Methods
- Policy Gradient (REINFORCE)
- A2C
- LP-based Minimax solver
- Target networks + replay buffers

## Results
- Policy oscillations observed in zero-sum setup
- Averaging strategies converge toward mixed equilibrium
- Comparison plots included

