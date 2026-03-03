
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

ACTIONS_RPS = ["R", "P", "S"]

def rps_payoff(a1: int, a2: int) -> float:
    # 0 Rock, 1 Paper, 2 Scissors
    if a1 == a2:
        return 0.0
    # Rock beats Scissors, Paper beats Rock, Scissors beats Paper
    wins = {(0,2), (1,0), (2,1)}
    return 1.0 if (a1,a2) in wins else -1.0

@dataclass
class RPSGame:
    """Stateless 2-player zero-sum Rock-Paper-Scissors."""
    n_actions: int = 3

    def reset(self):
        s = 0  # single state
        return s

    def step(self, a1: int, a2: int):
        r = rps_payoff(a1,a2)
        done = True
        s2 = 0
        info = {}
        return s2, r, -r, done, info

    @property
    def n_states(self):
        return 1


ACTIONS_GRID = ["U","D","L","R","S"]  # include Stay for stability

MOVE = {
    0: (0,-1), # U
    1: (0, 1), # D
    2: (-1,0), # L
    3: (1, 0), # R
    4: (0, 0), # S
}

@dataclass
class CarBusGame:
    """Zero-sum 3x3 grid Markov game (no wrap-around).
    State = (car_x, car_y, bus_x, bus_y). Car tries to reach goal; bus tries to crash car.
    Rewards are for car; bus gets negative.
    """
    grid_size: int = 3
    crash_cost: float = 10.0
    goal_reward: float = 10.0
    step_cost: float = 0.1
    max_steps: int = 25
    start_car: tuple[int,int] = (0,0)
    start_bus: tuple[int,int] = (2,2)
    goal: tuple[int,int] = (2,0)  # top-right

    def reset(self):
        self.t = 0
        self.car = list(self.start_car)
        self.bus = list(self.start_bus)
        return self._state_id()

    @property
    def n_actions(self):
        return 5

    @property
    def n_states(self):
        g = self.grid_size
        return g*g*g*g

    def _state_id(self):
        g = self.grid_size
        cx, cy = self.car
        bx, by = self.bus
        return ((cx*g + cy)*g + bx)*g + by

    def _clip(self, x: int, y: int):
        g = self.grid_size
        x = min(max(x,0), g-1)
        y = min(max(y,0), g-1)
        return x,y

    def step(self, a1: int, a2: int):
        # simultaneous moves
        dcx, dcy = MOVE[a1]
        dbx, dby = MOVE[a2]
        cx, cy = self._clip(self.car[0]+dcx, self.car[1]+dcy)
        bx, by = self._clip(self.bus[0]+dbx, self.bus[1]+dby)
        self.car = [cx,cy]
        self.bus = [bx,by]
        self.t += 1

        # terminal checks
        r = -self.step_cost
        done = False
        if (cx,cy) == (bx,by):
            r = -self.crash_cost
            done = True
        elif (cx,cy) == self.goal:
            r = self.goal_reward
            done = True
        elif self.t >= self.max_steps:
            done = True

        s2 = self._state_id()
        return s2, r, -r, done, {}
