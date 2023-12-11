# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from grid2op.Action import BaseAction


class Agent(ABC):
    @abstractmethod
    def act_eval(self, state, reward) -> BaseAction:
        pass
