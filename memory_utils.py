from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from config import lr, max_batch_size, State


class FlatMemory:
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        pass

    def train_(self):
        pass

    def add_(self, _, __, ___):
        pass

    def add_with_symmetry_(self, ag_id, s, policy, symmetry):
        pass

    def assign_values_(self, _):
        pass

    def clear_(self):
        pass

    def get_p(self, _, __):
        return np.ones(self.n_actions) / self.n_actions

    @staticmethod
    def get_v(_, __):
        return 0



class NNMemory(FlatMemory):
    def __init__(self, model: nn.Module, n_actions: int, device: str):
        super().__init__(n_actions)
        self.model = model.to(device).float()
        self.device = device
        self.ag_ids = []
        self.states = []
        self.policies = []
        self.values = []
        self.ps_: Dict[Tuple[bytes, int], np.ndarray] = {}
        self.vs_: Dict[Tuple[bytes, int], float] = {}

    def clear_(self):
        self.ag_ids = []
        self.states = []
        self.policies = []
        self.values = []
        self.ps_ = {}
        self.vs_ = {}

    def add_(self, ag_id, s, policy):
        self.ag_ids.append(ag_id)
        self.states.append(s)
        self.policies.append(policy)

    def add_with_symmetry_(self, ag_id, s, policy, symmetry):
        states, policies = symmetry(s, policy)
        self.states += states
        self.ag_ids += len(states) * [ag_id]
        self.policies += policies

    def assign_values_(self, total_rewards):
        len_diff = len(self.ag_ids) - len(self.values)
        self.values += [total_rewards] * len_diff


    def train_(self):
        self.model.train()
        mem_size = len(self.values)
        states = torch.tensor(self.states)
        ag_ids = torch.tensor(self.ag_ids)
        policies = torch.tensor(self.policies).float()
        value_currs = [v[a] for v, a in zip(self.values, self.ag_ids)]
        values = torch.tensor(value_currs).float()
        if 0 < mem_size < max_batch_size:
            shuffle = torch.randperm(len(self.values))
            self.model.train_batch_(
                states = states[shuffle, :].to(self.device),
                ag_ids = ag_ids[shuffle].to(self.device),
                policies = policies[shuffle, :].to(self.device),
                values = values[shuffle].to(self.device))
        else:
            n_rounds = int(torch.ceil(mem_size / max_batch_size)) + 2
            for _ in range(n_rounds):
                sample = torch.randint(mem_size, max_batch_size)
                self.model.train_batch_(
                    states=states[sample, :].to(self.device),
                    ag_ids=ag_ids[sample].to(self.device),
                    policies=policies[sample, :].to(self.device),
                    values=values[sample].to(self.device))

    def get_p(self, s: State, ag_id: int) -> np.ndarray:
        if (s.tobytes(), ag_id) in self.ps_:
            return self.ps_[(s.tobytes(), ag_id)]
        else:
            torch_s = torch.tensor(s).unsqueeze(0).to(self.device)
            torch_agid = torch.tensor(ag_id).unsqueeze(0).to(self.device)
            p = self.model.forward_p(torch_s, torch_agid).flatten().cpu().numpy()
            self.ps_[(s.tobytes(), ag_id)] = p
            return p

    def get_v(self, s: State, ag_id: int) -> float:
        if (s.tobytes(), ag_id) in self.vs_:
            return self.vs_[(s.tobytes(), ag_id)]
        else:
            torch_s = torch.tensor(s).unsqueeze(0).to(self.device)
            torch_agid = torch.tensor(ag_id).unsqueeze(0).to(self.device)
            v = self.model.forward_v(torch_s, torch_agid).item()
            self.vs_[(s.tobytes(), ag_id)] = v
            return v
