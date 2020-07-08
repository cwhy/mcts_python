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

    def add_with_symmetry(self, ag_id, s, policy, symmetry):
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
        self.optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

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

    def add_with_symmetry(self, ag_id, s, policy, symmetry):
        states, policies = symmetry(s, policy)
        self.states += states
        self.ag_ids += len(states) * [ag_id]
        self.policies += policies

    def assign_values_(self, total_rewards):
        len_diff = len(self.ag_ids) - len(self.values)
        self.values += [total_rewards] * len_diff

    def train_(self):
        self.model.train()
        if 0 < len(self.values) < max_batch_size:
            self.optimizer.zero_grad()
            states = torch.tensor(self.states).to(self.device)
            ag_ids = torch.tensor(self.ag_ids).to(self.device)
            policies = torch.tensor(self.policies).to(self.device).float()
            value_currs = [v[a] for v, a in zip(self.values, self.ag_ids)]
            values = torch.tensor(value_currs).to(self.device).float()
            shuffle = torch.randperm(len(self.values))
            states = states[shuffle, :]
            ag_ids = ag_ids[shuffle]
            policies = policies[shuffle, :]
            values = values[shuffle]
            p_logits, v = self.model.forward(states, ag_ids)
            loss_p = F.kl_div(p_logits, policies, reduction='batchmean')
            loss_v = F.mse_loss(v.flatten(), values)
            (loss_p + loss_v).backward()
            self.optimizer.step()
        else:
            print(" Too long, not training")

    def get_p(self, s: State, ag_id: int) -> np.ndarray:
        if (s.tobytes(), ag_id) in self.ps_:
            return self.ps_[(s.tobytes(), ag_id)]
        else:
            torch_s = torch.tensor(s).unsqueeze(0)
            torch_agid = torch.tensor(ag_id).unsqueeze(0)
            p = self.model.forward_p(torch_s, torch_agid).flatten().numpy()
            self.ps_[(s.tobytes(), ag_id)] = p
            return p

    def get_v(self, s: State, ag_id: int) -> float:
        if (s.tobytes(), ag_id) in self.vs_:
            return self.vs_[(s.tobytes(), ag_id)]
        else:
            torch_s = torch.tensor(s).unsqueeze(0)
            torch_agid = torch.tensor(ag_id).unsqueeze(0)
            v = self.model.forward_v(torch_s, torch_agid).item()
            self.vs_[(s.tobytes(), ag_id)] = v
            return v
