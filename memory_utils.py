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

    def train_(self, _states, _ag_ids, _policies, _values):
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


# TODO:  enable multiprocessing
class NNMemory(FlatMemory):
    def __init__(self, model: nn.Module, n_actions: int):
        super().__init__(n_actions)
        self.model = model.float()
        self.ps_: Dict[Tuple[bytes, int], np.ndarray] = {}
        self.vs_: Dict[Tuple[bytes, int], float] = {}
        self.optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

    def train_batch_(self, states, ag_ids, policies, values):
        self.optimizer.zero_grad()
        p_logits, v = self.model.forward(states, ag_ids)
        loss_p = F.kl_div(F.log_softmax(p_logits, dim=-1),
                          policies, reduction='batchmean')
        loss_v = F.mse_loss(v.flatten(), values)
        print("P_loss: ", loss_p.cpu().item(), " V_loss", loss_v.cpu().item())
        (loss_p + loss_v).backward()
        self.optimizer.step()

    def train_(self, _ag_ids, _states, _policies, _values):
        self.model.train()
        mem_size = len(_values)
        states = torch.tensor(_states)
        ag_ids = torch.tensor(_ag_ids)
        policies = torch.tensor(_policies).float()
        value_currs = [v[a] for v, a in zip(_values, ag_ids)]
        values = torch.tensor(value_currs).float()
        if 0 < mem_size < max_batch_size:
            shuffle = torch.randperm(values.shape[0])
            self.train_batch_(
                states=states[shuffle, :].to(self.model.device).long(),
                ag_ids=ag_ids[shuffle].to(self.model.device).long(),
                policies=policies[shuffle, :].to(self.model.device),
                values=values[shuffle].to(self.model.device))
        else:
            n_rounds = int(np.ceil(mem_size / max_batch_size)) + 2
            for _ in range(n_rounds):
                sample = torch.randint(mem_size, (max_batch_size,))
                self.train_batch_(
                    states=states[sample, :].to(self.model.device).long(),
                    ag_ids=ag_ids[sample].to(self.model.device).long(),
                    policies=policies[sample, :].to(self.model.device),
                    values=values[sample].to(self.model.device))

    def get_p(self, s: State, ag_id: int) -> np.ndarray:
        if (s.tobytes(), ag_id) in self.ps_:
            return self.ps_[(s.tobytes(), ag_id)]
        else:
            torch_s = torch.tensor(s).unsqueeze(0).to(self.model.device).long()
            torch_agid = torch.tensor(ag_id).unsqueeze(0).to(self.model.device).long()
            p = self.model.forward_p(torch_s, torch_agid).flatten().cpu().numpy()
            self.ps_[(s.tobytes(), ag_id)] = p
            return p

    def get_v(self, s: State, ag_id: int) -> float:
        if (s.tobytes(), ag_id) in self.vs_:
            return self.vs_[(s.tobytes(), ag_id)]
        else:
            torch_s = torch.tensor(s).unsqueeze(0).to(self.model.device).long()
            torch_agid = torch.tensor(ag_id).unsqueeze(0).to(self.model.device).long()
            v = self.model.forward_v(torch_s, torch_agid).item()
            self.vs_[(s.tobytes(), ag_id)] = v
            return v
