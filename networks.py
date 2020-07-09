import torch
from torch import nn, Tensor
import torch.nn.functional as F

# TODO action query, add train part, enable multiprocessing
from config import lr


class TttNet(nn.Module):
    def __init__(self,
                 h: int,
                 n_agents: int,
                 max_actions: int,
                 hidden_dim: int = 16,
                 agent_embed_dim: int = 4):
        super().__init__()
        self.state_t = nn.Linear(h ** 2, hidden_dim)
        self.agent_embed = nn.Embedding(n_agents + 1, agent_embed_dim)
        self.get_v = nn.Linear(hidden_dim * agent_embed_dim, 1)
        self.get_p_logits = nn.Linear(hidden_dim * agent_embed_dim, max_actions)
        self.optimizer = torch.optim.Adam(lr=lr, params=self.parameters())

    def get_embed(self, s, ag_id) -> Tensor:
        a_embd = self.agent_embed(ag_id + 1)
        s_embd = self.agent_embed(s + 1).transpose(-1, -2)
        s_embd = self.state_t(s_embd).transpose(-1, -2)
        final_embs = (s_embd * a_embd.unsqueeze(-2)).flatten(-2, -1)
        return final_embs

    def forward(self, s, ag_id):
        embd = self.get_embed(s, ag_id)
        return self.get_p_logits(embd), self.get_v(embd)

    def forward_p(self, s, ag_id) -> Tensor:
        self.eval()
        with torch.no_grad():
            embd = self.get_embed(s, ag_id)
            return F.softmax(self.get_p_logits(embd), dim=-1)

    def forward_v(self, s, ag_id) -> Tensor:
        self.eval()
        with torch.no_grad():
            embd = self.get_embed(s, ag_id)
            return self.get_v(embd)

    def train_batch_(self, states, ag_ids, policies, values):
        self.optimizer.zero_grad()
        p_logits, v = self.forward(states, ag_ids)
        loss_p = F.kl_div(p_logits, policies, reduction='batchmean')
        loss_v = F.mse_loss(v.flatten(), values)
        (loss_p + loss_v).backward()
        self.optimizer.step()
