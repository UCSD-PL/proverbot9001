import torch
from torch import nn

class VModel(nn.Module):
    tactic_embedding: nn.Embedding
    prediction_network: nn.Module
    prev_tactic_vocab_size: int

    def __init__(self, local_context_embedding_size: int,
                 previous_tactic_vocab_size: int,
                 previous_tactic_embedding_size: int,
                 hidden_size: int,
                 num_layers: int) -> None:
        super().__init__()
        self.tactic_embedding = nn.Embedding(previous_tactic_vocab_size,
                                             previous_tactic_embedding_size)
        layers: List[nn.Module] = [nn.Linear(local_context_embedding_size +
                                   previous_tactic_embedding_size,
                                   hidden_size)]
        for layer_idx in range(num_layers - 1):
            layers += [nn.ReLU(), nn.Linear(hidden_size, hidden_size)]
        layers += [nn.ReLU(), nn.Linear(hidden_size, 1), nn.Sigmoid()]
        self.prediction_network = nn.Sequential(*layers)
        self.prev_tactic_vocab_size = previous_tactic_vocab_size
        pass
    def forward(self, local_context_embeddeds: torch.FloatTensor,
                prev_tactic_indices: torch.LongTensor) -> torch.FloatTensor:
        return self.prediction_network(torch.cat(
            (local_context_embeddeds,
             self.tactic_embedding(prev_tactic_indices)),
            dim=1))
