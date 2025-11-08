from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List

import torch
from torch import Tensor

class Status(Enum):
    PENDING = auto()
    PREFILL = auto()
    DECODE = auto()
    COMPLETE = auto()

@dataclass
class Sequence:
    seq_id: int
    max_seq_len: int

    prompt_ids: Tensor
    prompt_len: int
    content: Tensor = field(init=False)

    status: Status = field(default=Status.PENDING)
    num_prefilled_tokens: int = field(default=0)
    num_generated_tokens: int = field(default=0)
    cur_pos: int = field(default=0)
    
    def __post_init__(self):
        self.content = torch.empty(self.max_seq_len, dtype=torch.int32, device="cuda")
        self.content[:self.prompt_len] = self.prompt_ids

    def get_id(self) -> int:
        return self.seq_id
