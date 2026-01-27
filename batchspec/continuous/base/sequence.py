from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import torch
from torch import Tensor

class Status(Enum):
    PENDING = auto()
    PREFILL = auto()
    DECODE = auto()
    COMPLETE = auto()
    
    # MTP-specific status
    FIRST_DRAFT = auto()
    DRAFT_VERIFY = auto()

@dataclass
class Sequence:
    seq_id: int

    # Generation will be stopped either when the sequence length reaches max_seq_len 
    # or num_generated_tokens reaches max_gen_len
    max_seq_len: int
    max_gen_len: int

    prompt_ids: Tensor
    prompt_len: int
    content: Tensor = field(init=False)

    status: Status = field(default=Status.PENDING)
    num_prefilled_tokens: int = field(default=0)
    num_generated_tokens: int = field(default=0)
    
    # `cur_pos` is intended to be the position of the lastest written token to the content
    # hence it should be num_prefilled_tokens + num_generated_tokens - 1
    cur_pos: int = field(default=0)
    
    # MTP-specific fields
    accept_lens: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.content = torch.empty(self.prompt_len + self.max_gen_len, dtype=torch.int32, device="cuda")
        self.content[:self.prompt_len] = self.prompt_ids
