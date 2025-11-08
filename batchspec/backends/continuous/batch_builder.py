import torch

from dataclasses import dataclass
from typing import Union, Optional, List, Tuple
from torch import Tensor
from ..base import PageTable
from .sequence import Sequence, Status
from .scheduler import Workload

@dataclass
class BatchPack:
    """
    Unit of input data for continuous batching.
    Note that KV page table is not included in the batch pack since it persists in the engine.

    input_ids: Input token indices of shape (nnz)
    qo_indptr: Index pointer for Query/Output tokens of shape (n_seqs + 1)
    position_ids: Position offsets for RoPE of shape (nnz)
    """
    input_ids: Tensor # [nnz]
    qo_indptr: Tensor # [n_seqs + 1]
    position_ids_or_offsets: Optional[Tensor] = None # [nnz] (ids) or [n_seqs] (offsets)


class BatchBuilderMixin:
    """
    Mixin for building a batch of sequences into a batch pack based on the scheduler's allocation.
    """

    def build_batch(
        self,
        workloads: List[Workload],
        kv_page_table: PageTable,
        use_position_ids: bool = False,
    ) -> 'BatchPack':
        device = kv_page_table.device

        num_tokens_per_seq = torch.tensor([workload.n_tokens for workload in workloads], dtype=torch.int32, device=device)
        nnz = int(num_tokens_per_seq.sum().item())

        # Allocate memory for input ids and position ids/offsets
        input_ids = torch.empty(nnz, dtype=torch.long, device=device)
        if use_position_ids:
            position_ids = torch.empty(nnz, dtype=torch.int32, device=device)
        else:
            # use position offsets
            position_offsets = kv_page_table.cachelens.clone()
        
        # Write input ids and position ids/offsets to the allocated memory
        write_ptr = 0
        for workload in workloads:
            seq = workload.seq
            n_tokens = workload.n_tokens

            if seq is None:
                input_ids[write_ptr:write_ptr+n_tokens].fill_(self.pad_token_id)
                if use_position_ids:
                    position_ids[write_ptr:write_ptr+n_tokens].fill_(0)
                write_ptr += n_tokens
                continue
            
            start = seq.cur_pos
            end = start + n_tokens
            input_ids[write_ptr:write_ptr+n_tokens].copy_(seq.content[start:end])
            if use_position_ids:
                position_ids[write_ptr:write_ptr+n_tokens].copy_(torch.arange(start, end, dtype=torch.int32, device=device))
            write_ptr += n_tokens

        # Query/Output index pointer
        n_seqs = len(workloads)
        qo_indptr = torch.empty(n_seqs + 1, dtype=torch.int32, device=device)
        qo_indptr[0] = 0; qo_indptr[1:] = torch.cumsum(num_tokens_per_seq, dim=0)

        # Update page table
        kv_page_table.insert_kv(num_tokens_per_seq)

        return BatchPack(
            input_ids=input_ids,
            position_ids_or_offsets=position_ids if use_position_ids else position_offsets,
            qo_indptr=qo_indptr
        )
        