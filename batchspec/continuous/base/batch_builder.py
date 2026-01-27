import torch

from dataclasses import dataclass, field
from typing import Union, Optional, List, Tuple, Dict
from torch import Tensor
from batchspec.backends.base import PageTable
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
    position_ids_or_offsets: Tensor # [nnz] (ids) or [n_seqs] (offsets)

    # MTP-specific fields
    attn_mask: Optional[Tensor] = None # [_] where the length depends on the cached KV length of each sequence
    gate_mask: Optional[Tensor] = None # [nnz] where 0 for regular tokens, 1 for masks
    
    # map of status to the index of the sequence in the batch
    # DEPRECATED: Use the status_map in the scheduler instead.
    status_map: Dict[str, Optional[Tensor]] = field(default_factory=lambda: {"PREFILL": None, "FIRST_DRAFT": None, "DRAFT_VERIFY": None, "NULL": None})


class BatchBuilderMixin:
    """
    Mixin for building a batch of sequences into a batch pack based on the scheduler's allocation.
    """

    def build_batch(
        self,
        workloads: List[Workload],
        kv_page_table: PageTable,
    ) -> 'BatchPack':
        device = kv_page_table.device

        num_tokens_per_seq = torch.tensor([workload.n_tokens for workload in workloads], dtype=torch.int32, device=device)
        nnz = int(num_tokens_per_seq.sum().item())
        status_map = {"PREFILL": [], "DECODE": [], "NULL": []}

        # Allocate memory for input ids
        input_ids = torch.empty(nnz, dtype=torch.long, device=device)
        
        # Write input ids and position ids/offsets to the allocated memory
        write_ptr = 0
        for workload in workloads:
            seq = workload.seq
            n_tokens = workload.n_tokens

            if seq is None:
                input_ids[write_ptr:write_ptr+n_tokens].fill_(self.pad_token_id)
                write_ptr += n_tokens
                status_map["NULL"].append(workload.slot_idx)
                continue

            if seq.status == Status.PREFILL:
                status_map["PREFILL"].append(workload.slot_idx)
            elif seq.status == Status.DECODE:
                status_map["DECODE"].append(workload.slot_idx)
            
            start = seq.cur_pos
            end = start + n_tokens
            input_ids[write_ptr:write_ptr+n_tokens].copy_(seq.content[start:end])
            write_ptr += n_tokens

        # Query/Output index pointer
        n_seqs = len(workloads)
        qo_indptr = torch.empty(n_seqs + 1, dtype=torch.int32, device=device)
        qo_indptr[0] = 0; qo_indptr[1:] = torch.cumsum(num_tokens_per_seq, dim=0)

        # Finalize status_map
        status_map = {status: torch.tensor(indices, dtype=torch.int64, device=device) if indices else None
                      for status, indices in status_map.items()}

        return BatchPack(
            input_ids=input_ids,
            qo_indptr=qo_indptr,
            position_ids_or_offsets=kv_page_table.cachelens.clone(),
            status_map=status_map,
        )


class MTPBatchBuilderMixin:
    """
    Mixin for building a batch of sequences into a batch pack based on the scheduler's allocation for MTP.
    """

    def build_batch(
        self,
        workloads: List[Workload],
        kv_page_table: PageTable,
        draft_tokens_buffer: Tensor,
    ) -> 'BatchPack':
        device = kv_page_table.device
        cachelens = kv_page_table.cachelens

        num_tokens_per_seq = torch.tensor([self._get_actual_n_tokens_for_mtp(workload) for workload in workloads], dtype=torch.int32, device=device)
        nnz = int(num_tokens_per_seq.sum().item())

        # Allocate memory for input ids and position ids
        input_ids = torch.empty(nnz, dtype=torch.long, device=device)
        position_ids = torch.empty(nnz, dtype=torch.int32, device=device)
        gate_mask = torch.empty(nnz, dtype=torch.int32, device=device)
        attn_mask_arr = []
        status_map = {"PREFILL": [], "FIRST_DRAFT": [], "DRAFT_VERIFY": [], "NULL": []}

        # Write input ids and position ids/offsets to the allocated memory
        write_ptr = 0
        for slot_idx, workload in enumerate(workloads):
            seq = workload.seq
            if seq is None:
                n_tokens = workload.n_tokens
                input_ids[write_ptr:write_ptr+n_tokens].fill_(self.pad_token_id)
                position_ids[write_ptr:write_ptr+n_tokens].fill_(0)
                gate_mask[write_ptr:write_ptr+n_tokens].fill_(0)
                attn_mask_arr.append(torch.zeros(n_tokens, device=device))

                write_ptr += n_tokens
                status_map["NULL"].append(slot_idx)
                continue

            if seq.status == Status.PREFILL:
                n_tokens = workload.n_tokens
                start, end = seq.cur_pos, seq.cur_pos + n_tokens
                
                seq_input_ids = seq.content[start:end]
                seq_pos_ids = torch.arange(start, end, dtype=torch.int32, device=device)
                seq_gate_mask = torch.zeros(n_tokens, dtype=torch.int32, device=device)
                seq_attn_mask = torch.tril(torch.ones((n_tokens, n_tokens), device=device)) # causal mask
                status_map["PREFILL"].append(slot_idx)

            elif seq.status == Status.FIRST_DRAFT:
                draft_len = workload.n_tokens
                n_tokens = draft_len + 1
                seq_input_ids, seq_gate_mask = self._interleave_mask_tokens(seq.content[seq.cur_pos][None], draft_len)
                seq_pos_ids = cachelens[slot_idx] + torch.arange(n_tokens, device=device)
                seq_attn_mask = torch.tril(torch.ones((n_tokens, n_tokens), device=device)) # causal mask
                status_map["FIRST_DRAFT"].append(slot_idx)

            elif seq.status == Status.DRAFT_VERIFY:
                draft_len = workload.n_tokens
                n_tokens = (draft_len + 1) ** 2
                seq_input_ids, seq_gate_mask = self._interleave_mask_tokens(draft_tokens_buffer[slot_idx], draft_len)
                seq_pos_ids = cachelens[slot_idx] + self.common_position_ids
                seq_attn_mask = self.common_attn_mask
                status_map["DRAFT_VERIFY"].append(slot_idx)
            
            # Setup attn_mask
            past_kv_attn_mask = torch.ones((n_tokens, cachelens[slot_idx]), device=device)
            attn_mask = torch.cat((past_kv_attn_mask, seq_attn_mask), dim=-1)
            attn_mask_arr.append(attn_mask.flatten())

            # Write input ids, position ids and gate mask to the allocated memory
            input_ids[write_ptr:write_ptr+n_tokens].copy_(seq_input_ids)
            position_ids[write_ptr:write_ptr+n_tokens].copy_(seq_pos_ids)
            gate_mask[write_ptr:write_ptr+n_tokens].copy_(seq_gate_mask)
            write_ptr += n_tokens

        # Query/Output index pointer
        n_seqs = len(workloads)
        qo_indptr = torch.empty(n_seqs + 1, dtype=torch.int32, device=device)
        qo_indptr[0] = 0; qo_indptr[1:] = torch.cumsum(num_tokens_per_seq, dim=0)

        # Finalize attn_mask
        attn_mask = torch.cat(attn_mask_arr, dim=0).contiguous().to(device=device, dtype=torch.bool)

        # Finalize status_map
        status_map = {status: torch.tensor(indices, dtype=torch.int64, device=device) if indices else None
                      for status, indices in status_map.items()}

        return BatchPack(
            input_ids=input_ids,
            qo_indptr=qo_indptr,
            position_ids_or_offsets=position_ids,
            attn_mask=attn_mask,
            gate_mask=gate_mask,
            status_map=status_map,
        )

    # ============ Helper Functions =============
    @staticmethod
    def _get_actual_n_tokens_for_mtp(workload: Workload) -> int:
        if workload.seq is None:
            return workload.n_tokens
        elif workload.seq.status == Status.PREFILL:
            return workload.n_tokens
        elif workload.seq.status == Status.FIRST_DRAFT:
            return workload.n_tokens + 1
        elif workload.seq.status == Status.DRAFT_VERIFY:
            return (workload.n_tokens + 1) ** 2
        else:
            raise ValueError(f"Invalid status: {workload.seq.status}")

    @torch.no_grad()
    def _interleave_mask_tokens(self, input_ids: Tensor, draft_len: int) -> Tuple[Tensor, Tensor]:
        """Interleave mask tokens with input tokens.
        
        Transforms [x_0, x_1, ...] into [x_0, m_1, ..., m_k, x_1, m_1, ..., m_k, ...]
        where m_i are mask tokens and k is the draft length.
        
        Args:
            input_ids: Input token IDs of shape (seq_len,)
            draft_len: Draft length

        Returns:
            Tuple of (interleaved_ids, gate_mask)
            - interleaved_ids: Shape (seq_len * (draft_len + 1),)
            - gate_mask: 0 for tokens, 1 for masks
        """
        assert self.mask_token_id is not None, "Mask token ID is not set (Intended to be set in the engine)."
        L = len(input_ids)
        D = draft_len

        # Output length = L * (D + 1)
        out_len = L * (D + 1)

        # 1) Allocate and fill with mask_token_id
        out_ids = torch.empty(out_len, dtype=input_ids.dtype, device=input_ids.device)
        out_ids.fill_(self.mask_token_id)

        # View as [L, D+1] and write tokens at slot 0 of each block
        view_ids = out_ids.view(L, D + 1)
        view_ids[:, 0] = input_ids  # tokens at the first position of each (D+1)-block

        # 2) gate_mask: 1 for masks, 0 for tokens
        gate_mask = torch.ones(out_len, dtype=input_ids.dtype, device=input_ids.device)
        view_gate = gate_mask.view(L, D + 1)
        view_gate[:, 0] = 0  # token slots

        return out_ids, gate_mask
