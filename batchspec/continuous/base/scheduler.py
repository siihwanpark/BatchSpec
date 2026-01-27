from typing import List, Optional, Deque, Dict
from collections import deque

import torch
from torch import Tensor
from dataclasses import dataclass

from .sequence import Status, Sequence


# ---------------------------------------------------------------------
# Workload: represents a unit of work for one sequence within a batch.
# Each workload tells the engine: “for slot i, process n_tokens from seq”.
# ---------------------------------------------------------------------
@dataclass
class Workload:
    """
    A single processing unit representing one sequence in the current batch step.

    Attributes:
        slot_idx (int): Index of the slot in the batch where the sequence is located.
        seq (Sequence): The sequence currently occupying the slot.
        n_tokens (int): Number of tokens to process for this sequence in the current batch step.
                        Typically, this is the length of the prefill chunk or the decode length.
    """
    slot_idx: int
    seq: Sequence
    n_tokens: int


# ---------------------------------------------------------------------
# Scheduler: orchestrates dynamic batching across prefilling and decoding.
# ---------------------------------------------------------------------
class Scheduler:
    """
    A decode-first + leftover scheduler for continuous batching.

    Core components:
    - slots: 
        Fixed-size array of length `max_concurrency`. Each slot represents a
        persistent "lane" in the batch that holds at most one sequence.
        The model processes all active slots simultaneously each iteration.

    - free_slots:
        Stack (LIFO list) of currently available slot indices.
        When a sequence completes, its slot index is returned here and reused
        for a new waiting sequence.

    - waiting_queue:
        FIFO queue (deque) of sequences that have not yet been assigned to any slot.
        When a slot becomes free, one of these sequences is popped and attached.

    - workloads:
        The output of `plan()`. Each workload defines what each slot should
        process in the upcoming forward pass (how many tokens, which sequence, etc.).

    - realloc:
        Boolean flag indicating whether a reallocation step (replanning) is needed.
        This becomes True whenever a sequence finishes, transitions phase, or
        when there are both free slots and waiting sequences.
    """

    def __init__(self, max_concurrency: int, prefill_chunk_len: int):
        self.max_concurrency: int = max_concurrency
        self.prefill_chunk_len: int = prefill_chunk_len

        self.slots: List[Optional[Sequence]] = [None] * max_concurrency
        self.free_slots: List[int] = list(range(max_concurrency - 1, -1, -1))
        self.waiting_queue: Deque[Sequence] = deque()

        self.total: int = 0
        self.realloc: bool = True
        self.completed: List[Sequence] = []


    def is_done(self) -> bool:
        """Return True if all sequences are finished."""
        return len(self.completed) == self.total


    def add_sequences(self, sequences: List[Sequence]) -> None:
        """
        Add new sequences to the waiting queue.

        Args:
            sequences: A list of sequences to be added to the waiting queue.
        """
        self.total += len(sequences)
        self.waiting_queue.extend(sequences)


    def profile_maximum_concurrency(self) -> int:
        """
        Optional: profile the effective concurrency limit.
        This could consider runtime constraints (e.g., GPU memory or kernel limits)
        and return a maximum number of sequences that can be active at the same time.

        This method is intended to be acted as a hint for the engine to adjust the batch size.
        Especially when the `max_concurrency` is set to None.
        """
        raise NotImplementedError("Not implemented yet.")


    def _assign_waiting_to_free_slots(self) -> None:
        """
        Assign waiting sequences to any available free slots.

        For each available slot:
          1. Pop a new sequence from waiting_queue.
          2. Attach it to that slot.
          3. Set its status to PREFILL (it must first process the prompt).
        """
        while self.free_slots and self.waiting_queue:
            slot_idx = self.free_slots.pop()
            seq = self.waiting_queue.popleft()

            seq.status = Status.PREFILL
            self.slots[slot_idx] = seq


    def _release_kv_cache_for_free_slots(self, kv_page_table: "PageTable") -> None:
        """
        Release KV-cache entries for all free slots in the page table.

        The KV page table keeps track of how many tokens each slot currently holds in the KV cache.
        Construct a deletion mask (del_lens) for the free slots and instruct
        the page table to clear those tokens in one operation.
        """
        if not self.free_slots:
            return

        # Get the number of tokens each slot currently holds in the KV cache
        cachelens = kv_page_table.cachelens

        # Construct a deletion mask for the free slots
        del_lens = torch.zeros_like(cachelens)
        del_lens[self.free_slots] = cachelens[self.free_slots]

        # Delete the tokens from the KV cache
        kv_page_table.delete_kv(del_lens)


    def plan(self) -> tuple[List[Workload], bool]:
        """
        Create the workload plan for the next forward step.

        The plan determines which slots will be active and how many tokens each
        will process. It always prioritizes decode-ready slots first, since
        decoding is latency-critical.

        Returns:
            List[Workload]: ordered by slot index, describing the batch layout.
            bool: True if the scheduler has reached the tail generation.
        """
        workloads: List[Workload] = []

        # (1) Assign new waiting sequences to free slots if possible.
        if self.free_slots and self.waiting_queue:
            self._assign_waiting_to_free_slots()

        # (2) Separate the slots into decode-ready and prefill-ready.
        decode_ready_slots: Deque[int] = deque()
        prefill_ready_slots: Deque[int] = deque()
        for slot_idx, seq in enumerate(self.slots):
            # Empty slot
            if seq is None:
                continue

            if seq.status == Status.DECODE:
                decode_ready_slots.append(slot_idx)
            elif seq.status == Status.PREFILL:
                prefill_ready_slots.append(slot_idx)
            else:
                raise ValueError(f"Sequence {seq.seq_id} is in an invalid status: {seq.status}.")

        # (2) Fill workloads from decode-ready slots first (decode-first policy).
        while decode_ready_slots and len(workloads) < self.max_concurrency:
            slot_idx = decode_ready_slots.popleft()
            seq = self.slots[slot_idx]

            assert seq is not None, f"Slot {slot_idx} is unassigned."
            assert seq.status == Status.DECODE, f"Sequence {seq.seq_id} not in decode state."
            assert seq.max_seq_len - (seq.prompt_len + seq.num_generated_tokens) >= 1, \
                f"Sequence {seq.seq_id} has less than 1 tokens left."

            workloads.append(Workload(slot_idx=slot_idx, seq=seq, n_tokens=1))

        # (3) Fill remaining capacity with prefill-ready slots.
        while prefill_ready_slots and len(workloads) < self.max_concurrency:
            slot_idx = prefill_ready_slots.popleft()
            seq = self.slots[slot_idx]

            assert seq is not None, f"Slot {slot_idx} is unassigned."
            assert seq.status == Status.PREFILL, f"Sequence {seq.seq_id} not in prefill state."
            assert seq.prompt_len - seq.num_prefilled_tokens > 0, \
                f"Sequence {seq.seq_id} has no tokens left to prefill."

            workloads.append(Workload(slot_idx=slot_idx, seq=seq,
                                      n_tokens=min(self.prefill_chunk_len, seq.prompt_len - seq.num_prefilled_tokens)))

        # (4) Fill remaining capacity with dummy workloads.
        if len(workloads) < self.max_concurrency:
            assert len(self.free_slots) == self.max_concurrency - len(workloads), f"num free slots mismatch with remaining capacity: {self.max_concurrency} - {len(workloads)} != {len(self.free_slots)}."
            workloads.extend([Workload(slot_idx=slot_idx, seq=None, n_tokens=1) for slot_idx in self.free_slots])

        # (5) Sort by slot index so the model input order matches slot order
        # At this point, workloads are simply assignments of how many tokens to process for each slot.
        workloads.sort(key=lambda x: x.slot_idx)

        # (6) Check if the scheduler has reached the tail generation.
        # The tail generation is the situation where there are free slots but no waiting sequences.
        if self.free_slots and not self.waiting_queue:
            tail = True
        else:
            tail = False
        
        return workloads, tail


    def process_results(
        self,
        workloads: List[Workload],
        next_tokens: Tensor,
        eos_token_id: int,
        kv_page_table: "PageTable",
    ) -> None:
        """
        Integrate model outputs (next_tokens) back into scheduler state.

        For each workload:
            - Update the sequence progress counters (prefilled/generated tokens).
            - Check if its phase transitions (PREFILL→DECODE or DECODE→COMPLETE).
            - Move the slot index into the appropriate ready queue or free list.
            - Reclaim KV cache entries if slots became free.

        Args:
            workloads: A list of workloads that have been processed.
            next_tokens: The next tokens generated by the workloads.
            eos_token_id: The EOS token id to check if the sequence is completed.
            kv_page_table: The KV page table to be updated.
        """
        status_change_occurred = False
        prefill_len_changed = False

        for workload in workloads:
            slot_idx = workload.slot_idx
            seq = workload.seq
            n_tokens = workload.n_tokens
            next_token = next_tokens[slot_idx]

            if seq is None:
                # This is a dummy workload, i.e. a slot that is not assigned to any sequence
                # Nothing to do
                continue

            assert seq.seq_id == self.slots[slot_idx].seq_id, f"Sequence {seq.seq_id} is not in slot {slot_idx}."

            # PREFILL → PREFILL or DECODE transition
            if seq.status == Status.PREFILL:
                seq.num_prefilled_tokens += n_tokens
                seq.cur_pos += n_tokens

                if seq.num_prefilled_tokens == seq.prompt_len:
                    # Prefill completed → move to decoding phase
                    seq.status = Status.DECODE
                    seq.content[seq.cur_pos] = next_token
                    seq.num_generated_tokens += 1
                    
                    status_change_occurred = True
                else:
                    # Still more tokens to prefill → check if the workload should be reallocated
                    if seq.prompt_len - seq.num_prefilled_tokens < self.prefill_chunk_len:
                        # Prefill length is less than the chunk length → indicates the workload should be reallocated (with different n_tokens)
                        prefill_len_changed = True

            # DECODE → DECODE or COMPLETE transition
            elif seq.status == Status.DECODE:
                assert n_tokens == 1, f"n_tokens for sequence {seq.seq_id} must be 1 in decode state."
                seq.content[seq.cur_pos+1] = next_token
                seq.num_generated_tokens += 1
                seq.cur_pos += 1

                eos_token_generated = (next_token == eos_token_id)
                gen_budget_exhausted = (seq.num_generated_tokens + 1 > seq.max_gen_len)
                model_budget_exhausted = (seq.prompt_len + seq.num_generated_tokens + 1 > seq.max_seq_len)
                if eos_token_generated or gen_budget_exhausted or model_budget_exhausted:
                    # Decoding completed → free the slot
                    seq.status = Status.COMPLETE
                    self.completed.append(seq)
                    self.slots[slot_idx] = None
                    self.free_slots.append(slot_idx)
                    status_change_occurred = True

            else:
                raise ValueError(f"Sequence {seq.seq_id} is in an invalid status: {seq.status}.")

        # Release KV cache entries for any slots that became free
        self._release_kv_cache_for_free_slots(kv_page_table)

        # If any status changed or new slots became available or prefill length changed, trigger reallocation next step
        self.realloc = (bool(self.free_slots) and bool(self.waiting_queue))\
                        or status_change_occurred or prefill_len_changed


# ---------------------------------------------------------------------
# MTScheduler: orchestrates dynamic batching for Speculative Decoding with MTP.
# ---------------------------------------------------------------------
class MTPScheduler(Scheduler):
    """
    A scheduler for continuous batching for Speculative Decoding with MTP.
    """

    def __init__(self, max_concurrency: int, prefill_chunk_len: int, draft_length: int):
        super().__init__(max_concurrency, prefill_chunk_len)
        self.draft_length: int = draft_length

    def plan(self) -> tuple[List[Workload], bool]:
        """
        Create the workload plan for the next forward step.

        The plan determines which slots will be active and how many tokens each
        will process. It always prioritizes decode-ready slots first, since
        decoding is latency-critical.

        Returns:
            List[Workload]: ordered by slot index, describing the batch layout.
            bool: True if the scheduler has reached the tail generation.
        """
        workloads: List[Workload] = []

        # (1) Assign new waiting sequences to free slots if possible.
        if self.free_slots and self.waiting_queue:
            self._assign_waiting_to_free_slots()

        # (2) Separate the slots into draft-ready and prefill-ready.
        draft_ready_slots: Deque[int] = deque()
        prefill_ready_slots: Deque[int] = deque()
        for slot_idx, seq in enumerate(self.slots):
            # Empty slot
            if seq is None:
                continue

            if seq.status in [Status.FIRST_DRAFT, Status.DRAFT_VERIFY]:
                draft_ready_slots.append(slot_idx)
            elif seq.status == Status.PREFILL:
                prefill_ready_slots.append(slot_idx)
            else:
                raise ValueError(f"Sequence {seq.seq_id} is in an invalid status: {seq.status}.")

        # (3) Fill workloads from draft-ready slots first (draft-first policy).
        while draft_ready_slots and len(workloads) < self.max_concurrency:
            slot_idx = draft_ready_slots.popleft()
            seq = self.slots[slot_idx]

            assert seq is not None, f"Slot {slot_idx} is unassigned."
            assert seq.status in [Status.FIRST_DRAFT, Status.DRAFT_VERIFY], f"Sequence {seq.seq_id} not in draft state."
            assert seq.max_seq_len - (seq.prompt_len + seq.num_generated_tokens) >= self.draft_length, \
                f"Sequence {seq.seq_id} has less than {self.draft_length} tokens left."

            workloads.append(Workload(slot_idx=slot_idx, seq=seq, n_tokens=self.draft_length))

        # (4) Fill remaining capacity with prefill-ready slots.
        while prefill_ready_slots and len(workloads) < self.max_concurrency:
            slot_idx = prefill_ready_slots.popleft()
            seq = self.slots[slot_idx]

            assert seq is not None, f"Slot {slot_idx} is unassigned."
            assert seq.status == Status.PREFILL, f"Sequence {seq.seq_id} not in prefill state."
            assert seq.prompt_len - seq.num_prefilled_tokens > 0, \
                f"Sequence {seq.seq_id} has no tokens left to prefill."

            workloads.append(Workload(slot_idx=slot_idx, seq=seq,
                                      n_tokens=min(self.prefill_chunk_len, seq.prompt_len - seq.num_prefilled_tokens)))

        # (4) Fill remaining capacity with dummy workloads.
        if len(workloads) < self.max_concurrency:
            assert len(self.free_slots) == self.max_concurrency - len(workloads), f"num free slots mismatch with remaining capacity: {self.max_concurrency} - {len(workloads)} != {len(self.free_slots)}."
            workloads.extend([Workload(slot_idx=slot_idx, seq=None, n_tokens=1) for slot_idx in self.free_slots])

        # (5) Sort by slot index so the model input order matches slot order
        # At this point, workloads are simply assignments of how many tokens to process for each slot.
        workloads.sort(key=lambda x: x.slot_idx)

        # (6) Check if the scheduler has reached the tail generation.
        # The tail generation is the situation where there are free slots but no waiting sequences.
        if self.free_slots and not self.waiting_queue:
            tail = True
        else:
            tail = False
        
        return workloads, tail


    def process_results(
        self,
        workloads,
        next_tokens,
        verify_buffer,
        accept_nums,
        eos_token_id: int,
        kv_page_table,
    ) -> None:
        """
        Integrate model outputs (next_tokens, draft_tokens, bonus_tokens, accept_nums) back into scheduler state.

        For each workload:
            - Update the sequence progress counters (prefilled/generated tokens, draft tokens, bonus tokens, accept nums).
            - Check if its phase transitions (PREFILL→FIRST_DRAFT or FIRST_DRAFT→DRAFT_VERIFY or DRAFT_VERIFY→COMPLETE).
            - Move the slot index into the appropriate ready queue or free list.
            - Reclaim KV cache entries if slots became free.

        Args:
            workloads: A list of workloads that have been processed.
            next_tokens: The next tokens generated by the workloads.
            draft_tokens: The draft tokens generated by the workloads.
            accept_nums: The number of accepted tokens by the workloads.
            eos_token_id: The EOS token id to check if the sequence is completed.
            kv_page_table: The KV page table to be updated.
        """

        slots = self.slots
        status_change = False
        prefill_len_changed = False

        # Collect active slot indices and classify by status
        slot_idx_list = [workload.slot_idx for workload in workloads]
        prefill_slot_indices, first_draft_slot_indices, draft_verify_slot_indices = [], [], []
        for slot_idx in slot_idx_list:
            seq = slots[slot_idx]
            if seq is None: continue
            status = seq.status
            if status == Status.PREFILL: prefill_slot_indices.append(slot_idx)
            elif status == Status.FIRST_DRAFT: first_draft_slot_indices.append(slot_idx)
            elif status == Status.DRAFT_VERIFY: draft_verify_slot_indices.append(slot_idx)

        # PREFILL: progress + prompt exhaustion -> FIRST_DRAFT transition
        if prefill_slot_indices:
            for slot_idx in prefill_slot_indices:
                seq = slots[slot_idx]
                
                # Find this slot's n_tokens from workloads
                n_tokens = 0
                for workload in workloads:
                    if workload.slot_idx == slot_idx:
                        n_tokens = int(workload.n_tokens); break
                
                seq.num_prefilled_tokens += n_tokens
                seq.cur_pos += n_tokens
            
            # Status transition: only slots that match condition
            for slot_idx in prefill_slot_indices:
                seq = slots[slot_idx]
                if seq.num_prefilled_tokens == seq.prompt_len:
                    seq.status = Status.FIRST_DRAFT
                    seq.content[seq.cur_pos] = next_tokens[slot_idx]
                    seq.num_generated_tokens += 1
                    status_change = True
                    prefill_len_changed = True
                else:
                    if seq.prompt_len - seq.num_prefilled_tokens < self.prefill_chunk_len:
                        prefill_len_changed = True

        # FIRST_DRAFT -> DRAFT_VERIFY transition
        if first_draft_slot_indices:
            for slot_idx in first_draft_slot_indices:
                slots[slot_idx].status = Status.DRAFT_VERIFY
            status_change = True

        # DRAFT_VERIFY: all slots have at least 1 acceptance
        if draft_verify_slot_indices:
            device = accept_nums.device
            draft_verify_slot_indices = torch.tensor(draft_verify_slot_indices, device=device, dtype=torch.long)

            # Batch accept_nums = accept_nums[draft_verify_slot_indices]
            accept_nums = accept_nums[draft_verify_slot_indices].to(torch.long)

            # Copy verify_buffer[slot_idx, :accept_num] to seq.content[seq.cur_pos+1 : seq.cur_pos+1+accept_num]
            for slot_idx, accept_num in zip(draft_verify_slot_indices.tolist(), accept_nums.tolist()):
                seq = slots[slot_idx]
                seq.content[seq.cur_pos+1 : seq.cur_pos+1+accept_num] = verify_buffer[slot_idx, :accept_num]
                seq.num_generated_tokens += accept_num
                seq.cur_pos += accept_num
                seq.accept_lens.append(int(accept_num))

            # Early termination decision: vectorized
            # Last acceptance token: verify_buffer[slot_idx, accept_num-1]
            last_cols = accept_nums - 1
            last_tokens = verify_buffer[draft_verify_slot_indices, last_cols]
            eos_mask = last_tokens.eq(eos_token_id)

            # Bonus decision (next step safety margin included): max_accept_len = D+1
            max_accept_len = self.draft_length + 1
            num_generated_tokens = torch.tensor([slots[slot_idx].num_generated_tokens for slot_idx in draft_verify_slot_indices.tolist()],
                            device=device, dtype=torch.long)
            prompt_len = torch.tensor([slots[slot_idx].prompt_len for slot_idx in draft_verify_slot_indices.tolist()],
                            device=device, dtype=torch.long)
            max_gen_len = torch.tensor([slots[slot_idx].max_gen_len for slot_idx in draft_verify_slot_indices.tolist()],
                            device=device, dtype=torch.long)
            max_seq_len = torch.tensor([slots[slot_idx].max_seq_len for slot_idx in draft_verify_slot_indices.tolist()],
                            device=device, dtype=torch.long)

            gen_exhausted   = (num_generated_tokens + max_accept_len >= max_gen_len)
            model_exhausted = (prompt_len + num_generated_tokens + max_accept_len >= max_seq_len)
            done_mask = eos_mask | gen_exhausted | model_exhausted

            # Complete slot batch processing
            if done_mask.any():
                done_slots = draft_verify_slot_indices[done_mask].tolist()
                for slot_idx in done_slots:
                    seq = slots[slot_idx]
                    seq.status = Status.COMPLETE
                    self.completed.append(seq)
                    self.slots[slot_idx] = None
                    self.free_slots.append(slot_idx)
                status_change = True

        # KV return (batch)
        self._release_kv_cache_for_free_slots(kv_page_table)

        # Reallocation trigger
        self.realloc = (bool(self.free_slots) and bool(self.waiting_queue)) or status_change or prefill_len_changed
