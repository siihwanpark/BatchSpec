import time

from dataclasses import dataclass
from typing import List, Optional

from batchspec.backends.base import PageTable
from .sequence import Status, Sequence


class _C:
    HDR = "\x1b[95m"; OK = "\x1b[92m"; WARN = "\x1b[93m"; ERR = "\x1b[91m"
    DIM = "\x1b[2m"; BOLD = "\x1b[1m"; RST = "\x1b[0m"
def _c(use, s, col): return f"{col}{s}{_C.RST}" if use else s


def _status_char(st: Status) -> str:
    return {Status.PENDING:"-", Status.PREFILL:"P", Status.DECODE:"D", Status.COMPLETE:"C", Status.FIRST_DRAFT:"FD", Status.DRAFT_VERIFY:"DV"}.get(st, "?")


@dataclass
class _Snap:
    step: int
    slots_status: List[str]
    slots_seqid: List[Optional[str]]
    free_slots: List[int]
    waiting_ids: List[str]
    workloads: List[tuple]
    next_tokens: Optional[List[int]]
    cachelens: List[int]
    accept_nums: Optional[List[int]]

class SchedulerTracer:
    """
    Wraps around your Scheduler to print a per-step diagnostic view.

    Assumptions:
      - workloads are sorted by slot_idx (as in your plan()).
      - next_tokens are indexed by slot_idx (as in your process_results()).
    """

    def __init__(self, scheduler, *, color: bool = True, show_queues: bool = True, print_every: int = 10):
        self.sch = scheduler
        self.color = color
        self.show_queues = show_queues
        self.print_every = print_every
        self._t0 = time.time()

    def _seq_id(self, s: Optional[Sequence]) -> Optional[str]:
        if s is None: return None
        sid = getattr(s, "sid", None) or getattr(s, "id", None)
        try:
            return sid if sid is not None else str(s.seq_id)
        except Exception:
            return str(id(s))

    def _snapshot(self, step: int, workloads, kv_page_table, next_tokens_tensor=None, accept_nums_tensor=None) -> _Snap:
        k = self.sch.max_concurrency
        slots_status, slots_seqid = [], []
        for i in range(k):
            seq = self.sch.slots[i]
            if seq is None:
                slots_status.append("-"); slots_seqid.append(None)
            else:
                slots_status.append(_status_char(seq.status))
                slots_seqid.append(self._seq_id(seq))

        free_slots = list(self.sch.free_slots)
        waiting_ids = [self._seq_id(s) for s in list(self.sch.waiting_queue)]

        wls = [(w.slot_idx, self._seq_id(w.seq), w.n_tokens) for w in workloads]

        nxt = None
        if next_tokens_tensor is not None:
            # next_tokens are indexed by slot_idx in your current design
            nxt = []
            for i in range(k):
                try:
                    nxt.append(int(next_tokens_tensor[i].item()))
                except Exception:
                    nxt.append(None)

        cachelens = kv_page_table.cachelens.cpu().tolist()

        accept_nums = None
        if accept_nums_tensor is not None:
            accept_nums = accept_nums_tensor.cpu().tolist()

        return _Snap(step, slots_status, slots_seqid, free_slots, waiting_ids, wls, nxt, cachelens, accept_nums)

    def _print_header(self):
        print(_c(self.color, "\n=== Continuous Batching Trace ===", _C.HDR))
        print("Legend: P=PREFILL, D=DECODE, C=COMPLETE, FD=FIRST_DRAFT, DV=DRAFT_VERIFY, -=empty slot")

    def _render(self, snap: _Snap, phase: str):
        self._print_header()

        elapsed = time.time() - self._t0
        if phase == "PLAN":
            print(_c(self.color, "\n========= START OF STEP ==========", _C.HDR))
        print(_c(self.color, f"\n[{phase}] step={snap.step}  elapsed={elapsed:.3f}s", _C.BOLD))

        # Slots line
        k = len(snap.slots_status)
        slots_line = " | ".join(
            f"{i:02d}:{snap.slots_status[i]}{f'({snap.slots_seqid[i]})' if snap.slots_seqid[i] else ''}"
            for i in range(k)
        )
        print("Slots: ", slots_line)

        # Workloads
        if snap.workloads:
            wl_line = " ; ".join(f"s{si:02d}→{sid if sid else '∅'}×{n}"
                                 for (si, sid, n) in snap.workloads)
            print(_c(self.color, "Plan:   ", _C.OK) + wl_line)
        else:
            print(_c(self.color, "Plan:   (none)", _C.WARN))

        # Next tokens (optional)
        if snap.next_tokens is not None:
            # show only for slots used in plan
            used = {si for si, *_ in snap.workloads}
            nt_pairs = [f"s{si:02d}:{snap.next_tokens[si]}" for si in sorted(used)]
            print("Out:    " + ", ".join(nt_pairs))

        # Accept numbers (optional)
        if snap.accept_nums is not None:
            used = {si for si, *_ in snap.workloads}
            an_pairs = [f"s{si:02d}:{snap.accept_nums[si]}" for si in sorted(used)]
            print(_c(self.color, "Accept: ", _C.OK) + ", ".join(an_pairs))

        if self.show_queues:
            print(_c(self.color, "Queues:", _C.DIM),
                  f"waiting={snap.waiting_ids}  free={snap.free_slots}  ")
            # Cachelens
            if snap.cachelens:
                cl_line = " ; ".join(f"s{si:02d}:{cl}" for si, cl in enumerate(snap.cachelens))
                print(_c(self.color, "Cache Lengths:  ", _C.OK) + cl_line)
        
        if phase == "UPDATE":
            print(_c(self.color, "\n========= END OF STEP ==========", _C.HDR))

    # Call before model forward
    def on_plan(self, step: int, workloads, kv_page_table: PageTable):
        if step % self.print_every != 0:
            return
        snap = self._snapshot(step, workloads, kv_page_table, next_tokens_tensor=None, accept_nums_tensor=None)
        self._render(snap, phase="PLAN")

    # Call after process_results (pass the same workloads and produced next_tokens)
    def on_update(self, step: int, workloads, kv_page_table: PageTable, next_tokens_tensor=None, accept_nums_tensor=None):
        if step % self.print_every != 0:
            return
        snap = self._snapshot(step, workloads, kv_page_table, next_tokens_tensor=next_tokens_tensor, accept_nums_tensor=accept_nums_tensor)
        self._render(snap, phase="UPDATE")
