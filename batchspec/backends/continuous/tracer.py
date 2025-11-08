# tracer.py
import time
from dataclasses import dataclass
from typing import List, Optional

from .sequence import Status, Sequence
from ..base import PageTable

# --- tiny ANSI helper ---
class _C:
    HDR = "\x1b[95m"; OK = "\x1b[92m"; WARN = "\x1b[93m"; ERR = "\x1b[91m"
    DIM = "\x1b[2m"; BOLD = "\x1b[1m"; RST = "\x1b[0m"
def _c(use, s, col): return f"{col}{s}{_C.RST}" if use else s

def _status_char(st: Status) -> str:
    return {Status.PENDING:"-", Status.PREFILL:"P", Status.DECODE:"D", Status.COMPLETE:"C"}.get(st, "?")

@dataclass
class _Snap:
    step: int
    slots_status: List[str]           # e.g., ["P","D","-","C",...]
    slots_seqid: List[Optional[str]]  # your seq.get_id() or similar
    free_slots: List[int]
    waiting_ids: List[str]
    workloads: List[tuple]            # (slot_idx, seq_id, n_tokens)
    next_tokens: Optional[List[int]]  # aligned to slot_idx (your current design)
    cachelens: List[int]              # cachelens of the KV page table

class SchedulerTracer:
    """
    Wraps around your Scheduler to print a per-step diagnostic view.

    Assumptions:
      - workloads are sorted by slot_idx (as in your plan()).
      - next_tokens are indexed by slot_idx (as in your process_results()).
    """

    def __init__(self, scheduler, *, color: bool = True, show_queues: bool = True, show_header_every: int = 25):
        self.sch = scheduler
        self.color = color
        self.show_queues = show_queues
        self.show_header_every = show_header_every
        self._t0 = time.time()

    def _seq_id(self, s: Optional[Sequence]) -> Optional[str]:
        if s is None: return None
        # prefer a short stable identifier
        sid = getattr(s, "sid", None) or getattr(s, "id", None)
        try:
            return sid if sid is not None else str(s.get_id())
        except Exception:
            return str(id(s))

    def _snapshot(self, step: int, workloads, kv_page_table, next_tokens_tensor=None) -> _Snap:
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

        return _Snap(step, slots_status, slots_seqid, free_slots, waiting_ids, wls, nxt, cachelens)

    def _print_header(self):
        print(_c(self.color, "\n=== Continuous Batching Trace ===", _C.HDR))
        print("Legend: P=PREFILL, D=DECODE, C=COMPLETE, -=empty slot")

    def _render(self, snap: _Snap, phase: str):
        # if snap.step % self.show_header_every == 0:
        #     self._print_header()

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
        snap = self._snapshot(step, workloads, kv_page_table, next_tokens_tensor=None)
        self._render(snap, phase="PLAN")

    # Call after process_results (pass the same workloads and produced next_tokens)
    def on_update(self, step: int, workloads, kv_page_table: PageTable, next_tokens_tensor):
        snap = self._snapshot(step, workloads, kv_page_table, next_tokens_tensor=next_tokens_tensor)
        self._render(snap, phase="UPDATE")
