"""Optional graph-region timing, inactive during normal training/evaluation."""
from contextlib import contextmanager, nullcontext
import time

import torch

_active = None


class GraphMeasurements:
    def __init__(self, device):
        self.device = device
        self.records = []

    @contextmanager
    def region(self):
        if self.device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(self.device)
            start.record(stream)
            try:
                yield
            finally:
                end.record(stream)
                self.records.append((start, end))
        else:
            start = time.perf_counter()
            try:
                yield
            finally:
                self.records.append(time.perf_counter()-start)

    def seconds(self):
        if not self.records:
            return None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            return sum(start.elapsed_time(end) for start, end in self.records)/1000.
        return sum(self.records)


def graph_region():
    return _active.region() if _active is not None else nullcontext()


@contextmanager
def measure_graph(device):
    global _active
    previous, collector = _active, GraphMeasurements(device)
    _active = collector
    try:
        yield collector
    finally:
        _active = previous
