from torch.utils.data import Sampler


class GroupedBatchSampler(Sampler[list[int]]):
    """Yields batches where all indices belong to the same group (parent dir)."""

    def __init__(self, group_indices: dict[str, list[int]], batch_size: int):
        self.batches: list[list[int]] = []
        for indices in group_indices.values():
            for i in range(0, len(indices), batch_size):
                self.batches.append(indices[i : i + batch_size])

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)
