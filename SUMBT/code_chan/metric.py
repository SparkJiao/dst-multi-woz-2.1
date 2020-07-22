from typing import Iterable

import torch


class Average:
    """
    This [`Metric`](./metric.md) breaks with the typical `Metric` API and just stores values that were
    computed in some fashion outside of a `Metric`.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    `Metric` API.
    """

    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

    def __call__(self, value, num=1):
        """
        # Parameters
        value : `float`
            The value to average.
        """
        self._total_value += list(self.detach_tensors(value))[0] * num
        self._count += num

    def get_metric(self, reset: bool = False):
        """
        # Returns
        The average of all values that were passed to `__call__`.
        """
        average_value = self._total_value / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return average_value

    def reset(self):
        self._total_value = 0.0
        self._count = 0

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)
