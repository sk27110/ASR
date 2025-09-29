from typing import Dict, Optional


class SimpleMetricTracker:

    def __init__(self):
        self._sum: Dict[str, float] = {}
        self._count: Dict[str, int] = {}

    def reset(self) -> None:
        self._sum.clear()
        self._count.clear()

    def update(self, key: str, value: float, n: int = 1) -> None:
        self._sum[key] = self._sum.get(key, 0.0) + float(value) * n
        self._count[key] = self._count.get(key, 0) + int(n)

    def avg(self, key: str) -> Optional[float]:
        if key not in self._count or self._count[key] == 0:
            return None
        return self._sum[key] / self._count[key]

    def result(self) -> Dict[str, float]:
        return {k: (self._sum[k] / self._count[k]) for k in self._sum if self._count.get(k, 0) > 0}
