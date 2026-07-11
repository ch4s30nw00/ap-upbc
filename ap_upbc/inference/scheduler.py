import heapq
import itertools
from typing import List, Optional

from ap_upbc.core.evidence import Evidence


class DynamicHeapMapper:
    """Search frontier를 max-heap에 매핑해 O(log n) 질의 선택을 지원한다."""

    def __init__(self):
        self.heap = []
        self._counter = itertools.count()  # 동일 utility는 삽입 순서로 처리

    def push_frontier(self, evidence: Evidence) -> None:
        # (-utility, seq, evidence)로 max-heap 구성
        heapq.heappush(self.heap, (-evidence.utility, next(self._counter), evidence))

    def pop_highest_utility(self) -> Optional[Evidence]:
        if not self.heap:
            return None
        _, _, evidence = heapq.heappop(self.heap)
        return evidence

    def get_remaining_evidence(self) -> List[Evidence]:
        return [item[2] for item in self.heap]

    def is_empty(self) -> bool:
        return len(self.heap) == 0

    def __len__(self) -> int:
        return len(self.heap)
