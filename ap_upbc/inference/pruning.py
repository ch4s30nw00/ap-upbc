from typing import List

from ap_upbc.core.evidence import Evidence


class Pruner:
    def __init__(self, threshold_tau: float):
        self.threshold_tau = threshold_tau

    def calculate_omax(self, current_odds: float, remaining_evidence: List[Evidence]) -> float:
        # O_max = O_curr * Π max(LS_i, 1)
        o_max = current_odds
        for ev in remaining_evidence:
            o_max *= max(ev.ls, 1.0)
        return o_max

    def should_prune(self, current_odds: float, remaining_evidence: List[Evidence]) -> bool:
        return self.calculate_omax(current_odds, remaining_evidence) < self.threshold_tau
