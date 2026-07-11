from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ap_upbc.core.evidence import Evidence
from ap_upbc.inference.pruning import Pruner
from ap_upbc.inference.scheduler import DynamicHeapMapper

AnswerProvider = Callable[[Evidence], bool]


@dataclass
class InferenceResult:
    final_odds: float
    status: str  # ACCEPTED / PRUNED / EXHAUSTED
    queries_asked: int = 0
    query_log: List[str] = field(default_factory=list)


class APUPBCEngine:
    def __init__(self, prior_odds: float, tau: float, verbose: bool = True):
        if prior_odds <= 0:
            raise ValueError(f"prior_odds must be positive, got {prior_odds}")

        self.current_odds = prior_odds
        self.tau = tau
        self.verbose = verbose
        self.scheduler = DynamicHeapMapper()
        self.pruner = Pruner(threshold_tau=tau)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def load_frontier(self, evidences: List[Evidence]) -> None:
        for ev in evidences:
            self.scheduler.push_frontier(ev)

    def run_inference(self, answer_provider: Optional[AnswerProvider] = None) -> InferenceResult:
        """answer_provider가 None이면 모든 증거를 확인된 것으로 간주한다."""
        if answer_provider is None:
            answer_provider = lambda ev: True

        result = InferenceResult(final_odds=self.current_odds, status="EXHAUSTED")

        self._log(f"--- inference start (prior odds: {self.current_odds:.2f}, tau: {self.tau}) ---")

        while not self.scheduler.is_empty():
            # 현재 odds가 이미 임계치를 넘으면 남은 질의 없이 채택
            if self.current_odds >= self.tau:
                self._log(f"\n[accept] odds {self.current_odds:.4f} >= tau {self.tau}")
                result.status = "ACCEPTED"
                break

            remaining = self.scheduler.get_remaining_evidence()
            o_max = self.pruner.calculate_omax(self.current_odds, remaining)

            self._log(f"\ncurrent odds: {self.current_odds:.4f} | O_max: {o_max:.4f}")

            # O_max < tau면 채택 불가능한 경로이므로 중단
            if self.pruner.should_prune(self.current_odds, remaining):
                self._log(f"[prune] O_max {o_max:.4f} < tau {self.tau}")
                result.status = "PRUNED"
                break

            target_ev = self.scheduler.pop_highest_utility()
            self._log(f"[query] {target_ev.name} (U={target_ev.utility:.2f})")

            confirmed = answer_provider(target_ev)
            result.queries_asked += 1
            result.query_log.append(target_ev.name)

            # 확인되면 LS, 부정되면 LN을 곱해 odds 갱신
            if confirmed:
                self._log(f"  -> confirmed, odds *= LS({target_ev.ls})")
                self.current_odds *= target_ev.ls
            else:
                self._log(f"  -> denied, odds *= LN({target_ev.ln})")
                self.current_odds *= target_ev.ln

        if result.status == "EXHAUSTED" and self.current_odds >= self.tau:
            result.status = "ACCEPTED"

        result.final_odds = self.current_odds
        self._log(f"\n--- inference end [{result.status}] queries={result.queries_asked}, final odds={self.current_odds:.4f} ---")
        return result
