import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from ap_upbc.core.evidence import Evidence
from ap_upbc.inference.engine import APUPBCEngine


def build_knowledge_base():
    # (이름, LS, LN) - LS가 크거나 LN이 0에 가까울수록 utility가 높다
    return [
        Evidence("환자의 고열 여부", ls=5.0, ln=0.8),
        Evidence("피부 발진 여부", ls=1.5, ln=0.9),
        Evidence("최근 해외 여행력", ls=8.0, ln=0.5),
        Evidence("근육통 여부", ls=1.2, ln=0.95),
        Evidence("특정 바이러스 항체", ls=0.5, ln=0.1),
    ]


def interactive_answer(evidence):
    while True:
        answer = input(f"[{evidence.name}] 해당 증거가 확인됩니까? (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def simulated_answer(evidence):
    return evidence.name != "피부 발진 여부"


def main():
    engine = APUPBCEngine(prior_odds=1.0, tau=15.0)
    engine.load_frontier(build_knowledge_base())

    if "--interactive" in sys.argv or "-i" in sys.argv:
        result = engine.run_inference(answer_provider=interactive_answer)
    else:
        result = engine.run_inference(answer_provider=simulated_answer)

    print(f"\nquery order: {' -> '.join(result.query_log)}")
    print(f"result: {result.status} (queries={result.queries_asked}, final odds={result.final_odds:.4f})")


if __name__ == "__main__":
    main()
