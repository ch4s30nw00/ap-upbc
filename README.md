# AP-UPBC: Adaptive Pruning and Utility-Prioritization Backward Chaining

> **Prioritize to Prune: Utility-Guided Search Control for Efficient Backward Chaining Inference**

전통적인 후방향 추론(backward chaining) 엔진의 정적 LIFO 탐색 구조를 개선한 추론 프레임워크의 Python 구현입니다. 논리적 추론 계층과 물리적 탐색 스케줄링 계층을 분리하여, 가장 결정적인 증거부터 질의하고 가망 없는 추론 경로를 조기에 차단합니다.

## 배경: 기존 방법론의 문제

전통적인 전문가 시스템의 추론 엔진은 가설을 최상위 목표로 놓고, 규칙에서 추출한 하위 증거들을 스택에 쌓아 후방향으로 탐색합니다. 이 구조에는 세 가지 근본적인 문제가 있습니다.

**1. 탐색 순서가 지식의 중요도가 아니라 규칙의 물리적 입력 순서로 결정됩니다.**
스택의 LIFO 특성상 어떤 증거를 먼저 물어볼지는 지식베이스에 규칙이 저장된 순서가 정합니다. Davis(1980)는 어떤 규칙을 먼저 적용할지 판단하는 제어 지식(control knowledge)이 없는 이런 맹목적 탐색(blind search)이 대규모 지식베이스에서 심각한 비효율을 낳는다고 지적했습니다.

**2. 모든 증거가 동등한 무게로 취급됩니다.**
실제로는 증거마다 결정력이 다릅니다. 우도비 LS=8.0인 증거 하나가 LS=1.2인 증거 여러 개보다 가설 판정에 훨씬 결정적입니다. 그러나 기존 엔진은 이런 지표(LS, LN)를 탐색 순서에 반영하지 못하므로, 사용자는 결론에 거의 기여하지 않는 질문에까지 일일이 답해야 합니다. Buchanan & Shortliffe(1984)는 이것이 실시간 진단 환경에서 불필요한 사용자 피로(query fatigue)로 이어진다고 보고했습니다.

**3. 가망 없는 가설도 끝까지 탐색합니다.**
남은 증거가 전부 확인되어도 가설이 채택될 수 없는 상황, 즉 수학적으로 이미 결론이 난 상황에서도 기존 엔진은 이를 감지하지 못하고 남은 질의를 모두 수행합니다.

AP-UPBC는 이 세 문제를 각각 다음과 같이 해결합니다.

| 기존 방식의 문제 | AP-UPBC의 해결 |
|---|---|
| 규칙 입력 순서에 의존하는 맹목적 LIFO 탐색 | Utility Index `U(E)`로 증거의 정보 가치를 정량화하고, 스택 대신 우선순위 큐로 결정적인 증거부터 질의 |
| 비정형 추론 트리에 우선순위 큐를 적용하기 어려움 | Dynamic Heap-Mapping: search frontier만 추출해 Max-Heap에 매핑, O(log n) 질의 선택 보장 |
| 채택 불가능한 가설도 끝까지 질의 | Adaptive Pruning: 도달 가능한 최대 odds `O_max`를 실시간 계산해 임계치 미달 시 즉시 중단 |

## 핵심 아이디어

### 1. Utility Index
Duda et al. (1976)의 우도비(LS, LN)를 재해석하여 각 증거의 정보 가치를 정량화합니다. 우도비가 1.0에서 멀수록 결정적인 증거입니다.

```
U(E) = max(|LS − 1|, |1 − LN|)
```

### 2. Dynamic Heap-Mapping
규칙 기반 추론 트리는 비정형·비이진 구조라서 Complete Binary Tree를 요구하는 Priority Queue와 구조적으로 맞지 않습니다. AP-UPBC는 지식베이스 전체를 재구성하는 대신, 현재 활성화된 search frontier만 추출하여 Max-Heap에 실시간 매핑합니다. 이로써 규칙 구조와 무관하게 가장 결정적인 질의가 항상 루트에 위치하며, O(log n) 복잡도가 보장됩니다.

### 3. Adaptive Pruning
남은 모든 증거가 최대로 기여했을 때 도달 가능한 Maximum Potential Odds를 실시간 계산합니다.

```
O_max = O_curr × Π max(LS_i, 1)   (i ∈ 남은 증거)
```

`O_max < τ` (채택 임계치)이면 해당 가설은 수학적으로 채택이 불가능하므로 즉시 추론을 중단해 불필요한 질의와 연산을 차단합니다. 반대로 `O_curr ≥ τ`이면 남은 질의 없이 가설을 조기 채택하여 사용자 피로(query fatigue)를 최소화합니다.

## 프로젝트 구조

```
ap_upbc/
├── ap_upbc/
│   ├── core/
│   │   └── evidence.py      # Evidence 클래스, Utility Index U(E) 계산
│   └── inference/
│       ├── scheduler.py     # DynamicHeapMapper: Search Frontier → Max-Heap 매핑
│       ├── pruning.py       # Pruner: O_max 계산 및 Adaptive Pruning 판정
│       └── engine.py        # APUPBCEngine: 추론 루프 (질의 → Odds 갱신 → Pruning)
├── main_test.py             # 의료 진단 시나리오 데모
└── tests/
    └── test_ap_upbc.py      # 단위 테스트
```

## 실행 방법

Python 3.8+ 만 있으면 됩니다 (외부 의존성 없음).

```bash
# 시뮬레이션 데모 실행
python main_test.py

# 직접 y/n으로 응답하는 인터랙티브 모드
python main_test.py --interactive

# 테스트 실행
python -m unittest discover tests -v
```

## 사용 예시

```python
from ap_upbc import Evidence, APUPBCEngine

evidences = [
    Evidence("환자의 고열 여부", ls=5.0, ln=0.8),   # U = 4.0
    Evidence("최근 해외 여행력", ls=8.0, ln=0.5),   # U = 7.0 → 가장 먼저 질의됨
    Evidence("근육통 여부", ls=1.2, ln=0.95),      # U = 0.2
]

engine = APUPBCEngine(prior_odds=1.0, tau=15.0)
engine.load_frontier(evidences)

result = engine.run_inference()  # answer_provider 콜백으로 응답 주입 가능
print(result.status, result.final_odds)  # 예: ACCEPTED 40.0
```

증거가 확인되면 `LS`를, 부정되면 `LN`을 곱해 Odds를 갱신합니다 (Bayesian odds-likelihood 갱신).

## 한계 및 향후 과제

- Search Frontier의 잦은 재매핑에 따른 메모리·처리 오버헤드가 활성 sub-goal 수에 비례해 증가할 수 있음
- 전문가가 제공하는 LS/LN 파라미터의 주관적 정확도와 증거 간 조건부 독립 가정에 의존
- 향후: 비이진 분기 간 상관관계를 반영하는 동적 가중치 조정, 질의별 비용 함수 통합

## References

- Davis, R. *Meta-rules: Reasoning about control.* Artificial Intelligence, 15(3):179–222, 1980.
- Buchanan, B. G. and Shortliffe, E. H. *Rule-Based Expert Systems: The MYCIN Experiments.* Addison-Wesley, 1984.
- Duda, R. O., Hart, P. E., and Nilsson, N. J. *Subjective Bayesian methods for rule-based inference systems.* NCC, 1976.
- Pearl, J. *Heuristics: Intelligent Search Strategies for Computer Problem Solving.* 1983.
