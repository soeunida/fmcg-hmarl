# [Multi-Agent Reinforcement Learning for FMCG Supply Chains]

본 프로젝트는 **Fast-Moving Consumer Goods (FMCG)** 산업의 **다계층 공급망**에서 재고 보급 정책을 최적화하기 위한 **Multi-Agent Reinforcement Learning (MARL)** 기반 시뮬레이션 및 학습 프레임워크입니다.

두 가지 구현 버전을 통해 **기초 H-MARL 구조**와 **Bullwhip Effect 완화를 위한 고도화된 구조**를 비교하며, 각 버전은 Soft Actor-Critic (SAC) 알고리즘을 중심으로 계층 간 협력 학습을 수행합니다.

---

## 파일 구성

| 파일명                    | 설명 |
|---------------------------|------|
| `fmcg_hmarl_baseline.py`  | 기본 H-MARL(SAC) 구조를 사용한 FMCG 공급망 시뮬레이터 및 학습 코드 |
| `fmcg_hmarl_enhanced.py`  | Bullwhip 완화 및 협력 보상을 포함한 개선된 H-MARL 구조 구현 |

---

## 공통 사항

- **공급망 구조**: 4계층 (Retail → RDC → Wholesaler → Manufacturer)
- **에이전트 수**: 각 계층별 하나씩 (총 4개)
- **알고리즘**: Soft Actor-Critic (SAC)
- **학습 방식**: Centralized Critic, Local Actor
- **환경 모델링**: 부분 관측 기반 `Dec-POMDP` 프레임워크

---

### 실행 방법
```bash
python3 fmcg_hmarl_baseline.py
python3 fmcg_hmarl_baseline.py

---

### 실험 결과 (Baseline: `fmcg_hmarl_baseline.py`)

| 기법                  | 평균 비용 (↓) | 표준편차 | 설명 |
|------------------------|---------------|----------|------|
| **H-MARL (제안기법)**  | **130,146**   | 4,618    | Dec-POMDP 기반 다중 에이전트 협력 SAC 방식 |
| Centralized SAC        | 456,305       | 23,369   | 전체 계층에 대해 단일 에이전트로 학습된 SAC 방식 |
| Rule-based             | 593,197       | 35,053   | 전통적인 재주문점 및 안전재고 기반 휴리스틱 정책 |

> **해석**: 제안한 H-MARL 기법은 Rule-based 및 중앙집중식 SAC 방식에 비해 총 비용을 약 **70~78% 절감**하며, **부분 관측성과 계층 분산성**을 효과적으로 처리함을 보여줍니다.

---
