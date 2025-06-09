# 🧠 Multi-Agent Reinforcement Learning for FMCG Supply Chains

본 프로젝트는 **Fast-Moving Consumer Goods (FMCG)** 산업의 **다계층 공급망**에서 재고 보급 정책을 최적화하기 위한 **Multi-Agent Reinforcement Learning (MARL)** 기반 시뮬레이션 및 학습 프레임워크입니다.

두 가지 구현 버전을 통해 **기초 H-MARL 구조**와 **Bullwhip Effect 완화를 위한 고도화된 구조**를 비교하며, 각 버전은 Soft Actor-Critic (SAC) 알고리즘을 중심으로 계층 간 협력 학습을 수행합니다.

---

## 📁 파일 구성

| 파일명                    | 설명 |
|---------------------------|------|
| `fmcg_hmarl_baseline.py`  | 기본 H-MARL(SAC) 구조를 사용한 FMCG 공급망 시뮬레이터 및 학습 코드 |
| `fmcg_hmarl_enhanced.py`  | Bullwhip 완화 및 협력 보상을 포함한 개선된 H-MARL 구조 구현 |

---

## 🧩 공통 사항

- **공급망 구조**: 4계층 (Retail → RDC → Wholesaler → Manufacturer)
- **에이전트 수**: 각 계층별 하나씩 (총 4개)
- **알고리즘**: Soft Actor-Critic (SAC)
- **학습 방식**: Centralized Critic, Local Actor
- **환경 모델링**: 부분 관측 기반 `Dec-POMDP` 프레임워크

---

## 📌 `fmcg_hmarl_baseline.py`

### 구성 요약
- 기본 SAC 기반 H-MARL 구조
- 간단한 수요 모델링과 재고 관리
- 글로벌 보상 기반 에이전트 학습
- 정보 공유 및 bullwhip 제어 없음

### 특징
- 빠르고 단순한 실험 구조
- Bullwhip Effect가 학습 중에 자연스럽게 발생
- 비교군으로 사용 가능

### 실행 방법
```bash
python3 fmcg_hmarl_baseline.py
python3 fmcg_hmarl_baseline.py
