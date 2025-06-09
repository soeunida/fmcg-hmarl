# 🧠 Multi-Agent Reinforcement Learning for FMCG Supply Chains

본 프로젝트는 **Fast-Moving Consumer Goods (FMCG)** 산업의 **4계층 공급망**에서 효율적인 재고 보급 정책을 학습하기 위한 **Multi-Agent Reinforcement Learning (MARL)** 프레임워크를 구현합니다.  
**Soft Actor-Critic (SAC)** 알고리즘 기반의 다중 에이전트들이 **Decentralized POMDP (Dec-POMDP)** 환경에서 협력적으로 학습하며, **Bullwhip Effect**를 최소화하고 **총 비용을 줄이는 것**을 목표로 합니다.

---

## 🗂️ 파일 구성

| 파일명                     | 설명 |
|----------------------------|------|
| `fmcg_hmarl_baseline.py`   | 기본 H-MARL 구조 구현. 각 계층 에이전트가 SAC로 학습하며 중앙 비평가를 사용함. |
| `fmcg_hmarl_enhanced.py`   | Bullwhip 완화, 협력 보상, 수요 예측 개선 등을 포함한 향상된 구조. 성능 향상 목적. |

---

## 🏗️ 공급망 환경 구성

- **계층 구성**: 4계층 공급망  
  - Retail → Regional DC (RDC) → Wholesaler → Manufacturer
- **에이전트 수**: 4개 (계층별 1개씩)
- **관측 및 제어**: 각 에이전트는 자신의 재고 상태만 관측 가능 (partial observability)
- **학습 방식**:  
  - **Actor**: 개별 에이전트 단위 학습  
  - **Critic**: centralized critic 기반 공동 업데이트

---

## ⚙️ 실행 방법

```bash
# 기본 구조 실행 (baseline)
python3 fmcg_hmarl_baseline.py

# 개선 구조 실행 (bullwhip 완화 등)
python3 fmcg_hmarl_enhanced.py
```

옵션 및 하이퍼파라미터는 각 파일 상단의 `Config` 클래스를 수정하여 조절할 수 있습니다.

---

## 📊 실험 결과 비교

| 기법                     | 평균 비용 (↓) | 표준편차    | 특징 |
|--------------------------|---------------|-------------|------|
| **H-MARL (Baseline)**    | **130,146**   | ±4,618      | SAC 기반 계층 분산 협력 학습 |
| Centralized SAC          | 456,305       | ±23,369     | 단일 중앙 에이전트 방식 |
| Rule-based               | 593,197       | ±35,053     | 재주문점 + 안전재고 휴리스틱 |

> 📌 **H-MARL 기법은 rule-based 및 centralized 방식 대비 평균 비용을 약 70~78% 절감**하며, 분산적 의사결정의 효율성을 입증함.

---

## 🔬 시나리오 기반 평가 (`fmcg_hmarl_enhanced.py` 기준)

| 시나리오          | 평균 보상   | 평균 비용      | 서비스 레벨 | Bullwhip 비율 |
|-------------------|--------------|----------------|--------------|----------------|
| Normal            | -12,658.80   | 2,250,691.38   | 0.949        | 0.945          |
| High Volatility   | -12,781.59   | 2,291,353.48   | 0.949        | **0.334**      |
| Supply Disruption | -11,678.44   | 2,065,576.18   | 0.942        | 0.906          |

> 🔍 **Bullwhip 완화 효과는 특히 수요 변동성(High Volatility) 시나리오에서 두드러짐**, 비용 효율성과 안정적 재고 공급이 가능함을 확인.

---

## 📌 주요 기법 요약

- ✅ **Soft Actor-Critic 기반 협력적 MARL 구조**
- ✅ **Centralized Training, Decentralized Execution (CTDE)**
- ✅ **Bullwhip 완화를 위한 reward shaping 및 forecasting 개선 (enhanced.py)**
- ✅ **Dec-POMDP 구조에서의 효율적인 에이전트 학습**
- ✅ **성능 비교, 시나리오 실험, 비용/서비스 레벨/Bullwhip 지표 분석 포함**

---

