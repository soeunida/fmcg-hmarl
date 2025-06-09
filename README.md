# 📦 FMCG Supply Chain Optimization using H-MARL (Multi-Agent SAC)

본 프로젝트는 Fast-Moving Consumer Goods (FMCG) 산업의 다계층 공급망에서 재고 보급을 최적화하기 위해 **Multi-Agent Reinforcement Learning (MARL)**을 적용한 시뮬레이션 및 학습 프레임워크입니다. 특히 **Bullwhip Effect** 완화를 위해 정보 공유, 주문 평활화, 협력 보상 등의 요소가 포함된 개선 구조를 제안합니다.

---

## 📁 파일 구성

| 파일명                    | 설명 |
|---------------------------|------|
| `fmcg_hmarl_baseline.py`  | 기본 H-MARL (SAC) 기반 공급망 시뮬레이션 코드 |
| `fmcg_hmarl_enhanced.py`  | Bullwhip 완화 및 협력 보상 구조가 추가된 개선 버전 |

---

## 🚚 시뮬레이션 구성

- **공급망 계층:** Retail → RDC → Wholesaler → Manufacturer (4계층)
- **에이전트 수:** 4 (계층별 하나씩)
- **모델:** Soft Actor-Critic 기반의 Multi-Agent 구조
- **강화 요소:**
  - 정보 공유 (Information Sharing)
  - 주문 평활화 (Order Smoothing)
  - 협력 보상 (Collaboration Bonus)
  - Bullwhip Penalty
- **시나리오 설정:** normal / high volatility / supply disruption / seasonal

---

## 🧪 실행 방법

```bash
python3 fmcg_hmarl_baseline.py
python3 fmcg_hmarl_enhanced.py
