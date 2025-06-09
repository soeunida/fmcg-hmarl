# fmcg-hmarl
# 🧠 FMCG Supply Chain Optimization via Cooperative H-MARL (Dec-POMDP + SAC)

본 프로젝트는 Fast-Moving Consumer Goods (FMCG) 산업의 **4계층 공급망**을 대상으로, Hierarchical Multi-Agent Reinforcement Learning (H-MARL)을 이용한 재고 보급 최적화 시스템을 구현하고 평가합니다. 문제는 **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)** 형태로 모델링되었으며, 각 에이전트는 SAC 기반의 연속 행동 공간을 학습합니다.

---

## 🎯 목표 (Objective)
- **Dec-POMDP 기반 강화학습**을 이용하여, 각 계층 에이전트들이 **부분 정보** 하에서도 협력적으로 최적 보급 정책을 학습하도록 구성
- 기존 **Rule-based 정책** 및 **Centralized SAC**에 비해 **비용 효율성과 안정성**을 동시에 달성

---

## 🔍 환경 구성 (Simulation Setup)
- **공급망 계층:** Retailer → RDC → Wholesaler → Manufacturer (총 4계층)
- **에이전트 수:** 4 (계층별 1개)
- **상태 차원:** 재고, 예측 수요, 리드타임, 주문 비용, 보관 비용, 예산 등 7차원
- **행동 공간:** 연속적인 주문량 결정 (SAC 기반)
- **보상 함수:** 전체 시스템 비용을 음수 보상으로 정의

---

## 🧪 실험 결과 요약 (Experimental Results)

| 방법             | 평균 비용 | 비용 표준편차 | 설명                           |
|------------------|-----------|----------------|--------------------------------|
| H-MARL (제안)    | **130,146** | ± 4,618       | Dec-POMDP + Cooperative SAC    |
| Centralized SAC  | 456,305   | ± 23,369      | Oracle 기반 완전 관측          |
| Rule-based       | 593,197   | ± 35,053      | Safety stock + Reorder policy |

### ✅ 성능 개선 효과
- Rule-based 대비 **78.1% 비용 절감**
- Centralized SAC 대비 **71.5% 비용 절감**
- Bullwhip Effect 완화 및 서비스 수준 향상

---

## 📌 기술적 특징 (Key Features)
- ✅ **Dec-POMDP 기반 공급망 모델링**
- ✅ **부분 관측성 하에서의 협력적 의사결정**
- ✅ **SAC 기반 연속 행동 공간 학습**
- ✅ **Bullwhip Effect 감소**
- ✅ **실시간 수요 변동성 대응**

---

## 💡 실무 적용 권장사항
- 📦 **단계적 도입:** 파일럿 테스트 → 점진적 확장
- 📈 **데이터 품질 확보:** 수요 예측 정확도 향상 필요
- 🏗️ **IT 인프라 구축:** 실시간 강화학습 연계
- 👩‍🏫 **운영진 교육:** AI 기반 SCM 툴 활용 교육

---

## 🧰 실행 방법 (Run)

```bash
python3 Total Supply Chain Cost Comparison.py
