# Designing a Framework for Real-Time Bio-signal Analysis and Personalized Learning Interventions

웨어러블 생체 신호를 실시간으로 수집·분석하고 맞춤형 학습 지도를 지원하기 위한 통합 데이터베이스 및 처리 시스템.

## 구현 범위

- FastAPI 기반 수집 API와 PostgreSQL 저장 계층
- 이질적인 센서 주기의 동기화와 세션 단위 처리
- 개인별 baseline normalization 및 개인 모델 학습
- Logistic Regression/XGBoost 비교와 하이퍼파라미터 튜닝
- 규칙·모델 예측을 이용한 실시간 중재 판단
- latency, throughput, data loss, classification metrics 평가

## Structure

```text
.
├── src/realtime_db/  # API, DB access, loading, simulation, intervention logic
├── scripts/          # collection, training, modeling, tuning, evaluation workflows
├── db/               # PostgreSQL schema and stage-specific SQL
├── docs/             # completed workflow notes
├── models/           # trained personal model artifact
├── results/          # aggregate evaluation outputs only
└── requirements.txt
```

## Data Description

| 항목 | 설명 |
|---|---|
| 생체 신호 | 심박수(HR), 심박 간격(IBI), 피부전기활동(EDA), 혈류량(BVP), 3축 가속도(ACC), 피부 온도(TEMP) |
| 데이터 단위 | 학습자별 학습 세션 및 세션 내 시간 순서 이벤트 |
| 처리 단위 | 센서별 timestamp 정렬, 시간창(window) 구성, 특징량 산출 |
| 주요 특징 | HR 평균·변동, HRV 지표, EDA 변동, ACC 기반 움직임, TEMP 변화 |
| 분석 목적 | 학습 상태 변화 분석, 개인별 기준선 산출, 실시간 학습 중재 판단 |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PostgreSQL 접속값은 환경변수로 관리. 실제 비밀번호나 운영 DB 주소는 저장소에 커밋하지 않음.

```bash
cp .env.example .env
# Edit .env, then export/load DATASET_ROOT and DATABASE_URL for your shell.
```

## Representative Workflow

```bash
python scripts/run_stage3_preprocess_baseline.py
python scripts/run_stage4_modeling.py
python scripts/run_stage5_tuning.py
python scripts/run_final_performance_evaluation.py
```

세부 수집·세션 부트스트랩·실시간 중재 실행은 `scripts/`와 `docs/`에 정리.

## Result Snapshot

- Stage 5 held-out test: accuracy 0.5557, macro-F1 0.4406, macro one-vs-rest AUC 0.6867
- Runtime benchmark: 120 iterations, mean inference 117.2 ms, p95 189.3 ms
- Database batch test: 21/21 sessions met the recorded latency, throughput, and data-loss targets
- Trained artifact: `models/personal_model_batch1_v20260527_205613.pkl`

위 수치는 제한된 연구 데이터와 특정 실험 설정에서 얻은 결과이며 임상적·교육적 효능을 의미하지 않음.
