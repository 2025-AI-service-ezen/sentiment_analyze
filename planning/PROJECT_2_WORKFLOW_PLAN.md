## 프로젝트 2: 점진적 자기지도학습 기반 데이터셋 정제 및 모델 재훈련 - 워크플로우 계획

**문서 버전:** 1.0
**작성일:** 2025년 7월 21일
**작성자:** Gemini

### 1. 프로젝트 개요

본 문서는 감성 분석 앙상블 모델 프로젝트의 두 번째 하위 프로젝트인 "점진적 자기지도학습 기반 데이터셋 정제 및 모델 재훈련"에 대한 워크플로우 계획을 정의합니다. 이 프로젝트는 프로젝트 1에서 구축된 베이스라인 모델들을 활용하여 데이터셋의 라벨을 지속적으로 개선하고, 이를 통해 모델 성능을 점진적으로 향상시키는 자기지도학습(Self-training) 기반의 MLOps 파이프라인을 구축하는 것을 목표로 합니다.

### 2. 워크플로우 단계

#### Phase 1: 데이터 로딩 및 앙상블 예측 (Data Loading and Ensemble Prediction)

*   **목표:** SQLite 데이터베이스에 저장된 모든 문장 데이터를 효율적으로 로드하고, 프로젝트 1에서 학습된 모델들을 활용하여 감성 예측을 수행합니다.
*   **작업:**
    *   SQLite `sentences` 테이블에서 모든 문장 데이터를 청크(Chunk) 단위로 로드합니다.
    *   프로젝트 1에서 학습된 4개 모델(Word Sentiment, LSTM, BiLSTM, BERT)을 활용하여 각 문장의 감성을 예측합니다.
    *   CPU 바운드 작업(예: 전처리, 일부 모델 예측)에 멀티프로세싱을 활용하고, BERT와 같은 GPU 기반 모델의 예측은 GPU에서 직접 수행합니다.
*   **예상 산출물:** 각 문장에 대한 개별 모델 예측 및 통합 앙상블 예측 결과.
*   **주요 도구:** Python 스크립트 (Pandas, SQLite3, multiprocessing), 프로젝트 1에서 학습된 모델들.

#### Phase 2: 신뢰도 기반 라벨 보정 및 Gemini API 통합 (Confidence-Based Label Correction and Gemini API Integration)

*   **목표:** 앙상블 예측의 신뢰도를 기반으로 데이터셋 라벨을 보정하고, Gemini API를 활용하여 특정 고가치 작업을 수행합니다.
*   **작업:**
    *   앙상블 예측 결과에 대한 신뢰도를 계산합니다.
    *   계산된 신뢰도가 일정 임계값 이상인 예측에 대해 SQLite `sentences` 테이블의 `current_label`을 업데이트합니다.
    *   **Gemini API (gemini-2.5-flash)를 활용하여 다음 작업을 수행합니다:**
        *   앙상블 모델의 신뢰도가 낮거나 예측이 충돌하는 샘플에 대한 재라벨링.
        *   앙상블 모델의 정성적 평가 (예: 엣지 케이스 또는 오분류된 샘플 분석).
        *   데이터셋 또는 모델 예측의 잠재적 편향 감지 및 완화.
*   **예상 산출물:** 라벨이 보정된 `sentences` 테이블, Gemini API를 통한 분석 결과 및 제안.
*   **주요 도구:** Python 스크립트, Gemini API, MLflow (LLM 사용량 추적).

#### Phase 3: 반복적 모델 재훈련 (Iterative Model Retraining)

*   **목표:** 정제된 고품질 데이터셋으로 각 감성 분석 모델을 반복적으로 재훈련하여 성능을 점진적으로 향상시킵니다.
*   **작업:**
    *   라벨이 보정된 `sentences` 테이블의 전체 데이터를 기반으로 각 모델의 재훈련을 위한 데이터셋을 준비합니다.
    *   Word Sentiment, LSTM, BiLSTM, BERT 모델을 `retraining_strategy_report.md`에 명시된 최적화된 전략과 순서에 따라 순차적으로 재훈련합니다.
    *   GPU 자원 활용을 최적화하고 OOM(Out Of Memory)을 방지하기 위한 로직을 구현합니다.
*   **예상 산출물:** 성능이 향상된 재훈련된 모델들.
*   **주요 도구:** Python 스크립트 (Scikit-learn, TensorFlow, PyTorch, Hugging Face Transformers), MLflow (재훈련 메트릭 추적).

#### Phase 4: 수렴 감지 및 MLOps 통합 (Convergence Detection and MLOps Integration)

*   **목표:** 자기지도학습 루프의 수렴 여부를 판단하고, MLOps 파이프라인을 통해 모델 및 데이터셋 버전을 체계적으로 관리합니다.
*   **작업:**
    *   라벨 변화율 및 감성 사전 변화율을 모니터링하여 자기지도학습 루프의 수렴을 감지합니다.
    *   확증 편향(Confirmation Bias) 완화를 위한 전략(예: 신뢰도 임계값 점진적 조정)을 구현합니다.
    *   MLflow Model Registry에 업데이트된 모델을 적절한 스테이지(Staging/Production)로 등록합니다.
    *   DVC를 사용하여 정제된 데이터셋의 버전을 관리합니다.
*   **예상 산출물:** 수렴 보고서, MLflow Model Registry에 등록된 최신 모델, DVC로 버전 관리되는 데이터셋.
*   **주요 도구:** Python 스크립트, MLflow API, DVC 명령.
