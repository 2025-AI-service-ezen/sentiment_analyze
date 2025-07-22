## 프로젝트 1: 감성 분석 모델 구현 및 학습 - 워크플로우 계획

**문서 버전:** 1.0
**작성일:** 2025년 7월 21일
**작성자:** Gemini

### 1. 프로젝트 개요

본 문서는 감성 분석 앙상블 모델 프로젝트의 첫 번째 하위 프로젝트인 "감성 분석 모델 구현 및 학습"에 대한 워크플로우 계획을 정의합니다. 이 프로젝트는 초기 데이터 소스를 정제하고, 이를 기반으로 4가지 감성 분석 모델의 베이스라인을 구축하며, 모든 실험 과정을 체계적으로 관리하는 것을 목표로 합니다.

### 2. 워크플로우 단계

#### Phase 1: 설정 및 데이터 수집 (Setup and Data Ingestion)

*   **목표:** 개발 환경을 설정하고 원시 리뷰 데이터를 데이터베이스에 통합합니다.
*   **작업:**
    *   개발 환경 설정 (Python, 필요한 라이브러리, SQLite).
    *   원시 리뷰 데이터 (`y_review.csv`, `naver_shopping.txt` 등)를 SQLite `raw_reviews` 테이블로 통합합니다.
*   **예상 산출물:** 설정된 개발 환경, `sentiment_analysis.db` 파일에 통합된 원시 데이터.
*   **주요 도구:** `run_shell_command` (환경 설정), Python 스크립트 (Pandas, SQLite3).

#### Phase 2: 데이터 전처리 및 데이터셋 생성 (Data Preprocessing and Dataset Creation)

*   **목표:** 원시 리뷰 데이터를 문장 단위로 분리하고, 초기 모델 학습을 위한 균형 잡힌 데이터셋을 생성합니다.
*   **작업:**
    *   KSS를 사용하여 `raw_reviews` 테이블의 리뷰를 문장으로 분리하고 임시 라벨과 함께 `sentences` 테이블에 저장합니다.
    *   `sentences` 테이블에서 평점 1과 5의 데이터만 사용하여 최대 2만개(각 라벨당 최대 1만개)의 균형 잡힌 초기 데이터셋 (`v1.0-initial-20k.csv`)을 생성합니다.
    *   DVC를 사용하여 생성된 데이터셋을 버전 관리합니다.
*   **예상 산출물:** `sentences` 테이블에 저장된 문장 데이터, DVC로 버전 관리되는 `v1.0-initial-20k.csv` 파일.
*   **주요 도구:** Python 스크립트 (KSS, Pandas, Scikit-learn), `run_shell_command` (DVC).

#### Phase 3: 기준 모델 훈련 (Baseline Model Training)

*   **목표:** 4가지 감성 분석 모델의 베이스라인 성능을 확보합니다.
*   **작업:**
    *   Word Sentiment 모델을 훈련하고 감성 사전을 저장합니다.
    *   LSTM 모델을 훈련하고 모델 파일 및 토크나이저를 저장합니다.
    *   BiLSTM 모델을 훈련하고 모델 파일 및 토크나이저를 저장합니다.
    *   BERT 모델을 파인튜닝하고 모델 파일 및 토크나이저를 저장합니다.
*   **예상 산출물:** 학습된 모델 파일 (Word Sentiment 사전, LSTM/BiLSTM/BERT 모델), 각 모델별 토크나이저.
*   **주요 도구:** Python 스크립트 (Scikit-learn, TensorFlow, PyTorch, Hugging Face Transformers).

#### Phase 4: 실험 관리 및 보고 (Experiment Management and Reporting)

*   **목표:** 모든 모델 학습 실험을 체계적으로 관리하고 결과를 보고합니다.
*   **작업:**
    *   MLflow를 사용하여 각 모델에 대한 모든 실험 매개변수, 메트릭, 아티팩트(모델 파일, 토크나이저, 감성 사전, 평가 결과)를 추적합니다.
    *   훈련된 모든 모델에 대한 기준 성능 비교 보고서를 생성합니다.
*   **예상 산출물:** MLflow 실험 로그, 베이스라인 성능 비교 보고서 (Markdown).
*   **주요 도구:** Python 스크립트 (MLflow API), Markdown. 
