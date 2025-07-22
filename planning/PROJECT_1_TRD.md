## 프로젝트 1: 감성 분석 모델 구현 및 학습 - 기술 요구사항 정의서 (TRD)

**문서 버전:** 1.0
**작성일:** 2025년 7월 18일
**작성자:** Gemini

### 1. 서론

본 문서는 감성 분석 앙상블 모델 프로젝트의 첫 번째 하위 프로젝트인 "감성 분석 모델 구현 및 학습"에 대한 기술 요구사항을 정의합니다. 이 프로젝트는 원본 데이터의 전처리부터 4가지 감성 분석 모델의 베이스라인 학습 및 평가에 이르는 기술적 구현 세부 사항을 다룹니다. [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md), [`PROJECT_1_DATA_PIPELINE_DESIGN.md`](PROJECT_1_DATA_PIPELINE_DESIGN.md), [`PROJECT_1_TECHNICAL_SPECIFICATION.md`](PROJECT_1_TECHNICAL_SPECIFICATION.md)의 내용을 기반으로 작성되었습니다.

### 2. 시스템 아키텍처 및 데이터 흐름

```mermaid
graph TD
    A[Raw Data Sources<br>- 숙박업소 리뷰 데이터 (예: y_review.csv, naver_shopping.txt)] --> B{1. Data Integration &<br>Sentence Splitting (KSS)};
    B --> C[SQLite Database<br>- raw_reviews<br>- sentences];
    C --> D{2. Balanced Dataset Creation<br>(200k Undersampling)};
    D --> E(v1.0-balanced-200k.csv);
    E -- DVC for Versioning --> F{3. Model Training Pipeline};
    F --> G1[Word Sentiment Model];
    F --> G2[LSTM Model (TensorFlow)];
    F --> G3[BiLSTM Model (PyTorch)];
    F --> G4[BERT Model (Transformers)];
    
    subgraph MLflow Tracking
        G1 --> H{Experiment Tracking};
        G2 --> H;
        G3 --> H;
        G4 --> H;
    end

    H -- Log Artifacts --> I[Model Storage<br>- .pickle<br>- .h5<br>- .pt<br>- Hugging Face format];
    H -- Log Metrics --> J[Evaluation Reports];

```

**설명:**

*   **데이터 수집 및 전처리:** 숙박업소 리뷰 원본 데이터 소스(`y_review.csv`, `naver_shopping.txt` 등)는 KSS를 활용한 문장 분리 및 임시 라벨링 과정을 거쳐 SQLite 데이터베이스(`raw_reviews`, `sentences` 테이블)에 저장됩니다.
*   **데이터셋 생성:** `sentences` 테이블에서 긍정/부정 라벨의 균형을 맞춘 20만개 규모의 데이터셋(`v1.0-balanced-200k.csv`)이 생성되며, 이는 DVC를 통해 버전 관리됩니다.
*   **모델 학습:** 생성된 데이터셋을 기반으로 Word Sentiment, LSTM, BiLSTM, BERT 4가지 모델이 각각 학습됩니다.
*   **실험 관리:** 모든 모델 학습 과정은 MLflow를 통해 추적되며, 학습된 모델 아티팩트와 성능 지표가 기록됩니다.

### 3. 기술 사양

#### 3.1. 데이터 파이프라인 구현

*   **데이터 통합 및 DB 저장:**
    *   **입력:** `y_reviews.csv`, `naver_shopping.txt`
    *   **도구:** Pandas, SQLite3
    *   **최적화:** Pandas의 벡터화 연산 및 SQLite의 `executemany()`를 활용한 벌크 삽입을 통해 대용량 데이터 처리 성능을 최적화합니다.
    *   **라벨 0 처리:** 원본 평점 0은 3(중립)으로 처리하며, `is_temporary_label` 플래그를 `True`로 설정하여 향후 정제 과정에서 우선 검토될 수 있도록 합니다.
*   **문장 분리 및 임시 라벨링:**
    *   **도구:** KSS (Korean Sentence Splitter)
    *   **로직:** `raw_reviews` 테이블의 `review_text`를 문장 단위로 분리하고, 원본 `original_label`을 `sentences` 테이블의 `current_label`로 임시 할당합니다. `is_temporary_label`은 `True`로 설정합니다.
*   **균형 잡힌 학습 데이터셋 생성:**
    *   **도구:** Pandas, Scikit-learn (`train_test_split` 또는 `sample`)
    *   **로직:** `sentences` 테이블에서 평점 1과 5의 데이터만 사용하여 최대 2만개(각 라벨당 최대 1만개)의 균형 잡힌 데이터셋을 생성한다. 이 데이터셋은 `v1.0-initial-20k.csv`로 명명된다.
    *   **버전 관리:** 생성된 `v1.0-initial-20k.csv` 파일은 DVC를 통해 버전을 관리합니다.

#### 3.2. 모델별 구현 사양

모든 모델은 `v1.0-initial-20k.csv` 데이터셋을 기반으로 학습 및 평가됩니다.

*   **Word Sentiment 모델:**
    *   **프레임워크:** Scikit-learn, Pandas
    *   **전처리:** KoNLPy(Mecab)를 이용한 형태소 분석 및 명사/동사/형용사 추출, 단어별 긍정/부정 리뷰 빈도 및 TF-IDF 가중치 계산.
    *   **산출물:** `models/word_sentiment/word_sentiment_dict.pickle`
*   **LSTM 모델:**
    *   **프레임워크:** TensorFlow, Keras
    *   **전처리:** `tf.keras.preprocessing.text.Tokenizer`를 사용한 정수 인코딩 및 `pad_sequences`를 통한 패딩.
    *   **아키텍처:** `Embedding` -> `LSTM` -> `Dense` (sigmoid 활성화 함수).
    *   **산출물:** `models/lstm/best_model.h5`, `models/tokenizers/word_level_tokenizer.pickle`
*   **BiLSTM 모델:**
    *   **프레임워크:** PyTorch
    *   **전처리:** LSTM과 동일한 토크나이저 및 패딩 방식 사용.
    *   **아키텍처:** `nn.Embedding` -> `nn.LSTM` (bidirectional) -> `nn.Linear`.
    *   **산출물:** `models/bilstm/best_model.pt`, `models/tokenizers/word_level_tokenizer.pickle`
*   **BERT 모델:**
    *   **프레임워크:** Hugging Face Transformers (TensorFlow/PyTorch 백엔드)
    *   **전처리:** `BertTokenizer` (`klue/bert-base`)를 사용한 토큰화 및 입력 형식 인코딩.
    *   **아키텍처:** `TFBertForSequenceClassification.from_pretrained('klue/bert-base')`를 활용한 파인튜닝.
    *   **산출물:** `models/bert/` 디렉토리 (Hugging Face 형식 모델 파일 및 BERT 전용 토크나이저)

#### 3.3. 학습 및 평가 파이프라인

*   **통합 스크립트:** `train_all_models.py`를 통해 4개 모델의 학습 및 평가를 순차적으로 실행합니다.
*   **실험 추적 (MLflow):**
    *   각 모델의 하이퍼파라미터, 학습/검증 메트릭(loss, accuracy 등), 최종 평가 메트릭(accuracy, precision, recall, f1-score)을 기록합니다.
    *   학습된 모델 파일, 각 모델에 특화된 토크나이저, 감성 사전, 평가 결과(Confusion Matrix 이미지, Classification Report 텍스트 파일)를 아티팩트로 저장합니다.
*   **평가:** `v1.0-initial-20k.csv` 데이터셋을 훈련/검증/테스트 세트로 분할(예: 8:1:1)하여, 모든 모델이 동일한 테스트 세트로 성능을 평가받도록 합니다.

### 4. 기술 스택

*   **언어:** Python 3.9+
*   **데이터 처리:** Pandas, NumPy, KSS, KoNLPy(Mecab)
*   **머신러닝/딥러닝:** Scikit-learn, TensorFlow, PyTorch, Hugging Face Transformers
*   **데이터베이스:** SQLite3
*   **실험 관리:** MLflow
*   **데이터 버전 관리:** DVC

### 5. 개발 환경 요구사항

*   **CPU:** 4코어 8스레드 이상 (멀티프로세싱 활용)
*   **RAM:** 16GB 이상 (대용량 데이터 처리 및 모델 로딩)
*   **GPU:** NVIDIA GPU (GTX 960, GTX 1070, GTX 1080Ti 등 CUDA 지원 모델)
    *   **VRAM:** 최소 8GB 권장 (BERT 학습 시)
*   **운영체제:** Ubuntu 20.04+ 또는 호환 가능한 Linux 배포판
*   **소프트웨어:** Conda (가상 환경 관리), CUDA Toolkit, cuDNN (GPU 가속)

### 6. 테스트 전략

*   **단위 테스트:** 각 데이터 처리 함수, 모델 아키텍처 구성 요소, 유틸리티 함수에 대한 단위 테스트를 작성합니다.
*   **통합 테스트:** 데이터 파이프라인의 각 단계(통합, 문장 분리, 데이터셋 생성)가 올바르게 연동되는지 확인하는 통합 테스트를 수행합니다.
*   **성능 테스트:** 생성된 데이터셋으로 각 모델이 정상적으로 학습되고 평가되는지 확인합니다.
*   **재현성 테스트:** DVC 태그를 사용하여 특정 데이터셋 버전으로 롤백한 후 모델을 재학습하여 동일한 결과가 나오는지 검증합니다.

### 7. 리스크 및 완화 전략

*   **데이터 처리 성능 병목:**
    *   **리스크:** 대용량 데이터 처리 시 `for` 루프 사용으로 인한 성능 저하.
    *   **완화:** Pandas 벡터화 연산 및 SQLite `executemany()`를 통한 벌크 삽입 적용.
*   **GPU 자원 부족:**
    *   **리스크:** BERT와 같은 대규모 모델 학습 시 GPU 메모리 부족(OOM).
    *   **완화:** 동적 배치 크기 조정, `torch.cuda.empty_cache()`, `tf.keras.backend.clear_session()`을 통한 메모리 관리, MLflow를 통한 학습 시간 추적 및 조기 종료 활용.
*   **하드웨어 호환성:**
    *   **리스크:** 다양한 GPU 모델(GTX 960, 1070, 1080Ti) 간의 CUDA/cuDNN 버전 호환성 문제.
    *   **완화:** `cuda_gpu_setup_guide.md` 및 `cuda_system_analysis.md` 문서를 참조하여 환경 설정 및 테스트를 철저히 수행하고, 필요한 경우 환경별 설정 스크립트를 제공합니다.

### 8. 배포 고려사항

*   프로젝트 1의 최종 산출물(학습된 모델, 토크나이저, 감성 사전, 버전 관리된 데이터셋)은 프로젝트 2(데이터 정제 및 재훈련) 및 프로젝트 3(API 서비스)의 입력으로 사용됩니다.
*   MLflow 모델 레지스트리에 모델을 등록하여 향후 모델 배포 및 버전 관리를 용이하게 합니다.
