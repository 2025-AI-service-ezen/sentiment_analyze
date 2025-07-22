## 프로젝트 1: 감성 분석 모델 구현 및 학습 - 기술 기획서

**문서 버전:** 1.1
**작성일:** 2025년 7월 18일
**작성자:** Gemini

### 1. 프로젝트 목표

본 프로젝트는 4개의 독립적인 감성 분석 모델(Word Sentiment, LSTM, BiLSTM, BERT)의 베이스라인을 구축하는 것을 목표로 한다. 불안정한 초기 데이터 소스를 정제하고, DBMS 기반의 체계적인 데이터 파이프라인을 통해 생성된 **2만개의 균형 잡힌 데이터셋**을 사용하여 각 모델을 학습하고 성능을 평가한다. 이 과정의 모든 산출물(데이터셋, 모델, 성능 지표)은 재현 가능하도록 MLflow와 DVC를 통해 체계적으로 관리한다.

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

### 3. 데이터 파이프라인 및 관리

상세한 데이터베이스 스키마 및 처리 파이프라인 설계는 [`PROJECT_1_DATA_PIPELINE_DESIGN.md`](PROJECT_1_DATA_PIPELINE_DESIGN.md) 문서를 따른다. 주요 전략은 다음과 같다.

*   **데이터 통합:** 숙박업소 리뷰 데이터(예: `y_review.csv`와 `naver_shopping.txt`)를 SQLite `raw_reviews` 테이블로 통합한다.
*   **문장 단위 처리:** KSS 라이브러리를 사용하여 리뷰 문단을 문장으로 분리하고, 원본 라벨을 `is_temporary_label=True` 플래그와 함께 `sentences` 테이블에 저장한다.
*   **베이스라인 데이터셋 구축:** 평점 1과 5의 데이터만 사용하여 최대 2만개(각 라벨당 최대 1만개)의 균형 잡힌 데이터셋을 생성한다. 이 데이터셋은 `v1.0-initial-20k.csv`로 명명된다.
*   **데이터 버전 관리:** 생성된 `v1.0-initial-20k.csv` 파일은 DVC를 통해 버전을 관리하여, 모든 모델이 동일한 데이터로 학습 및 평가될 수 있도록 보장한다.

### 4. 모델별 구현 사양

모든 모델은 `v1.0-initial-20k.csv` 데이터셋을 기반으로 학습 및 평가된다.

#### 4.1. Word Sentiment 모델

*   **목표:** 단어와 감정 점수 간의 상관관계를 분석하여 해석 가능한 베이스라인 모델을 구축한다. 최종 감성 점수는 -1 (부정)에서 1 (긍정) 사이의 실수 값을 산출한다. **단어의 앞뒤 문맥(부정어, 강조어, N-그램 등)을 고려하여 감성 점수를 조정하고, 숙박업소 도메인에 특화된 감성 사전을 구축하거나 확장하여 도메인 적응성을 높인다.**
*   **프레임워크:** Scikit-learn, Pandas
*   **입력 데이터:** 문장 텍스트와 라벨
*   **전처리:**
    1.  KoNLPy(Mecab)를 이용한 형태소 분석 및 명사/동사/형용사 추출
    2.  단어별로 긍정/부정 리뷰에 등장하는 빈도 및 TF-IDF 가중치 계산
    3.  최종 감정 점수를 산출하여 `word_sentiment_dict.pickle` 파일로 저장
*   **산출물:** 감성 사전 파일 (`models/word_sentiment/word_sentiment_dict.pickle`)

#### 4.2. LSTM 모델

*   **목표:** 순차적인 데이터 특성을 학습하는 RNN 기반의 딥러닝 모델을 구축한다.
*   **프레임워크:** TensorFlow, Keras
*   **입력 데이터:** 정수 인코딩된 시퀀스 데이터
*   **전처리:**
    1.  `tf.keras.preprocessing.text.Tokenizer`를 사용하여 텍스트를 정수 시퀀스로 변환
    2.  `pad_sequences`를 사용하여 모든 시퀀스의 길이를 동일하게 맞춤
    3.  토크나이저 객체는 `tokenizer.pickle`로 저장
*   **아키텍처:**
    *   `Embedding` Layer (vocab_size, embedding_dim=128)
    *   `LSTM` Layer (units=128)
    *   `Dense` Layer (output, activation='sigmoid') -> 0 (부정) 또는 1 (긍정) 이진 분류
*   **산출물:** 학습된 모델 파일 (`models/lstm/best_model.h5`), 토크나이저 파일

#### 4.3. BiLSTM 모델

*   **목표:** 양방향 문맥 정보를 모두 활용하여 LSTM 모델의 성능을 개선한다.
*   **프레임워크:** PyTorch
*   **입력 데이터:** 정수 인코딩된 시퀀스 데이터
*   **전처리:** LSTM 모델과 동일한 토크나이저 및 패딩 방식 사용
*   **아키텍처:**
    *   `nn.Embedding` (vocab_size, embedding_dim=128)
    *   `nn.LSTM` (input_size, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
    *   `nn.Linear` (output) -> 0 (부정) 또는 1 (긍정) 이진 분류
*   **산출물:** 학습된 모델 파일 (`models/bilstm/best_model.pt`)

#### 4.4. BERT 모델

*   **목표:** 사전 학습된 대규모 언어 모델을 파인튜닝하여 최고 수준의 성능을 달성한다.
*   **프레임워크:** Hugging Face Transformers, TensorFlow/PyTorch
*   **입력 데이터:** BERT 전용 형식 (Input IDs, Attention Mask, Token Type IDs)
*   **전처리:**
    1.  `BertTokenizer` (`klue/bert-base`)를 사용하여 텍스트를 토큰화하고, 스페셜 토큰([CLS], [SEP])을 추가한다.
    2.  입력 형식에 맞게 인코딩한다.
*   **아키텍처:**
    *   `TFBertForSequenceClassification.from_pretrained('klue/bert-base')` 모델을 활용한 파인튜닝 -> 0 (부정) 또는 1 (긍정) 이진 분류
*   **산출물:** 학습된 모델 파일 (Hugging Face 형식, `models/bert/` 디렉토리)

### 5. 학습 및 평가 파이프라인

*   **통합 학습 스크립트:** `train_all_models.py` 스크립트를 통해 4개 모델의 학습 및 평가를 순차적으로 실행한다.
*   **실험 추적 (MLflow):**
    *   **Parameters:** `learning_rate`, `batch_size`, `epochs`, `embedding_dim`, `lstm_units` 등 각 모델의 주요 하이퍼파라미터 기록
    *   **Metrics:** `accuracy`, `precision`, `recall`, `f1-score`, `loss` 등 학습 및 검증 과정의 모든 성능 지표 기록
    *   **Artifacts:** 학습된 모델 파일, 토크나이저, 감성 사전, 평가 결과(Confusion Matrix 이미지, Classification Report 텍스트 파일)를 모두 아티팩트로 저장
*   **평가:** `v1.0-initial-20k.csv` 데이터셋을 훈련/검증/테스트 세트로 분할(예: 8:1:1)하여, 모든 모델이 동일한 테스트 세트로 성능을 평가받도록 한다. 이를 통해 공정한 성능 비교가 가능하다. 이를 통해 공정한 성능 비교가 가능하다.

### 6. 최종 산출물 (Deliverables)

1.  **데이터베이스:** 데이터 처리 파이프라인이 완료된 `sentiment_analysis.db` 파일
2.  **버전 관리된 데이터셋:** DVC로 관리되는 `data/processed/v1.0-initial-20k.csv`
3.  **학습된 모델 아티팩트:**
    *   `models/word_sentiment/word_sentiment_dict.pickle`
    *   `models/tokenizers/word_level_tokenizer.pickle` (LSTM/BiLSTM용)
    *   `models/lstm/best_model.h5`
    *   `models/bilstm/best_model.pt`
    *   `models/bert/` (Hugging Face 형식 모델 파일들 및 BERT 전용 토크나이저)
4.  **MLflow 실험 로그:** 각 모델의 학습 과정과 결과가 기록된 MLflow UI
5.  **베이스라인 성능 비교 보고서:** 4개 모델의 테스트셋 성능을 비교 분석한 마크다운 문서

### 7. 기술 스택

*   **언어:** Python 3.9+
*   **데이터 처리:** Pandas, NumPy, KSS, KoNLPy(Mecab)
*   **머신러닝/딥러닝:** Scikit-learn, TensorFlow, PyTorch, Hugging Face Transformers
*   **데이터베이스:** SQLite3
*   **실험 관리:** MLflow
*   **데이터 버전 관리:** DVC
