## 프로젝트 2: 점진적 자기지도학습 기반 데이터셋 정제 및 모델 재훈련 - 기술 기획서

**문서 버전:** 1.1
**작성일:** 2025년 7월 18일
**작성자:** Gemini

### 1. 프로젝트 목표

프로젝트 1에서 구축된 4개의 감성 분석 모델(Word Sentiment, LSTM, BiLSTM, BERT)을 활용하여, **전체 숙박업소 리뷰 데이터셋**의 임시 라벨을 검증하고 정제한다. 이렇게 정제된 고품질 데이터셋으로 모델을 반복적으로 재훈련하여 성능을 점진적으로 극대화하는 자기지도학습(Self-training) 루프를 구축한다. 이 과정에서 멀티코어 CPU 및 GPU 자원을 최대한 활용하여 효율적인 학습을 달성하고, 자기지도학습의 확증 편향(Confirmation Bias) 문제를 방지하는 전략을 적용한다.

### 2. 시스템 아키텍처 및 데이터 흐름

```mermaid
graph TD
    A[SQLite Database<br>- sentences (All Data)] --> B{1. Ensemble Prediction<br>(Batch Processing)};
    B --> C{2. Confidence Calculation &<br>Label Correction};
    C --> D[Refined Dataset<br>(Updated Labels in DB)];
    D -- DVC for Versioning --> E{3. Sequential Model Retraining<br>(Full Dataset)};
    E --> F1[Word Sentiment Model];
    E --> F2[LSTM Model];
    E --> F3[BiLSTM Model];
    E --> F4[BERT Model];
    
    subgraph MLflow Tracking
        F1 --> G{Experiment Tracking};
        F2 --> G;
        F3 --> G;
        F4 --> G;
    end

    G -- Log Artifacts --> H[Updated Model Storage];
    G -- Log Metrics --> I[Performance Reports];
    
    I --> J{4. Convergence Check &<br>Confirmation Bias Mitigation};
    J -- Loop if not converged --> A;
    J -- End if converged --> K[Final Refined Models & Dataset];

    subgraph Resource Optimization
        direction LR
        L[Multi-core CPU<br>(4C/8T+)] --> B;
        M[GPU (GTX 960/1070/1080Ti)] --> E;
    end

```

### 3. 데이터 정제 및 재훈련 파이프라인

#### 3.1. 데이터 로딩 및 앙상블 예측 (전체 데이터셋 활용)
*   **목표:** SQLite DB에 저장된 모든 숙박업소 리뷰 문장 데이터를 효율적으로 로드하고, 프로젝트 1에서 학습된 4개 모델을 활용하여 각 문장의 감성을 예측한다.
*   **전략:**
    *   **청크(Chunk) 단위 로딩:** 대용량 데이터셋을 한 번에 메모리에 올리지 않고, Pandas의 `read_sql` `chunksize` 옵션을 활용하여 청크 단위로 데이터를 로드한다.
    *   **멀티프로세싱 예측:** 앙상블 모델의 예측 과정은 CPU 바운드 작업이 될 수 있으므로, `multiprocessing.Pool`을 활용하여 여러 코어에서 병렬적으로 예측을 수행한다.
    *   **GPU 활용:** BERT와 같은 GPU 기반 모델의 예측은 GPU에서 직접 수행하여 효율을 높인다.

#### 3.2. 신뢰도 기반 라벨 보정
*   **목표:** 앙상블 예측 결과의 신뢰도를 계산하고, 높은 신뢰도를 가진 예측을 기반으로 `sentences` 테이블의 `current_label`을 업데이트한다.
*   **전략:**
    *   **신뢰도 임계값:** `refinement_guide.md`에서 제시된 앙상블 모델의 예측 신뢰도 계산 로직을 활용하여, 일정 임계값(예: 0.7) 이상의 신뢰도를 가진 예측에 대해서만 라벨 보정을 수행한다.
    *   **라벨 업데이트:** 보정된 라벨은 `sentences` 테이블의 `current_label` 필드를 업데이트하고, `label_last_updated` 타임스탬프를 갱신한다.

#### 3.3. 모델 재훈련 (순차적 GPU 활용)
*   **목표:** 정제된 전체 데이터셋을 활용하여 4개 모델을 재훈련하고 성능을 향상시킨다.
*   **전략:**
    *   **순차적 학습:** 사용자의 요구사항에 따라, 여러 모델을 동시에 학습시키지 않고 GPU 자원을 최대한 활용하여 **순차적으로** 모델을 학습시킨다. 학습 순서는 `retraining_strategy_report.md`에서 제안된 효율성 기반 순서(Word Sentiment -> LSTM -> BiLSTM -> BERT)를 따른다.
    *   **GPU 자원 최적화:** 각 모델 학습 시, `tf.config.experimental.set_memory_growth(gpu, True)` (TensorFlow) 또는 `torch.cuda.empty_cache()` (PyTorch)와 같은 GPU 메모리 관리 기법을 적극 활용하여 OOM(Out Of Memory)을 방지하고 GPU 활용률을 극대화한다.
    *   **모델별 재훈련 전략:** `retraining_strategy_report.md`에 명시된 각 모델별 최적 재훈련 전략(BERT 파인튜닝, BiLSTM 부분 재훈련, LSTM 전이학습, Word Sentiment 점진적 사전 업데이트)을 적용한다.

#### 3.4. 수렴 감지 및 확증 편향 완화
*   **목표:** 데이터 정제 및 모델 재훈련 루프의 수렴 여부를 판단하고, 자기지도학습의 확증 편향 문제를 능동적으로 관리한다.
*   **전략:**
    *   **수렴 조건:** `refinement_guide.md`에서 제시된 라벨 변화율 및 감성 사전 변화율을 주요 수렴 지표로 활용한다.
    *   **확증 편향 완화:**
        *   **신뢰도 임계값 조정:** 루프가 진행됨에 따라 신뢰도 임계값을 점진적으로 높여, 모델의 예측에 대한 확신이 더 높은 데이터만 정제에 사용하도록 한다.
        *   **주기적인 수동 검토:** 일정 반복마다 또는 성능 저하 감지 시, 무작위 샘플 또는 모델 간 불일치 샘플을 추출하여 숙박업소 도메인 전문가가 수동으로 라벨을 검토하고 수정하는 프로세스를 도입한다.
        *   **다양한 앙상블 기법:** 가중 평균 외에 스태킹(Stacking) 등 다양한 앙상블 기법을 실험하여 모델의 다양성을 확보하고 확증 편향을 줄인다.
        *   **데이터 증강:** 특히 소수 클래스에 대한 데이터 증강 기법을 활용하여 모델이 더 다양한 데이터를 학습하도록 유도한다.

    #### 3.5. LLM (Gemini API) 활용 전략
    *   **목표:** Gemini API (`gemini-2.5-flash` 모델)를 활용하여 자기지도학습 루프의 특정 단계에서 인간 전문가의 개입을 보조하고, 데이터 품질 및 모델 견고성을 향상시킨다. API 호출 지연, 응답 없음, 비용 문제 등을 최소화하기 위해 다음과 같은 전략을 따른다.
    *   **재라벨링 (Re-labeling):**
        *   **대상:** 앙상블 모델의 예측 신뢰도가 낮거나(예: 0.5 미만) 모델 간 예측이 크게 불일치하는 샘플에 대해 `gemini-2.5-flash`를 사용하여 재라벨링을 시도한다.
        *   **비용 효율성:** 모든 저신뢰도 샘플을 LLM으로 재라벨링하는 대신, 가장 불확실한 소수의 샘플에만 적용하여 API 호출 횟수를 최소화한다.
        *   **프롬프트 엔지니어링:** 여러 문장을 하나의 API 호출로 묶어 배치 처리하고, 명확한 지침과 소수 학습(few-shot learning) 예시를 제공하여 LLM의 응답 품질을 높인다.
    *   **앙상블 모델 평가 (Qualitative Evaluation):**
        *   **대상:** 앙상블 모델의 예측이 잘못되었거나, 엣지 케이스로 판단되는 소수의 샘플에 대해 `gemini-2.5-flash`를 사용하여 정성적 분석을 수행한다.
        *   **활용:** LLM에게 문장, 앙상블 예측, 개별 모델 예측 등을 제공하고, 예측의 근거, 오분류 원인 분석, 개선 방안 등을 질의하여 모델 성능 개선에 활용한다.
        *   **비용 효율성:** 매우 제한된 수의 샘플에 대해서만 정성적 평가를 수행하여 API 호출 비용을 극소화한다.
    *   **편향 감지 및 완화 (Bias Detection & Mitigation):**
        *   **대상:** 데이터셋 또는 모델 예측에서 잠재적인 편향(예: 특정 성별, 인종, 도메인 특화 어휘에 대한 편향)이 의심되는 소수의 샘플 또는 패턴에 대해 `gemini-2.5-flash`를 활용한다.
        *   **활용:** LLM에게 편향된 것으로 의심되는 문장들을 제공하고, 편향 유형 식별, 편향된 표현의 재작성 제안, 또는 편향을 완화할 수 있는 새로운 데이터 생성 아이디어 등을 요청한다.
        *   **비용 효율성:** 편향 분석은 주기적으로, 그리고 매우 선별된 데이터셋에 대해서만 수행하여 비용을 관리한다.
    *   **API 호출 최적화:**
        *   **RPM/TPM 관리:** `gemini-2.5-flash`의 분당 호출 제한(RPM) 및 토큰 제한(TPM)을 준수하기 위해 API 호출 로직에 지연(delay)을 적용하거나, 호출 큐(queue)를 구현한다.
        *   **프롬프트 압축:** 불필요한 정보를 제거하고 핵심 내용만 포함하도록 프롬프트를 간결하게 작성하여 입력 토큰 사용량을 최소화한다.
        *   **응답 파싱:** LLM의 응답에서 필요한 정보만 효율적으로 추출하도록 파싱 로직을 최적화한다.

### 4. 최적화 전략

*   **멀티코어 CPU 활용:**
    *   데이터 로딩, 전처리(형태소 분석 등), 앙상블 예측 등 CPU 바운드 작업에 `multiprocessing` 모듈을 적극 활용하여 4코어 8스레드 이상의 CPU 자원을 효율적으로 사용한다. 특히, GIL(Global Interpreter Lock)의 영향을 최소화하기 위해 CPU 바운드 작업은 별도의 프로세스에서 실행하고, I/O 바운드 작업(예: DB 읽기/쓰기)에는 멀티스레딩을 고려한다.
    *   TensorFlow의 `tf.config.threading.set_intra_op_parallelism_threads` 및 `set_inter_op_parallelism_threads` 설정을 통해 CPU 활용을 최적화한다.
*   **GPU 자원 관리:**
    *   GTX 960, GTX 1070, GTX 1080Ti 등 다양한 GPU 환경에서 CUDA를 활용할 수 있도록 PyTorch와 TensorFlow의 GPU 설정(`cuda_gpu_setup_guide.md` 참조)을 통합 관리한다.
    *   **동적 배치 사이즈 조절:** GPU 메모리 사용량을 실시간으로 모니터링하여 OOM(Out Of Memory)을 방지하고 GPU 활용률을 극대화하기 위해 배치 사이즈를 동적으로 조절하는 로직을 구현한다. (예: `torch.cuda.max_memory_allocated()` 또는 `tf.config.experimental.get_memory_info()` 활용)
    *   **GPU별 학습 파라미터 튜닝 가이드라인:** 각 GPU 모델의 VRAM 크기 및 Compute Capability를 고려하여, 모델별 최적의 학습 파라미터(예: 학습률, 옵티마이저)를 동적으로 선택하거나 권장하는 가이드라인을 제시한다.
    *   모델 학습 전 GPU 메모리를 초기화(`torch.cuda.empty_cache()`, `tf.keras.backend.clear_session()`)하여 이전 학습의 잔여 메모리를 제거한다.
*   **디스크 I/O 최적화:** SQLite 데이터베이스의 `PRAGMA` 설정을 통해 디스크 I/O 성능을 최적화한다.

### 5. 리스크 관리

*   **확증 편향:** 위 3.4절에서 제시된 전략들을 통해 확증 편향을 지속적으로 모니터링하고 완화한다.
*   **자원 부족:** 대용량 데이터 처리 및 GPU 학습 시 메모리 부족(OOM) 문제가 발생할 수 있으므로, 동적 배치 크기 조정, 가비지 컬렉션 강제 실행, 그리고 `psutil`을 이용한 메모리 사용량 모니터링을 통해 관리한다.
*   **학습 시간:** 전체 데이터셋을 활용하고 여러 모델을 순차적으로 학습함에 따라 학습 시간이 길어질 수 있다. MLflow를 통한 학습 시간 추적 및 조기 종료(Early Stopping)를 적극 활용하여 불필요한 학습을 방지한다.
*   **하드웨어 호환성:** 다양한 GPU 환경에서의 호환성 문제를 `cuda_gpu_setup_guide.md` 및 `cuda_system_analysis.md`를 참조하여 사전에 테스트하고 필요한 경우 환경별 설정 스크립트를 제공한다.

### 6. 최종 산출물 (Deliverables)

1.  **정제된 데이터셋:** 자기지도학습 루프를 통해 라벨이 보정된 최종 데이터셋 (SQLite DB 및 DVC 버전 관리 파일)
2.  **성능 향상된 모델 아티팩트:** 재훈련을 통해 성능이 개선된 4개 모델의 최신 버전 (MLflow 모델 레지스트리 등록 및 파일 저장). **프로젝트 3에서 최신 모델을 자동으로 로드할 수 있도록 모델 버전 관리 및 배포 메커니즘을 포함한다.**
3.  **재훈련 과정 보고서:** 각 반복(iteration)별 데이터 정제 통계, 모델 성능 변화, 수렴 지표 등을 포함하는 상세 보고서
4.  **MLflow 실험 로그:** 재훈련 과정의 모든 실험 정보가 기록된 MLflow UI
5.  **확증 편향 분석 보고서:** 확증 편향 완화 전략의 효과 및 데이터 품질 변화에 대한 분석 보고서
