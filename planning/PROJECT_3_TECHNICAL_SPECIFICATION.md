## 프로젝트 3: 앙상블 모델 기반 감성 분석 API 서비스 - 기술 기획서

**문서 버전:** 1.0
**작성일:** 2025년 7월 21일
**작성자:** Gemini

### 1. 프로젝트 목표

본 문서는 감성 분석 앙상블 모델 프로젝트의 세 번째 하위 프로젝트인 "앙상블 모델 기반 감성 분석 API 서비스"에 대한 기술 기획서입니다. 이 프로젝트는 프로젝트 2에서 점진적으로 완성되는 최신 감성 분석 앙상블 모델을 외부 시스템에 안정적이고 확장 가능한 RESTful API 형태로 제공하며, 모델 업데이트 시 서비스 중단 없이 자동으로 최신 모델을 반영하는 것을 목표로 합니다. 이 문서는 [`PROJECT_3_PRD.md`](PROJECT_3_PRD.md) 및 [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md)에 명시된 프로젝트의 목표와 범위를 기반으로 합니다.

### 2. 시스템 아키텍처 및 데이터 흐름

```mermaid
graph TD
    A[External Client] --> B[FastAPI Service];
    B --> C{Model Loader & Hot-swapper};
    C --> D[MLflow Model Registry<br>(Project 2 Output)];
    C --> E[Loaded Ensemble Model];
    E --> B;
    B --> F[Prometheus Exporter];
    F --> G[Prometheus Server];
    G --> H[Grafana Dashboard];

    subgraph Model Update Flow
        direction LR
        I[Project 2 MLOps Pipeline] --> J[New Model Version Registered in MLflow];
        J --> C;
    end
```

**설명:**

*   **FastAPI Service:** 외부 클라이언트의 API 요청을 처리하는 핵심 서비스입니다.
*   **Model Loader & Hot-swapper:** MLflow Model Registry를 주기적으로 모니터링하거나, 특정 이벤트를 감지하여 최신 모델 버전을 로드하고, 서비스 중단 없이 현재 사용 중인 모델을 교체하는 역할을 합니다.
*   **MLflow Model Registry:** 프로젝트 2에서 학습 및 버전 관리되는 최신 앙상블 모델이 등록되는 중앙 저장소입니다.
*   **Loaded Ensemble Model:** 현재 API 서비스에서 예측에 사용 중인 앙상블 모델 인스턴스입니다.
*   **Prometheus Exporter:** FastAPI 서비스의 메트릭(요청 수, 응답 시간, 모델 버전 등)을 Prometheus가 수집할 수 있는 형태로 노출합니다.
*   **Prometheus Server & Grafana Dashboard:** 서비스 메트릭을 수집, 저장 및 시각화하여 실시간 모니터링을 제공합니다.

### 3. 기술 스택

*   **API 프레임워크:** FastAPI
*   **모델 로딩:** MLflow Client API, PyTorch/TensorFlow (모델 형식에 따라)
*   **모델 서빙:** Uvicorn (ASGI 서버)
*   **모니터링:** Prometheus Client Library, Prometheus, Grafana
*   **컨테이너화:** Docker
*   **언어:** Python 3.9+

### 4. 구현 세부 사항

#### 4.1. API 엔드포인트

*   **`POST /predict`**
    *   **요청:** `application/json`
        ```json
        {
            "text": "분석할 텍스트입니다."
        }
        ```
    *   **응답:** `application/json`
        ```json
        {
            "sentiment": "긍정",
            "score": 0.95,
            "details": {
                "positive": 0.95,
                "negative": 0.03,
                "neutral_complex": 0.02
            }
        }
        ```
    *   **입력 유효성 검사:** Pydantic을 사용하여 요청 본문의 `text` 필드 유효성을 검사합니다.

#### 4.2. 모델 로딩

*   **MLflow Model Registry 연동:**
    *   API 서비스 시작 시, MLflow Model Registry에서 최신 버전의 앙상블 모델을 로드합니다.
    *   MLflow Client API를 사용하여 `mlflow.pyfunc.load_model()` 또는 `mlflow.<framework>.load_model()` (예: `mlflow.pytorch.load_model()`)을 통해 모델을 로드합니다.
    *   모델 로딩 경로는 환경 변수 또는 설정 파일(`config.ini`, `.env`)을 통해 관리합니다.

#### 4.3. 모델 자동 교체 (Hot-swapping)

*   **메커니즘:**
    1.  **모델 버전 모니터링:** 별도의 백그라운드 스레드 또는 주기적인 스케줄러(예: `APScheduler`)를 사용하여 MLflow Model Registry의 특정 모델(예: `ensemble-sentiment-model`)의 최신 버전을 주기적으로 확인합니다.
    2.  **새 모델 로드:** 새로운 버전이 감지되면, 해당 모델을 메모리에 로드합니다. 이 때, 현재 서비스 중인 모델은 계속해서 요청을 처리합니다.
    3.  **트래픽 전환:** 새로운 모델 로드가 완료되면, API 서비스의 예측 로직이 새로운 모델 인스턴스를 사용하도록 포인터를 전환합니다. (예: 전역 변수 또는 싱글톤 패턴으로 관리되는 모델 인스턴스를 업데이트)
    4.  **기존 모델 정리:** 이전 모델 인스턴스는 더 이상 새로운 요청을 받지 않지만, 현재 처리 중인 요청이 완료될 때까지 유지됩니다. 모든 요청이 완료되면 이전 모델 인스턴스를 메모리에서 해제합니다.
*   **무중단 서비스:** 모델 교체 중에도 API 서비스는 계속해서 요청을 처리하며, 클라이언트 입장에서는 서비스 중단을 인지할 수 없습니다.

#### 4.4. 오류 처리

*   API 입력 유효성 검사 실패 시 `HTTP 422 Unprocessable Entity` 응답을 반환합니다.
*   모델 로딩 실패, 예측 오류 등 내부 서버 오류 발생 시 `HTTP 500 Internal Server Error` 응답을 반환하고, 상세 오류 로그를 기록합니다.

#### 4.5. 모니터링

*   **Prometheus Exporter:** `prometheus_client` 라이브러리를 사용하여 API 요청 수(`Counter`), 응답 시간(`Histogram`), 오류율(`Counter`) 등 핵심 메트릭을 노출합니다.
*   **Grafana 대시보드:** Prometheus에서 수집된 데이터를 기반으로 실시간 서비스 상태를 시각화하는 대시보드를 구성합니다.

#### 4.6. 컨테이너화

*   **Dockerfile:** FastAPI 애플리케이션을 컨테이너화하기 위한 Dockerfile을 작성합니다. Python 환경 설정, 의존성 설치, 애플리케이션 코드 복사, Uvicorn을 통한 서비스 실행 명령을 포함합니다.
*   **Docker Compose (선택 사항):** 개발 및 테스트 환경에서 FastAPI 서비스, Prometheus, Grafana를 함께 실행하기 위한 Docker Compose 파일을 제공할 수 있습니다.

### 5. 배포 고려사항

*   **환경 변수:** MLflow 추적 서버 URL, 모델 이름 등 환경에 따라 달라지는 설정은 환경 변수를 통해 관리합니다.
*   **로깅:** 구조화된 로깅(예: JSON 형식)을 사용하여 로그 분석 및 모니터링 시스템과의 통합을 용이하게 합니다.
*   **리소스 관리:** Docker 컨테이너에 CPU 및 메모리 제한을 설정하여 안정적인 리소스 사용을 보장합니다.

### 6. 리스크 및 완화 전략

*   **모델 로딩 실패:**
    *   **리스크:** MLflow에서 모델 로딩 중 네트워크 문제, 파일 손상 등으로 인해 실패할 경우 서비스에 영향을 줄 수 있습니다.
    *   **완화:** 모델 로딩 로직에 재시도(retry) 메커니즘을 구현하고, 로딩 실패 시 이전 모델을 계속 사용하며 경고 알림을 발생시킵니다.
*   **모델 교체 중 지연:**
    *   **리스크:** 새로운 모델 로딩 또는 초기화에 시간이 오래 걸릴 경우, API 응답 지연이 발생할 수 있습니다.
    *   **완화:** 모델 로딩은 백그라운드에서 비동기적으로 수행하고, 로딩 완료 후 트래픽을 전환합니다. 모델 초기화 시간을 최소화하도록 최적화합니다.
*   **메모리 누수:**
    *   **리스크:** 모델 핫스왑 과정에서 이전 모델 인스턴스가 제대로 해제되지 않아 메모리 누수가 발생할 수 있습니다.
    *   **완화:** Python의 `gc.collect()`를 명시적으로 호출하거나, `weakref`를 사용하여 모델 인스턴스 참조를 관리하는 등 메모리 관리 전략을 적용합니다. 주기적인 메모리 사용량 모니터링을 통해 이상 징후를 감지합니다.
*   **동시성 문제:**
    *   **리스크:** 모델 교체 중 여러 요청이 동시에 들어올 때 일관성 없는 예측 결과를 반환할 수 있습니다.
    *   **완화:** 모델 인스턴스 전환 시 스레드 안전(thread-safe)한 방식으로 포인터를 업데이트하고, 예측 로직이 항상 유효한 모델 인스턴스를 참조하도록 보장합니다.
