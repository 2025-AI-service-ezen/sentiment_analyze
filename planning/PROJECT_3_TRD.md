## 프로젝트 3: 앙상블 모델 기반 감성 분석 API 서비스 - 기술 요구사항 정의서 (TRD)

**문서 버전:** 1.0
**작성일:** 2025년 7월 21일
**작성자:** Gemini

### 1. 서론

본 문서는 감성 분석 앙상블 모델 프로젝트의 세 번째 하위 프로젝트인 "앙상블 모델 기반 감성 분석 API 서비스"에 대한 기술 요구사항을 정의합니다. 이 프로젝트는 프로젝트 2에서 점진적으로 완성되는 최신 감성 분석 앙상블 모델을 외부 시스템에 안정적이고 확장 가능한 RESTful API 형태로 제공하며, 모델 업데이트 시 서비스 중단 없이 자동으로 최신 모델을 반영하는 것을 목표로 합니다. [`PROJECT_3_PRD.md`](PROJECT_3_PRD.md) 및 [`PROJECT_3_TECHNICAL_SPECIFICATION.md`](PROJECT_3_TECHNICAL_SPECIFICATION.md)의 내용을 기반으로 작성되었습니다.

### 2. 시스템 아키텍처 개요

`PROJECT_3_TECHNICAL_SPECIFICATION.md`에 명시된 시스템 아키텍처를 따르며, 핵심 구성 요소는 FastAPI 서비스, 모델 로더 및 핫스왑퍼, MLflow 모델 레지스트리, Prometheus 및 Grafana를 포함합니다.

### 3. 기술 요구사항

#### 3.1. API 엔드포인트

*   **`POST /predict`**
    *   **요청:** `application/json` 형식으로 `text` 필드를 포함해야 합니다.
        *   `text` (string, 필수): 감성 분석을 수행할 텍스트.
    *   **응답:** `application/json` 형식으로 감성 분석 결과(`sentiment`, `score`, `details`)를 반환해야 합니다.
        *   `sentiment` (string): 예측된 감성 라벨 (예: "긍정", "부정", "중립/복합").
        *   `score` (float): 예측된 감성 점수 (0.0 ~ 1.0).
        *   `details` (object): 각 감성 범주에 대한 상세 점수.
    *   **유효성 검사:** Pydantic 모델을 사용하여 `text` 필드의 존재 여부 및 타입 유효성을 검사해야 합니다.

#### 3.2. 모델 로딩 및 관리

*   **초기 모델 로딩:** API 서비스 시작 시, MLflow Model Registry에서 `ensemble-sentiment-model`의 최신 `Production` 또는 `Staging` 버전 모델을 로드해야 합니다.
*   **모델 버전 모니터링:** 백그라운드 스레드 또는 주기적인 스케줄러(예: `APScheduler`)를 사용하여 MLflow Model Registry의 `ensemble-sentiment-model`의 최신 버전을 5분 간격으로 확인해야 합니다.
*   **새 모델 로드:** 새로운 버전의 모델이 감지되면, 서비스 중인 모델에 영향을 주지 않고 새로운 모델을 메모리에 로드해야 합니다.
*   **모델 핫스왑:** 새로운 모델 로드가 완료되면, 현재 예측에 사용되는 모델 인스턴스를 새로운 모델로 원자적으로 교체해야 합니다. 이 과정에서 진행 중인 요청은 현재 모델로 완료되고, 새로운 요청은 새 모델로 처리되어야 합니다.
*   **이전 모델 해제:** 모델 핫스왑 후, 이전 모델 인스턴스는 더 이상 사용되지 않음을 확인한 후 메모리에서 안전하게 해제되어야 합니다.

#### 3.3. 모니터링

*   **메트릭 노출:** Prometheus Python 클라이언트 라이브러리를 사용하여 다음 메트릭을 노출해야 합니다.
    *   `http_requests_total` (Counter): 총 API 요청 수.
    *   `http_request_duration_seconds` (Histogram): API 요청 처리 시간.
    *   `model_version_info` (Gauge): 현재 서비스 중인 모델의 버전 정보.
    *   `model_load_total` (Counter): 모델 로드 시도 횟수.
    *   `model_load_success_total` (Counter): 모델 로드 성공 횟수.
    *   `model_hot_swap_total` (Counter): 모델 핫스왑 성공 횟수.
*   **Grafana 대시보드:** Prometheus에서 수집된 메트릭을 기반으로 API 서비스 상태를 실시간으로 모니터링할 수 있는 Grafana 대시보드를 구성해야 합니다.

#### 3.4. 컨테이너화

*   **Dockerfile:** FastAPI 애플리케이션을 컨테이너화하기 위한 Dockerfile을 제공해야 합니다. 이 Dockerfile은 Python 환경 설정, 필요한 라이브러리 설치, 애플리케이션 코드 복사, Uvicorn을 통한 서비스 실행 명령을 포함해야 합니다.
*   **환경 변수:** MLflow 추적 서버 URL, 모델 이름 등은 환경 변수를 통해 컨테이너 내부로 주입되어야 합니다.

### 4. 구현 세부 사항

#### 4.1. 모델 핫스왑 로직 (의사코드)

```python
import threading
import time
import mlflow

class ModelHotSwapper:
    def __init__(self, model_name, mlflow_tracking_uri):
        self.model_name = model_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self._current_model = self._load_latest_model() # Initial load
        self._lock = threading.Lock()

    def _load_latest_model(self):
        # Logic to query MLflow Model Registry for the latest Production/Staging version
        # and load the model. Handle potential errors.
        try:
            latest_version = mlflow.tracking.MlflowClient().get_latest_versions(self.model_name, stages=["Production", "Staging"])[0]
            model_uri = f"models:/{self.model_name}/{latest_version.version}"
            print(f"Loading new model version: {latest_version.version}")
            new_model = mlflow.pyfunc.load_model(model_uri)
            return new_model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def get_model(self):
        with self._lock:
            return self._current_model

    def monitor_and_swap(self):
        while True:
            time.sleep(300) # Check every 5 minutes
            new_model = self._load_latest_model()
            if new_model and new_model != self._current_model: # Simple check, more robust version comparison needed
                with self._lock:
                    # Perform atomic swap
                    self._current_model = new_model
                    print("Model hot-swapped successfully!")
                # Trigger Prometheus metric update for model version
                # Clean up old model (e.g., force garbage collection if needed)

# In FastAPI app startup:
model_swapper = ModelHotSwapper("ensemble-sentiment-model", "http://mlflow-server:5000")
threading.Thread(target=model_swapper.monitor_and_swap, daemon=True).start()

# In /predict endpoint:
model = model_swapper.get_model()
prediction = model.predict(text_input)
```

### 5. 테스트 전략

*   **단위 테스트:**
    *   API 엔드포인트의 입력 유효성 검사 로직.
    *   모델 로딩 및 예측 기능.
    *   Prometheus 메트릭 노출 기능.
*   **통합 테스트:**
    *   MLflow Model Registry에 새로운 모델 버전 등록 시 API 서비스가 자동으로 모델을 교체하는지 확인.
    *   모델 교체 중에도 API 서비스가 중단 없이 요청을 처리하는지 부하 테스트를 통해 검증.
    *   Prometheus 및 Grafana 대시보드에 메트릭이 올바르게 표시되는지 확인.
*   **성능 테스트:**
    *   JMeter 또는 Locust와 같은 도구를 사용하여 API 서비스의 RPS(Requests Per Second) 및 응답 시간 성능을 측정.

### 6. 배포 고려사항

*   **환경 변수:** MLflow 추적 서버 URL, 모델 이름 등은 배포 환경에서 환경 변수로 설정되어야 합니다.
*   **로깅:** 컨테이너 환경에 적합한 로깅 전략(예: stdout/stderr로 출력하여 컨테이너 오케스트레이션 시스템에서 수집)을 사용해야 합니다.
*   **리소스 제한:** Docker 또는 Kubernetes 배포 시 컨테이너에 CPU 및 메모리 제한을 명시적으로 설정하여 안정적인 운영을 보장해야 합니다.

### 7. 리스크 및 완화 전략

*   **모델 로딩 실패:**
    *   **리스크:** 네트워크 문제, MLflow 서버 문제, 모델 파일 손상 등으로 인해 새로운 모델 로딩이 실패할 수 있습니다.
    *   **완화:** 로딩 로직에 예외 처리 및 재시도 메커니즘을 구현하고, 실패 시 이전 모델을 계속 사용하며 모니터링 시스템을 통해 경고 알림을 발생시킵니다.
*   **메모리 누수:**
    *   **리스크:** 모델 핫스왑 과정에서 이전 모델 인스턴스가 제대로 해제되지 않아 메모리 사용량이 지속적으로 증가할 수 있습니다.
    *   **완화:** Python의 `gc.collect()`를 명시적으로 호출하거나, `weakref`를 사용하여 모델 인스턴스 참조를 관리하는 등 메모리 관리 전략을 적용합니다. 주기적인 메모리 사용량 모니터링을 통해 이상 징후를 감지하고, 필요한 경우 컨테이너 재시작 정책을 고려합니다.
*   **동시성 문제:**
    *   **리스크:** 모델 핫스왑 중 여러 API 요청이 동시에 들어올 때, 예측에 사용되는 모델 인스턴스에 대한 동시성 문제가 발생할 수 있습니다.
    *   **완화:** `threading.Lock`과 같은 동기화 메커니즘을 사용하여 모델 인스턴스 포인터 업데이트를 스레드 안전하게 처리하고, 예측 로직이 항상 유효한 모델 인스턴스를 참조하도록 보장합니다.
