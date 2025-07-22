# 🔄 반복적 데이터 정제 프로세스 가이드

> **감성 분석 데이터셋 품질 향상을 위한 Self-Supervised Learning 가이드**

## 🎯 정제 프로세스 개요

반복적 데이터 정제는 **앙상블 모델의 고신뢰도 예측을 활용하여 데이터셋의 라벨을 지속적으로 개선**하는 과정입니다. 동시에 감성사전도 진화시켜 전체 시스템의 성능을 향상시킵니다.

## 📊 정제 단계별 상세 설명

### **Step 1: 초기 모델 훈련** 🏁
```
원본 데이터 → 기본 모델 훈련 → 초기 성능 베이스라인 설정
```

#### **1.1 LSTM 모델 (CPU 최적화)**
```bash
python src/training/train_lstm_cpu.py
```

**설정 매개변수**:
- 배치 크기: 512 (32GB RAM 최적화)
- 시퀀스 길이: 128 토큰
- 임베딩 차원: 200
- LSTM 유닛: 128
- 드롭아웃: 0.3
- 에포크: 25

**예상 성능**: 정확도 85-88%

#### **1.2 BiLSTM 모델 (PyTorch)**
```bash
python src/training/train_bilstm_pytorch.py
```

**설정 매개변수**:
- 배치 크기: 256 (멀티프로세싱 고려)
- 양방향 LSTM: 64 × 2
- 워커 프로세스: 4개
- 학습률: 0.001 → 0.0001 (스케줄링)

**예상 성능**: 정확도 87-90%

#### **1.3 Word Sentiment 모델**
- 감성사전 기반 통계 모델
- 품사 태깅 활용 (명사, 형용사, 동사 중심)
- 문맥 고려 감성 점수 계산
- **단어의 앞뒤 문맥(부정어, 강조어, N-그램 등)을 고려하여 감성 점수를 조정하고, 숙박업소 도메인 특화 어휘 및 규칙 반영 (예: '체크인', '어메니티', '수압' 등)**

**예상 성능**: 정확도 75-80%

#### **1.4 BERT 모델 (백그라운드)**
- GPU 활용 딥러닝 모델
- 사전 훈련된 한국어 BERT 활용
- 현재 3/10 에포크 진행 중

**예상 성능**: 정확도 92-95%

---

### **Step 2: 앙상블 예측 생성** 🤖

#### **2.1 모델 가중치 설정**
```python
model_weights = {
    'lstm': 0.25,           # 기본 성능
    'bilstm': 0.30,         # 양방향 성능 우수
    'word_sentiment': 0.20,  # 해석 가능성
    'bert': 0.25            # 최고 성능 (가용시)
}
```

#### **2.2 신뢰도 계산 방법**
```python
def calculate_confidence(predictions):
    # 모델 간 표준편차가 낮을수록 높은 신뢰도
    std_dev = np.std(predictions)
    confidence = 1.0 - min(std_dev * 2, 1.0)
    
    # 극단값(0.9 이상, 0.1 이하)에 보너스
    if max(predictions) > 0.9 or min(predictions) < 0.1:
        confidence *= 1.2
    
    return min(confidence, 1.0)
```

#### **2.3 예측 결과 예시**
```
텍스트: "정말 좋은 서비스였어요!"
LSTM 예측: 0.85 (긍정)
BiLSTM 예측: 0.89 (긍정)  
Word Sentiment: 0.78 (긍정)
BERT 예측: 0.92 (긍정)

앙상블 예측: 0.86 (긍정)
신뢰도: 0.91 (높음) ← 모델들이 일치
```

---

### **Step 3: 고신뢰도 데이터 필터링** 🔍

#### **3.1 신뢰도 임계값**
```python
CONFIDENCE_THRESHOLD = 0.7
high_confidence_data = predictions[predictions['confidence'] >= 0.7]
```

#### **3.2 필터링 통계 예시**
```
전체 데이터: 387,566개
고신뢰도 (≥0.7): 310,053개 (80.0%)
중신뢰도 (0.5-0.7): 62,341개 (16.1%)
저신뢰도 (<0.5): 15,172개 (3.9%)

→ 80%의 고품질 데이터로 라벨 정제 진행
```

#### **3.3 신뢰도 분포 분석**
```python
confidence_stats = {
    'mean': 0.756,
    'std': 0.182,
    'min': 0.121,
    'max': 0.998,
    'q25': 0.634,
    'q50': 0.781,
    'q75': 0.897
}
```

---

### **Step 4: 라벨 보정 프로세스** ✏️

#### **4.1 보정 기준**
```python
def should_correct_label(original, predicted, confidence):
    # 높은 신뢰도 + 큰 차이 = 보정 대상
    threshold = 0.3
    return (confidence >= 0.7 and 
            abs(original - predicted) >= threshold)
```

#### **4.2 보정 통계 추적**
```python
correction_stats = {
    'total_samples': 387566,
    'corrected_samples': 15234,
    'correction_rate': 3.93,  # %
    'positive_to_negative': 2156,
    'negative_to_positive': 1987,
    'fine_tuning': 11091  # 미세 조정
}
```

#### **4.3 보정 사례**
```
원본: "서비스가 그저 그래요" → 라벨: 0.8 (긍정)
앙상블 예측: 0.3 (부정) → 신뢰도: 0.85
보정 결과: 0.3 (부정) ← 명백한 오라벨링 수정
```

---

### **Step 5: 감성사전 진화** 📚

#### **5.1 단어 추출 및 분석**
```python
def extract_words_with_context(text, sentiment, confidence):
    words = tokenize(text)
    return [{
        'word': word,
        'context': text,
        'sentiment': sentiment,
        'confidence': confidence,
        'position': position_in_text
    } for word in words if len(word) > 1]
```

#### **5.2 단어 중요도 계산**
```python
def calculate_word_importance(word, contexts):
    frequency = len(contexts)                    # 출현 빈도
    avg_confidence = np.mean([c['confidence'] for c in contexts])  # 평균 신뢰도
    consistency = 1.0 - np.std([c['sentiment'] for c in contexts])  # 일관성
    length_bonus = min(len(word) / 10, 1.0)     # 길이 보너스
    
    importance = (frequency * 0.4 + 
                 avg_confidence * 0.3 + 
                 consistency * 0.2 + 
                 length_bonus * 0.1)
    return importance
```

#### **5.3 사전 업데이트 알고리즘**
```python
def update_word_score(old_score, new_score, importance):
    # 중요도에 따른 점진적 업데이트
    alpha = 0.3 + 0.2 * importance  # 0.3~0.5 범위
    updated_score = (1 - alpha) * old_score + alpha * new_score
    return updated_score
```

#### **5.4 사전 진화 통계**
```
반복 1 결과:
- 업데이트된 단어: 12,456개
- 신규 추가: 3,421개  
- 제거된 단어: 892개
- 총 사전 크기: 47,231개
- 평균 중요도: 0.634
```

---

### **Step 6: 수렴 감지** 🎯

#### **6.1 수렴 조건**
```python
def check_convergence(current_iter, previous_iter):
    # 라벨 변화율 계산
    label_changes = count_label_differences(current_iter, previous_iter)
    label_change_rate = label_changes / len(current_iter)
    
    # 사전 변화율 계산
    dict_changes = count_dictionary_changes(current_iter, previous_iter)
    dict_change_rate = dict_changes / len(current_dictionary)
    
    # 수렴 판정 (둘 다 5% 미만)
    return (label_change_rate < 0.05 and dict_change_rate < 0.05)
```

#### **6.2 수렴 통계 예시**
```
반복 1 → 2:
- 라벨 변화율: 8.3% (미수렴)
- 사전 변화율: 12.1% (미수렴)

반복 2 → 3:  
- 라벨 변화율: 5.7% (미수렴)
- 사전 변화율: 7.4% (미수렴)

반복 3 → 4:
- 라벨 변화율: 3.2% (수렴)
- 사전 변화율: 2.8% (수렴)
→ 수렴 달성! ✅
```

---

## 📈 성능 향상 추적

### **반복별 성능 지표**
```
기본 모델 (반복 0):
- LSTM: 86.2%
- BiLSTM: 88.7%  
- Word Sentiment: 78.9%
- 앙상블: 89.1%

반복 1 완료:
- LSTM: 87.8% (+1.6%)
- BiLSTM: 90.1% (+1.4%)
- Word Sentiment: 82.3% (+3.4%)
- 앙상블: 91.2% (+2.1%)

반복 2 완료:
- LSTM: 88.9% (+1.1%)
- BiLSTM: 91.0% (+0.9%)
- Word Sentiment: 84.1% (+1.8%)
- 앙상블: 92.4% (+1.2%)
```

### **데이터 품질 개선**
```
노이즈 라벨 감소:
- 초기: 15.2% 잘못된 라벨
- 반복 1: 11.8% (-3.4%)
- 반복 2: 8.9% (-2.9%)
- 반복 3: 6.7% (-2.2%)

감성사전 품질:
- 초기: 평균 일관성 0.65
- 반복 1: 0.72 (+0.07)
- 반복 2: 0.78 (+0.06)
- 반복 3: 0.83 (+0.05)
```

---

## 🔧 최적화 팁

### **메모리 최적화**
```python
# 대용량 데이터 청크 처리
CHUNK_SIZE = 10000
for chunk in pd.read_sql(query, conn, chunksize=CHUNK_SIZE):
    process_chunk(chunk)
    gc.collect()  # 메모리 해제
```

### **CPU 활용 최적화**
```python
# 멀티프로세싱으로 예측 병렬화
from multiprocessing import Pool
with Pool(processes=4) as pool:
    results = pool.map(predict_batch, data_chunks)
```

### **디스크 I/O 최적화**
```python
# SQLite 성능 튜닝
conn.execute("PRAGMA cache_size = -65536")  # 64MB 캐시
conn.execute("PRAGMA synchronous = NORMAL")
conn.execute("PRAGMA journal_mode = WAL")
```

---

## 🚨 주의사항 및 문제 해결

### **일반적인 문제들**

#### **1. 메모리 부족 (OOM)**
**증상**: `MemoryError` 또는 시스템 멈춤
**해결책**:
```python
# 배치 크기 동적 조정
if psutil.virtual_memory().percent > 85:
    current_batch_size = max(current_batch_size // 2, 32)
    print(f"메모리 부족으로 배치 크기를 {current_batch_size}로 감소")
```

#### **2. 수렴하지 않음**
**증상**: 10회 반복 후에도 변화율이 높음
**해결책**:
- 신뢰도 임계값 상향 조정 (0.7 → 0.8)
- 앙상블 가중치 재조정
- 이상치 데이터 수동 검토

#### **3. 사전 품질 저하**
**증상**: 일관성 점수 감소, 이상한 단어 추가
**해결책**:
```python
# 품질 필터 강화
MIN_CONTEXTS = 5      # 최소 출현 횟수
MIN_CONSISTENCY = 0.7  # 최소 일관성
MIN_IMPORTANCE = 0.4   # 최소 중요도
```

### **모니터링 알림**
```python
# 이상 상황 자동 감지
def check_anomalies(stats):
    alerts = []
    
    if stats['memory_usage'] > 90:
        alerts.append("⚠️ 메모리 사용률 위험 수준")
    
    if stats['label_change_rate'] > 50:
        alerts.append("⚠️ 비정상적으로 높은 라벨 변화율")
    
    if stats['avg_confidence'] < 0.5:
        alerts.append("⚠️ 앙상블 신뢰도 저하")
    
    return alerts
```

---

## 🎯 품질 보증 체크리스트

### **각 반복 후 확인사항**
- [ ] **메모리 사용률**: 85% 이하 유지
- [ ] **라벨 변화율**: 합리적 범위 (1-20%)
- [ ] **신뢰도 분포**: 평균 0.6 이상
- [ ] **사전 일관성**: 0.7 이상 유지
- [ ] **모델 성능**: 이전 대비 동등 이상

### **최종 완료 시 검증**
- [ ] **수렴 달성**: 라벨 + 사전 변화율 < 5%
- [ ] **성능 향상**: 초기 대비 3% 이상 개선
- [ ] **데이터 무결성**: 결측값, 이상치 없음
- [ ] **재현성**: 동일 결과 재생산 가능
- [ ] **문서화**: 모든 변화 이력 기록

---

## 📊 결과 해석 가이드

### **성공적인 정제의 지표**
1. **지속적 성능 향상**: 매 반복마다 1-3% 개선
2. **안정적 수렴**: 3-7회 반복 내 수렴 달성
3. **균형 잡힌 개선**: 모든 모델이 고르게 향상
4. **품질 있는 사전**: 일관성 0.8 이상, 중요 단어 보존

### **문제가 있는 경우의 신호**
1. **성능 저하**: 특정 반복에서 성능 하락
2. **불안정한 변화**: 변화율이 계속 증가
3. **사전 오염**: 이상한 단어들이 대량 추가
4. **편향 증가**: 특정 클래스로 예측 집중

---

*정제 프로세스 가이드 버전: 1.0*  
*마지막 업데이트: 2025-07-17*
