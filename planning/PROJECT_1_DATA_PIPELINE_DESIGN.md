## 프로젝트 1: 데이터 파이프라인 및 DBMS 설계

**문서 버전:** 1.1
**작성일:** 2025년 7월 18일

### 1. 데이터베이스(SQLite) 스키마 설계

데이터의 통합, 정제, 버전 관리를 효율적으로 수행하기 위해 다음과 같이 정규화된 데이터베이스 스키마를 설계합니다.

```sql
-- 원본 리뷰 데이터를 저장하는 테이블
CREATE TABLE IF NOT EXISTS raw_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- 고유 ID
    source TEXT NOT NULL, -- 데이터 출처 (예: 'y_review', 'naver_shopping' 등 숙박업소 리뷰 데이터 소스)
    original_id TEXT, -- 원본 데이터의 ID (필요시)
    review_text TEXT NOT NULL, -- 원본 리뷰 문단
    original_label INTEGER NOT NULL, -- 원본 평점 (1-5)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- 생성 시각
);

-- 문장 단위로 분리된 데이터를 저장하는 테이블
CREATE TABLE IF NOT EXISTS sentences (
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- 고유 ID
    raw_review_id INTEGER NOT NULL, -- raw_reviews 테이블의 ID (FK)
    sentence_text TEXT NOT NULL, -- 분리된 문장
    is_temporary_label BOOLEAN NOT NULL, -- 임시 라벨 여부 (문장 분리 시 True)
    current_label INTEGER NOT NULL, -- 현재 할당된 라벨 (1-5)
    label_last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 라벨 최종 수정 시각
    FOREIGN KEY (raw_review_id) REFERENCES raw_reviews (id)
);

-- 모델 학습에 사용될 데이터셋 버전을 관리하는 테이블
CREATE TABLE IF NOT EXISTS dataset_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- 고유 ID
    version_name TEXT NOT NULL UNIQUE, -- 버전 이름 (e.g., 'v1.0-balanced-200k')
    description TEXT, -- 버전에 대한 설명
    dvc_tag TEXT, -- DVC 태그 (해당 버전을 재현하기 위한 태그)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- 생성 시각
);

-- 특정 데이터셋 버전에 어떤 문장이 포함되는지 매핑하는 테이블
CREATE TABLE IF NOT EXISTS dataset_sentences (
    dataset_version_id INTEGER NOT NULL,
    sentence_id INTEGER NOT NULL,
    PRIMARY KEY (dataset_version_id, sentence_id),
    FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions (id),
    FOREIGN KEY (sentence_id) REFERENCES sentences (id)
);

-- 모델별 토크나이저 및 임베딩 결과를 관리 (향후 확장용)
-- CREATE TABLE IF NOT EXISTS model_artifacts (...);
```

### 2. 데이터 처리 파이프라인 (의사코드)

Python과 Pandas, KSS, Scikit-learn, SQLite 라이브러리를 사용하여 다음의 파이프라인을 구현합니다.

**1단계: 원본 데이터 통합 및 DB 저장**

```python
import pandas as pd
import sqlite3

def integrate_raw_data_to_db(y_review_path, naver_shopping_path, db_path):
    # DB 연결
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # y_review.csv 처리 (숙박업소 리뷰 데이터에 맞게 수정 필요)
    y_review_df = pd.read_csv(y_review_path)
    # y_review.csv 처리 (벡터화 및 벌크 삽입 최적화)    y_review_df = pd.read_csv(y_review_path)    y_review_data = [('y_review', row['review'], row['rating'] if row['rating'] != 0 else 3) for _, row in y_review_df.iterrows()]    cursor.executemany("        INSERT INTO raw_reviews (source, review_text, original_label)        VALUES (?, ?, ?)    ", y_review_data)    # naver_shopping.txt 처리 (숙박업소 리뷰 데이터에 맞게 수정 필요)    naver_df = pd.read_csv(naver_shopping_path, sep='	', header=None, names=['rating', 'review'])    naver_data = [('naver_shopping', row['review'], row['rating'] if row['rating'] != 0 else 3) for _, row in naver_df.iterrows()]    cursor.executemany("        INSERT INTO raw_reviews (source, review_text, original_label)        VALUES (?, ?, ?)    ", naver_data)    conn.commit()    conn.close()    print("Raw data integration complete.")

```

**2단계: 문장 분리 및 임시 라벨링**

```python
import kss # Korean Sentence Splitter

def process_sentences_from_raw_reviews(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 모든 원본 리뷰 조회
    cursor.execute("SELECT id, review_text, original_label FROM raw_reviews")
    raw_reviews = cursor.fetchall()

    for review_id, review_text, original_label in raw_reviews:
        # KSS를 사용하여 문장 분리
        sentences = kss.split_sentences(review_text)

        for sentence in sentences:
            # 문장이 너무 짧거나 길면 제외 (예: 5자 미만)
            if len(sentence.strip()) < 5:
                continue

            cursor.execute("""
                INSERT INTO sentences (raw_review_id, sentence_text, is_temporary_label, current_label)
                VALUES (?, ?, ?, ?)
            """, (review_id, sentence, True, original_label))

    conn.commit()
    conn.close()
    print("Sentence splitting and temporary labeling complete.")
```

### 3단계: 초기 모델 학습을 위한 데이터셋 생성 (최대 2만개 샘플)

초기 감성 모델 학습을 위해 평점 1과 5의 데이터만 사용하여 최대 2만개(각 라벨당 최대 1만개)의 균형 잡힌 데이터셋을 생성합니다. 이 데이터셋은 초기 모델의 베이스라인 학습에 사용됩니다. 나머지 평점(2, 3, 4)의 문장들은 초기 모델 학습에 사용되지 않고, 이후 학습된 초기 감성 모델에 의해 정제될 예정입니다.

