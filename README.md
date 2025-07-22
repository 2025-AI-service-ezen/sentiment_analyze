# 감성 분석 앙상블 모델 프로젝트

이 프로젝트는 숙박업소 리뷰 데이터를 기반으로 감성 분석 앙상블 모델을 구축하고, 자기지도학습(Self-training) 기반의 MLOps 파이프라인을 통해 모델 성능을 지속적으로 개선하는 것을 목표로 합니다.

## 🚀 프로젝트 개요

전체 프로젝트는 세 가지 주요 하위 프로젝트로 구성됩니다. 각 하위 프로젝트는 독립적으로 진행되지만 유기적으로 연결되어 최종 감성 분석 API 서비스를 제공합니다.

*   **[PROJECT_OVERVIEW.md](planning/PROJECT_OVERVIEW.md)**: 프로젝트의 전체적인 목표, 핵심 요구사항 및 하위 프로젝트별 상세 정의를 포함합니다.

## 📚 문서 구조

프로젝트의 모든 문서는 체계적으로 관리되며, 아래 링크를 통해 각 문서에 접근할 수 있습니다.

### 📝 기획 문서 (Planning Documents)

프로젝트의 목표, 요구사항, 기술 사양 및 데이터 파이프라인 설계를 다룹니다.

*   **프로젝트 1: 감성 분석 모델 구현 및 학습**
    *   [PROJECT_1_PRD.md](planning/PROJECT_1_PRD.md): 제품 요구사항 정의서
    *   [PROJECT_1_TECHNICAL_SPECIFICATION.md](planning/PROJECT_1_TECHNICAL_SPECIFICATION.md): 기술 기획서
    *   [PROJECT_1_TRD.md](planning/PROJECT_1_TRD.md): 기술 요구사항 정의서
    *   [PROJECT_1_DATA_PIPELINE_DESIGN.md](planning/PROJECT_1_DATA_PIPELINE_DESIGN.md): 데이터 파이프라인 및 DBMS 설계
    *   [PROJECT_1_WORKFLOW_PLAN.md](planning/PROJECT_1_WORKFLOW_PLAN.md): 워크플로우 계획
*   **프로젝트 2: 점진적 자기지도학습 기반 데이터셋 정제 및 모델 재훈련**
    *   [PROJECT_2_PRD.md](planning/PROJECT_2_PRD.md): 제품 요구사항 정의서
    *   [PROJECT_2_TECHNICAL_SPECIFICATION.md](planning/PROJECT_2_TECHNICAL_SPECIFICATION.md): 기술 기획서
    *   [PROJECT_2_TRD.md](planning/PROJECT_2_TRD.md): 기술 요구사항 정의서
    *   [PROJECT_2_WORKFLOW_PLAN.md](planning/PROJECT_2_WORKFLOW_PLAN.md): 워크플로우 계획
*   **프로젝트 3: 앙상블 모델 기반 감성 분석 API 서비스**
    *   [PROJECT_3_PRD.md](planning/PROJECT_3_PRD.md): 제품 요구사항 정의서
    *   [PROJECT_3_TECHNICAL_SPECIFICATION.md](planning/PROJECT_3_TECHNICAL_SPECIFICATION.md): 기술 기획서
    *   [PROJECT_3_TRD.md](planning/PROJECT_3_TRD.md): 기술 요구사항 정의서
    *   [PROJECT_3_WORKFLOW_PLAN.md](planning/PROJECT_3_WORKFLOW_PLAN.md): 워크플로우 계획

### 📊 보고서 (Reports)

프로젝트의 분석, 평가 및 연구 결과를 담고 있습니다.

*   **분석 보고서**
    *   [ANALYSIS_REPORT_HOSPITALITY_DOMAIN.md](reports/analysis/ANALYSIS_REPORT_HOSPITALITY_DOMAIN.md): 숙박업소 도메인 감성 분석 모델 프로젝트 계획 분석 보고서
    *   [PROJECT_EVALUATION_REPORT.md](reports/analysis/PROJECT_EVALUATION_REPORT.md): 프로젝트 문서 종합 평가 보고서
*   **연구 보고서**
    *   [INTEGRATED_RESEARCH_REPORT.md](reports/research/INTEGRATED_RESEARCH_REPORT.md): 통합 연구 보고서
    *   [RESEARCH_REPORT_SENTIMENT_MODEL.md](reports/research/RESEARCH_REPORT_SENTIMENT_MODEL.md): 감성 분석 모델 학습, 데이터셋 정제 및 재훈련 분석 및 평가
    *   [RESEARCH_REPORT_WORD_CORRELATION_SENTIMENT.md](reports/research/RESEARCH_REPORT_WORD_CORRELATION_SENTIMENT.md): 단어 상관관계를 활용한 감성사전 기반 감성 분석 모델

### 📖 가이드 문서 (Guide Documents)

프로젝트의 특정 구현 전략 및 최종 보고서를 제공합니다.

*   [FINAL_PROJECT_REPORT.md](docs/others/FINAL_PROJECT_REPORT.md): 최종 프로젝트 보고서
*   [refinement_guide.md](docs/others/refinement_guide.md): 반복적 데이터 정제 프로세스 가이드
*   [retraining_strategy_report.md](docs/others/retraining_strategy_report.md): 4모델 앙상블 시스템 재훈련 방법론 종합 분석 보고서

### 📄 일반 프로젝트 문서 (General Project Documents)

프로젝트 전반에 걸쳐 적용되는 문서입니다.

*   [Comprehensive_Test_Plan.md](Comprehensive_Test_Plan.md): 종합 테스트 계획서
