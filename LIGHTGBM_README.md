# LightGBM 5-fold 데모

이 문서는 LightGBM 회귀 학습 스크립트의 feature 선택, 고정 모델 설정 및 결과 파일을 설명합니다.

## 실행

전체 데이터로 `features_all.csv`를 만든 다음 기존 `split_data_v1`의 `summary.json`과 `fold_membership.csv`를 사용합니다. `--data-root`에는 feature CSV 생성에 사용한 CIF 루트를 지정합니다.

```bash
.venv/bin/python scripts/train_lightgbm_demo.py \
  --features correlation_output/features_all.csv \
  --data-root /path/to/cif_root \
  --split-root split_data_v1 \
  -o ml_output/lightgbm_demo
```

기본 입력 feature는 아래 12개입니다.

```text
002_TM_layer_entropy
003_Na_content_per_O2
008_alkali_fraction_in_TM_layer
010_alkaline_earth_fraction_in_TM_layer
016_Ni_Co_Mn_Fe_required_charge
107_AtomicRadius_mean
108_AtomicRadius_avg_dev
110_Electronegativity_avg_dev
111_NdValence
112_CohesiveEnergy
113_is_Prismatic
114_is_2Layer
```

`--add-features`와 `--exclude-features`로 feature를 조정할 수 있습니다. 컬럼명은 `features_all.csv` 헤더와 같아야 합니다. 타겟은 `106_`으로 시작하는 유일한 컬럼을 자동 선택하며, 직접 지정하려면 `--target COLUMN`을 사용합니다.

```bash
.venv/bin/python scripts/train_lightgbm_demo.py \
  --exclude-features 002_TM_layer_entropy \
  --add-features 001_example_feature 115_example_feature \
  --target 106_O_Na_O_to_O_TM_O_ratio
```

## 모델 설정

기본 실행은 다음 파라미터를 사용합니다.

```text
max_depth=3, num_leaves=7, min_child_samples=8
learning_rate=0.05, colsample_bytree=0.7, n_estimators=150
reg_lambda=3.0, reg_alpha=0.1, objective='regression', random_state=42
```

모델은 위의 고정 파라미터를 모든 fold에 적용합니다. 별도의 하이퍼파라미터 탐색 옵션은 제공하지 않습니다. 실제 학습에 사용되는 파라미터는 CLI와 `train.log`에 출력됩니다.

## 데이터와 평가

공간군이 160, 166, 194인 구조만 사용합니다. CIF와 JSON 공간군이 모두 있으면 양쪽 모두 지원 목록에 있어야 합니다. 선택된 feature 또는 타겟에 결측값이 있는 행은 제외됩니다. 각 test fold에 사용 가능한 행이 없으면 학습이 중단됩니다.

fold별 train R², test R², test RMSE와 행 수가 CLI 및 로그에 출력됩니다. 결과 디렉토리에는 다음 파일이 저장됩니다.

- `train.log`: 실행 설정 및 fold별 학습 결과
- `fold_metrics.csv`: fold별 성능과 학습/테스트 행 수
- `test_predictions.csv`: CIF 경로, fold, 실제값, 예측값
- `feature_importance.csv`: split 기반 feature 중요도 평균 및 표준편차
- `gain_feature_importance.csv`: gain 기반 feature 중요도 평균 및 표준편차
- `metrics.json`: feature, 타겟, 설정, fold 지표 및 전체 요약

중요도와 예측 결과 그래프는 다음 스크립트로 생성합니다.

```bash
.venv/bin/python scripts/plot_lightgbm_results.py ml_output/lightgbm_demo
```

실행 결과 디렉토리가 다르면 그래프 명령의 경로를 그에 맞게 바꾸세요.
