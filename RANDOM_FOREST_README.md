# Random Forest 5-fold 데모

Random Forest 회귀를 기존 5-fold split으로 평가합니다. 공통 입력과 feature/타겟 선택 규칙은 [README.md](README.md#모델별-실행-안내)를 참고하세요.

## 실행

```bash
.venv/bin/python scripts/train_random_forest_demo.py \
  --features correlation_output/features_all.csv \
  --data-root /path/to/cif_root \
  --split-root split_data_v1 \
  -o ml_output/random_forest_demo
```

## 모델 설정

기본값은 다음과 같습니다.

```text
n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1
max_features=1.0, bootstrap=True, criterion='squared_error'
random_state=42, n_jobs=-1
```

`--tune PARAM=VALUE`로 기본 파라미터를 덮어쓸 수 있습니다. 지정한 값은 모든 fold에 적용되고 로그와 `metrics.json`에 기록됩니다. 자동 탐색은 수행하지 않습니다.

```bash
--tune n_estimators=500 max_depth=12 min_samples_leaf=2 max_features=sqrt -o ml_output/random_forest_tuned
```

위 옵션을 기본 실행 명령에 추가합니다. 지원 파라미터는 기본값 목록에 있는 항목 모두이며, `max_depth=none`으로 트리 깊이 제한을 없앨 수 있습니다. `max_features`는 `sqrt`, `log2`, 0과 1 사이의 분수 또는 1 이상의 정수입니다. `bootstrap`은 `true`/`false`를 받습니다. `criterion`은 `squared_error`, `absolute_error`, `friedman_mse`, `poisson` 중 하나입니다.

## 평가 및 결과

각 fold의 train/test R², test RMSE, 학습/테스트 행 수와 5-fold 평균 및 표준편차를 출력합니다. 공간군 필터링과 결측 행 제외는 다른 모델 데모와 같은 기준을 사용합니다.

- `train.log`: 실행 설정과 fold별 지표
- `fold_metrics.csv`: fold별 지표, 파라미터, 학습/테스트 행 수
- `test_predictions.csv`: CIF 경로, fold, 실제값, 예측값
- `feature_importance.csv`: fold 간 impurity 기반 feature 중요도 평균 및 표준편차
- `metrics.json`: 선택 feature, 타겟, 파라미터와 fold별/전체 지표
