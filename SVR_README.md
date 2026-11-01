# SVR 5-fold 데모

LightGBM 및 GPR 데모와 같은 feature CSV와 `split_data_v1` fold 배정을 사용해 RBF Support Vector Regression을 평가합니다.

## 실행

```bash
.venv/bin/python scripts/train_svr_demo.py \
  --features correlation_output/features_all.csv \
  --data-root /path/to/cif_root \
  --split-root split_data_v1 \
  -o ml_output/svr_demo
```

기본 feature는 LightGBM/GPR과 같은 12개입니다. `--add-features`와 `--exclude-features`로 입력을 조정할 수 있습니다. 기본 타겟은 `106_`으로 시작하는 유일한 컬럼이며, 직접 지정할 때는 `--target COLUMN`을 사용합니다.

## 스케일링과 모델 설정

매 outer fold에서 `StandardScaler`를 해당 fold의 학습 행에만 fit하고, 같은 스케일러로 train/test를 변환합니다. 기본 모델은 다음 설정입니다.

```text
SVR(kernel='rbf', C=10.0, epsilon=0.03, gamma='scale')
```

이 데모는 위의 고정 설정을 모든 fold에서 사용하며 별도의 하이퍼파라미터 탐색 옵션은 제공하지 않습니다. 사용 파라미터는 CLI와 `train.log`에 기록됩니다.

## 평가 및 결과

각 fold의 train R², test R², test RMSE를 CLI와 `train.log`에 출력합니다. 마지막에 세 평가 지표의 5-fold 평균 및 표준편차를 기록합니다.

- `train.log`: 설정, fold별 결과 및 전체 요약
- `fold_metrics.csv`: fold별 지표, 파라미터, 학습/테스트 행 수
- `test_predictions.csv`: CIF 경로, fold, 실제값, 예측값
- `metrics.json`: 입력 설정, 전체 요약, fold별 결과

타겟 또는 선택된 feature에 결측값이 있는 행과 지원하지 않는 공간군의 행은 학습에서 제외됩니다. 각 test fold에 사용 가능한 행이 하나도 없으면 실행이 중단됩니다.
