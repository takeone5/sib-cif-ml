# SVR 5-fold 데모

공통 입력과 feature/타겟 선택 규칙은 [README.md](README.md#모델별-실행-안내)를 참고하세요. 이 스크립트는 RBF Support Vector Regression을 평가합니다.

## 실행

```bash
.venv/bin/python scripts/train_svr_demo.py \
  --features correlation_output/features_all.csv \
  --data-root /path/to/cif_root \
  --split-root split_data_v1 \
  -o ml_output/svr_demo
```

## 스케일링과 모델 설정

매 outer fold에서 `StandardScaler`를 해당 fold의 학습 행에만 fit하고, 같은 스케일러로 train/test를 변환합니다. 기본 모델은 다음 설정입니다.

```text
SVR(kernel='rbf', C=10.0, epsilon=0.03, gamma='scale')
```

기본 실행은 위 설정을 사용합니다. `--tune PARAM=VALUE`로 일부 값을 덮어쓸 수 있으며, 자동 탐색이나 추가 CV는 수행하지 않습니다.

```bash
--tune C=100 epsilon=0.1 gamma=0.01 -o ml_output/svr_tuned
```

위 옵션을 기본 실행 명령에 추가합니다.

지원 파라미터는 `C`, `epsilon`, `gamma`, `kernel`입니다. 예를 들어 `--tune C=100`만 지정하면 나머지는 기본값을 사용합니다. 입력한 파라미터는 모든 outer fold에 동일하게 적용되며 결과 파일과 로그에 기록됩니다.

## 평가 및 결과

각 fold의 train R², test R², test RMSE와 5-fold 평균 및 표준편차를 CLI와 `train.log`에 출력합니다.

- `train.log`: 설정, fold별 결과 및 전체 요약
- `fold_metrics.csv`: fold별 지표, 파라미터, 학습/테스트 행 수
- `test_predictions.csv`: CIF 경로, fold, 실제값, 예측값
- `metrics.json`: 입력 설정과 fold별 결과, 전체 요약
