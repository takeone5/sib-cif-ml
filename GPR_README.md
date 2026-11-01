# Gaussian Process 5-fold 데모

이 문서는 `GaussianProcessRegressor` 기반 5-fold 교차 검증 스크립트의 실행 방법과 결과물을 설명합니다. feature 선택, 타겟 자동 선택, split 읽기, 지원 공간군 필터링은 LightGBM 데모와 같은 기준을 사용합니다.

## 실행

먼저 전체 데이터의 `features_all.csv`와 기존 `split_data_v1`의 `summary.json`, `fold_membership.csv`를 준비합니다. `--data-root`에는 feature CSV 생성에 사용한 CIF 루트를 지정합니다.

```bash
.venv/bin/python scripts/train_gaussian_process_demo.py \
  --features correlation_output/features_all.csv \
  --data-root /path/to/cif_root \
  --split-root split_data_v1 \
  -o ml_output/gaussian_process_demo
```

## Google Colab에서 실행

Colab에서도 실행할 수 있습니다. 저장소를 런타임에 준비하고 `pip install -e .`을 실행하면 GPR에 필요한 scikit-learn과 scipy를 포함한 프로젝트 의존성이 설치됩니다. 아래 예시는 입력 CSV와 split, 결과를 Google Drive에 두는 방법입니다.

먼저 Colab 셀에서 Drive를 연결하고 저장소를 준비합니다.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone https://github.com/takeone5/sib-cif-ml.git /content/sib-cif-ml
%cd /content/sib-cif-ml
!python -m pip install -e .
```

Drive에 이미 만들어 둔 전체 `features_all.csv`와 `split_data_v1` 경로를 지정합니다. `cif_root`는 feature CSV 생성에 쓴 CIF 루트이며, `source_cif`와 `fold_membership.csv`의 경로를 연결하는 데 사용합니다.

```python
from pathlib import Path

repo = Path('/content/sib-cif-ml')
cif_root = Path('/content/drive/MyDrive/sib_data/cif')
features_path = Path('/content/drive/MyDrive/lightgbm_features/features_all.csv')
split_root = Path('/content/drive/MyDrive/split_data_v1')
model_output = Path('/content/drive/MyDrive/gaussian_process_output')

assert features_path.is_file(), features_path
assert cif_root.is_dir(), cif_root
assert (split_root / 'summary.json').is_file()
assert (split_root / 'fold_membership.csv').is_file()
```

학습 명령은 다음과 같습니다.

```python
%cd /content/sib-cif-ml
%run scripts/train_gaussian_process_demo.py --features "$features_path" --data-root "$cif_root" --split-root "$split_root" -o "$model_output"
```

Colab의 `%run`으로 실행하면 학습이 끝난 뒤 생성된 그래프가 셀 출력에도 표시됩니다. 터미널이나 `!python` 서브프로세스로 실행한 경우에는 그래프 파일만 저장되므로, 노트북에 표시하려면 아래처럼 별도 셀에서 출력할 수 있습니다.

```python
from IPython.display import Image, display

for name in ['parity_plot.png', 'uncertainty_ribbon.png', 'partial_dependence.png']:
    display(Image(filename=str(model_output / name)))
```

이 스크립트는 CIF 자체를 다시 읽지 않고 feature CSV와 split 파일로 학습합니다. 따라서 CIF 루트에는 `source_cif` 경로와 split 상대 경로를 맞추는 목적으로 접근할 수 있어야 합니다. GPR 학습은 GPU를 사용하지 않으며, 계산량은 학습 행 수에 따라 크게 늘어날 수 있습니다. 실행 결과는 Drive의 `gaussian_process_output`에 남습니다.

기본 feature는 LightGBM 데모와 같은 12개입니다. `--add-features`와 `--exclude-features`로 feature를 조정할 수 있으며, 타겟은 `106_`으로 시작하는 유일한 컬럼을 자동으로 선택합니다. 타겟이 여러 개거나 다른 타겟을 지정하려면 `--target`을 사용합니다.

```bash
.venv/bin/python scripts/train_gaussian_process_demo.py \
  --exclude-features 002_TM_layer_entropy \
  --add-features 001_example_feature 115_example_feature \
  --target 106_O_Na_O_to_O_TM_O_ratio
```

## 학습과 누수 방지

각 fold에서 `StandardScaler`를 해당 fold의 `X_train`에만 fit합니다. 그 스케일러로 `X_train`과 `X_test`를 transform하므로 테스트 데이터 통계가 스케일링에 반영되지 않습니다.

GPR 커널은 아래 복합 커널을 사용하며, 각 fold의 학습 데이터에 맞춰 하이퍼파라미터를 최적화합니다.

```text
ConstantKernel(1.0, (1e-2, 1e2))
* RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2))
+ WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
```

모델 설정은 `n_restarts_optimizer=10`, `normalize_y=True`, `random_state=42`입니다. 각 fold에서 train/test R², test RMSE, 최적화된 커널(`model.kernel_`)을 로깅합니다. 테스트 예측은 평균과 표준편차를 반환하며, 실제 절대 오차와 예측 표준편차 사이의 Pearson 상관계수와 Spearman 순위 상관계수(ρ)를 계산합니다. 상관계수 계산에 필요한 표본이나 변동성이 부족하면 값은 `undefined`로 표시됩니다.

5개 fold의 train R², test R², test RMSE는 평균과 표준편차로 요약합니다.

## 결과 파일

기본 출력 디렉토리는 `ml_output/gaussian_process_demo`이며 `-o`로 변경할 수 있습니다.

- `train.log`: 실행 설정, fold별 지표와 최적화 커널, Pearson 및 Spearman 불확실성 상관계수, 전체 요약
- `fold_metrics.csv`: fold별 지표, 두 불확실성 상관계수, 최적화된 커널, 학습/테스트 행 수
- `test_predictions.csv`: CIF 경로, fold, 실제값, 예측값, 예측 표준편차, 절대 오차
- `metrics.json`: feature와 타겟, 모델 설정, fold별 지표 및 전체 요약
- `parity_plot.png`: 실제값 대 예측값과 95% 예측 구간
- `uncertainty_ribbon.png`: 예측값 기준으로 정렬한 불확실성 띠와 실제값
- `partial_dependence.png`: 각 입력 feature의 1D partial dependence 곡선과 95% 예측 구간
