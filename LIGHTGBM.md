# Google Colab에서 LightGBM 실행하기

이 문서는 이 저장소의 LightGBM 데모 모델을 Google Colab에서 실행하는 방법을 설명합니다.
모델은 `scripts/train_lightgbm_demo.py`를 사용하며, 기존 `split_data_v1`의 5-Fold 분할을
그대로 사용해 `106_`으로 시작하는 target feature를 예측합니다.

## 1. Colab 런타임 준비

Colab에서 새 노트북을 열고 런타임은 `Python 3`를 사용합니다. 이 데모는 CPU로 실행할 수
있으며, GPU 런타임은 필수가 아닙니다.

먼저 Google Drive를 연결합니다.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. 저장소 clone 및 패키지 설치

```bash
!git clone https://github.com/takeone5/sib-cif-ml.git /content/sib-cif-ml
%cd /content/sib-cif-ml
!python -m pip install -e .
```

설치가 정상적으로 끝났는지 확인합니다.

```python
import lightgbm
import pandas
import pymatgen

print('LightGBM:', lightgbm.__version__)
print('pandas:', pandas.__version__)
print('pymatgen:', pymatgen.__version__)
```

## 3. Google Drive 데이터 구조

전체 학습을 위해서는 다음 세 종류의 입력이 필요합니다.

- 전체 CIF 파일 디렉토리
- CIF에 대응하는 JSON metadata 디렉토리(없으면 생략 가능)
- `split_data_v1/summary.json` 및 `split_data_v1/fold_membership.csv`

예를 들어 Google Drive가 다음과 같이 구성되어 있다고 가정합니다.

```text
MyDrive/
├── sib_data/
│   ├── cif/                  # 전체 CIF 파일
│   └── json/                 # 전체 JSON metadata
└── split_data_v1/
    ├── summary.json
    ├── fold_membership.csv
    └── fold_1/
        ├── train/
        └── test/
```

`summary.json`과 `fold_membership.csv`는 저장소의 `split_data_v1`과 같은 형식이어야 합니다.
분할 파일을 Drive에 저장했다면 경로를 먼저 지정합니다.

```python
from pathlib import Path

repo = Path('/content/sib-cif-ml')
cif_root = Path('/content/drive/MyDrive/sib_data/cif')
json_root = Path('/content/drive/MyDrive/sib_data/json')
split_root = Path('/content/drive/MyDrive/split_data_v1')
feature_output = Path('/content/drive/MyDrive/lightgbm_features')
model_output = Path('/content/drive/MyDrive/lightgbm_output')
```

## 4. CIF에서 feature CSV 생성

`train_lightgbm_demo.py`는 feature CSV를 직접 생성하지 않으므로, 먼저
`correlation_heatmap.py`를 실행합니다. 이 명령은 이름이 같은 CIF와 JSON을 자동으로 매칭하고
`features_all.csv`를 생성합니다.

```bash
!python scripts/correlation_heatmap.py \
  "/content/drive/MyDrive/sib_data/cif" \
  --json-root "/content/drive/MyDrive/sib_data/json" \
  -o "/content/drive/MyDrive/lightgbm_features"
```

처음에는 전체 데이터 대신 `--limit 10`을 붙여 경로와 파일 형식을 시험하는 것을 권장합니다.

```bash
!python scripts/correlation_heatmap.py \
  "/content/drive/MyDrive/sib_data/cif" \
  --json-root "/content/drive/MyDrive/sib_data/json" \
  --limit 10 \
  -o "/content/drive/MyDrive/lightgbm_features_test"
```

생성 결과에서 다음 파일을 확인할 수 있습니다.

```python
from pathlib import Path

feature_dir = Path('/content/drive/MyDrive/lightgbm_features')
print((feature_dir / 'features_all.csv').exists())
print((feature_dir / 'extraction_errors.csv').exists())
```

`extraction_errors.csv`에 오류가 기록된 구조는 feature CSV에서 제외됩니다. 오류가 있는 경우
`warnings.log`와 오류 메시지를 먼저 확인합니다.

## 5. LightGBM 5-Fold 실행

다음 셀에서 고정 파라미터 LightGBM 데모를 실행합니다.

```bash
!python scripts/train_lightgbm_demo.py \
  --features "/content/drive/MyDrive/lightgbm_features/features_all.csv" \
  --data-root "/content/drive/MyDrive/sib_data/cif" \
  --split-root "/content/drive/MyDrive/split_data_v1" \
  -o "/content/drive/MyDrive/lightgbm_output"
```

각 인자의 의미는 다음과 같습니다.

- `--features`: 앞 단계에서 생성한 `features_all.csv`
- `--data-root`: feature CSV 생성에 사용한 전체 CIF 루트
- `--split-root`: `summary.json`과 `fold_membership.csv`가 있는 디렉토리
- `-o`: 학습 로그와 결과 파일을 저장할 디렉토리

`--data-root`는 생략할 수도 있지만, CIF 경로와 `fold_membership.csv`의 상대 경로가 다르면
fold 매칭에 실패할 수 있으므로 지정하는 것이 안전합니다.

## 6. 결과 확인

학습이 끝나면 다음 파일이 `lightgbm_output`에 생성됩니다.

```text
train.log             # 전체 실행 로그와 fold별 지표
fold_metrics.csv      # fold별 train/test R², test RMSE
test_predictions.csv  # 각 test 샘플의 실제값과 예측값
feature_importance.csv
metrics.json
```

주요 평균 지표는 `metrics.json`에서 확인할 수 있습니다.

```python
import json

metrics_path = '/content/drive/MyDrive/lightgbm_output/metrics.json'
with open(metrics_path, encoding='utf-8') as file:
    metrics = json.load(file)

print('Mean train R2:', metrics['mean_train_r2'])
print('Mean test R2:', metrics['mean_test_r2'])
print('Mean RMSE:', metrics['mean_rmse'])
```

## 7. 결과 그래프 생성 및 Colab에 표시

```bash
!python scripts/plot_lightgbm_results.py \
  "/content/drive/MyDrive/lightgbm_output"
```

```python
from IPython.display import Image, display
from pathlib import Path

output_dir = Path('/content/drive/MyDrive/lightgbm_output')
for filename in ['actual_vs_predicted.png', 'fold_metrics.png', 'feature_importance.png']:
    print(filename)
    display(Image(filename=str(output_dir / filename)))
```

## 자주 발생하는 문제

### `lightgbm is required` 오류

저장소 루트(`/content/sib-cif-ml`)에서 다음 명령을 다시 실행합니다.

```bash
!python -m pip install -e .
```

### `Could not match ... feature rows to fold membership` 오류

`--data-root`가 feature CSV를 만들 때 사용한 CIF 루트와 같은지 확인합니다. 또한
`fold_membership.csv`의 `relative_path`가 실제 CIF 파일의 상대 경로와 일치해야 합니다.

### `No CIF produced a feature row` 오류

`cif_root` 경로에 `.cif` 파일이 있는지, JSON을 사용하는 경우 `--json-root`가 올바른지
확인합니다. 먼저 `--limit 10`으로 소규모 실행을 해보면 경로 문제를 빠르게 확인할 수 있습니다.

### Colab 세션이 끊긴 경우

Colab 런타임이 초기화되면 `/content` 아래의 clone 및 설치 상태는 사라질 수 있습니다.
Google Drive에 저장한 원본 데이터와 출력 결과는 유지되므로, 런타임 재연결 후 1~2단계를
다시 실행하고 Drive에 있는 `features_all.csv`를 재사용하면 됩니다.
