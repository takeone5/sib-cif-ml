# Google Colab 실행 안내

이 안내는 Google Drive의 CIF·JSON 및 `split_data_v1`을 사용해 feature 계산, Spearman 히트맵 생성, LightGBM 5-fold 평가를 순서대로 실행합니다. feature 계산과 히트맵은 `correlation_heatmap.py` 한 번으로 함께 수행됩니다. 학습에는 전체 CIF로 만든 `features_all.csv`를 사용합니다.

## 1. Drive 연결 및 저장소 준비

Colab에서 CPU 런타임을 사용할 수 있습니다. GPU는 필요하지 않습니다.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone https://github.com/takeone5/sib-cif-ml.git /content/sib-cif-ml
%cd /content/sib-cif-ml
!python -m pip install -e .
```

설치 후 Drive의 실제 위치에 맞춰 경로를 지정합니다. CIF와 JSON은 서로 다른 폴더 트리에 둘 수 있습니다. `cif_root`는 `fold_membership.csv`의 상대 경로 기준과 같은 CIF 루트여야 합니다.

```python
from pathlib import Path

repo = Path('/content/sib-cif-ml')
cif_root = Path('/content/drive/MyDrive/sib_data/cif')
json_root = Path('/content/drive/MyDrive/sib_data/json')
split_root = Path('/content/drive/MyDrive/split_data_v1')
feature_output = Path('/content/drive/MyDrive/lightgbm_features')
model_output = Path('/content/drive/MyDrive/lightgbm_output')

assert cif_root.is_dir(), cif_root
assert json_root.is_dir(), json_root
assert (split_root / 'summary.json').is_file()
assert (split_root / 'fold_membership.csv').is_file()
```

## 2. Feature 계산 및 Spearman 히트맵

아래 명령은 CIF 루트와 JSON 루트의 하위 폴더를 재귀 탐색하고, 파일 stem 또는 지원되는 샘플 ID가 같은 JSON을 CIF에 연결합니다. 확장자는 대소문자를 구분하지 않습니다. CIF와 JSON에서 확인한 공간군이 각각 160, 166, 194 중 하나인 구조만 feature를 계산하고 히트맵 데이터에 넣습니다. `--limit`을 지정하지 않아야 전체 데이터가 계산됩니다.

```bash
%cd /content/sib-cif-ml
!python scripts/correlation_heatmap.py "{cif_root}" \
  --json-root "{json_root}" \
  -o "{feature_output}"
```

`features_all.csv`에는 성공적으로 처리한 적격 구조의 전체 feature와 출처별 CIF/JSON 공간군 번호가 저장됩니다. 히트맵에는 별도의 선택 기준이 적용되므로, CSV에 있는 모든 feature가 히트맵에 표시되는 것은 아닙니다. `excluded_space_groups.csv`에는 공간군 조건으로 제외한 구조가, `extraction_errors.csv`에는 계산에 실패한 구조가 기록됩니다. `run_metadata.json`의 `successful_rows`, `error_rows`, `excluded_unsupported_space_group_rows`도 확인합니다.

## 3. 학습 전 결측 및 fold 확인

LightGBM은 공간군 160, 166, 194인 구조 중 코드에 지정된 12개 입력 feature와 `106_` 타깃 값이 모두 있는 행만 사용합니다. 다음 셀에서 각 열의 결측 수, 학습 가능 행 수, fold별 행 수를 확인합니다.

```python
import pandas as pd

features_path = feature_output / 'features_all.csv'
table = pd.read_csv(features_path)
membership = pd.read_csv(split_root / 'fold_membership.csv')

feature_columns = [
    '002_TM_layer_entropy',
    '003_Na_content_per_O2',
    '008_alkali_fraction_in_TM_layer',
    '010_alkaline_earth_fraction_in_TM_layer',
    '016_Ni_Co_Mn_Fe_required_charge',
    '107_AtomicRadius_mean',
    '108_AtomicRadius_avg_dev',
    '110_Electronegativity_avg_dev',
    '111_NdValence',
    '112_CohesiveEnergy',
    '113_is_Prismatic',
    '114_is_2Layer',
]
target_columns = [column for column in table.columns if column.startswith('106_')]
assert len(target_columns) == 1, target_columns
target = target_columns[0]
selected = feature_columns + [target]

print('Feature rows:', len(table))
cif_sg = pd.to_numeric(table['source_cif_space_group_number'], errors='coerce')
json_sg = pd.to_numeric(table['source_json_space_group_number'], errors='coerce')
eligible_mask = cif_sg.isin([160, 166, 194]) & (json_sg.isna() | json_sg.isin([160, 166, 194]))
eligible = table.loc[eligible_mask].copy()
print('Rows excluded for unsupported or unknown space group:', len(table) - len(eligible))
print('Missing values by selected column:')
display(eligible[selected].isna().sum().sort_values(ascending=False).to_frame('missing'))
usable = eligible.dropna(subset=selected).copy()
print('Rows usable for training:', len(usable), '/', len(eligible), 'eligible rows')

# fold_membership.csv contains one test assignment per sample across all folds.
test_rows = membership[membership['split'].astype(str).str.lower() == 'test']
test_paths = test_rows['relative_path'].astype(str).str.replace(chr(92), '/', regex=False)
fold_by_path = dict(zip(test_paths, test_rows['fold']))
known_paths = list(fold_by_path)
def relative_cif_path(source):
    return Path(source).resolve().relative_to(cif_root.resolve()).as_posix()

usable['relative_path'] = usable['source_cif'].map(relative_cif_path)
def find_fold(relative):
    fold = fold_by_path.get(relative)
    if fold is None:
        matches = [path for path in known_paths if relative.endswith('/' + path) or relative == path]
        if len(matches) == 1:
            fold = fold_by_path[matches[0]]
    return fold

usable['fold'] = usable['relative_path'].map(find_fold)
print('Rows without a fold match:', int(usable['fold'].isna().sum()))
print('Usable rows per test fold:')
display(usable.groupby('fold').size().reindex(sorted(membership['fold'].unique()), fill_value=0).to_frame('rows'))
```

`Rows usable for training` is the number of complete rows after missing values are removed. Every test fold needs at least one usable row for the script to run; more rows per fold are needed for meaningful evaluation. If many rows are missing, inspect the columns listed above and the matching source files before rerunning the full extraction. The training script assigns each CIF to the fold where that CIF is marked `test` in `fold_membership.csv`.

## 4. LightGBM 5-fold 실행

학습은 3단계에서 확인한 `features_all.csv` 전체를 입력으로 받습니다. 행 필터링과 fold 배정은 스크립트가 수행합니다.

```bash
%cd /content/sib-cif-ml
!python scripts/train_lightgbm_demo.py \
  --features "{features_path}" \
  --data-root "{cif_root}" \
  --split-root "{split_root}" \
  -o "{model_output}"
```

`train.log`, `fold_metrics.csv`, `test_predictions.csv`, `feature_importance.csv`, `metrics.json`이 저장됩니다.

## 5. 학습 결과 그래프

```bash
%cd /content/sib-cif-ml
!python scripts/plot_lightgbm_results.py "{model_output}"
```

```python
from IPython.display import Image, display

for filename in ['actual_vs_predicted.png', 'fold_metrics.png', 'feature_importance.png']:
    display(Image(filename=str(model_output / filename)))
```

그래프 파일은 `model_output`에 저장됩니다. 실행 결과와 원본 데이터는 Google Drive에 남으므로 Colab 런타임을 다시 연결해도 사용할 수 있습니다.
