# SIB Structure Features

부분점유 CIF에서 조성·구조 feature를 계산하고, feature CSV 생성과 LightGBM 5-fold 데모를 실행하는 도구입니다. 부분점유 구조를 ordered structure로 바꾸지 않습니다. 조성 계산은 점유율을 반영하며, local bond·angle 통계는 점유율을 가중치로 사용하지 않습니다.

## 설치 및 단일 구조 계산

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/sib-extract structure.cif --json structure.json -o features.json
```

JSON은 선택 사항입니다. 제공하면 Wyckoff metadata 등을 사용합니다. 계산할 수 없는 feature는 JSON에서 `null`로 표시됩니다.

## 전체 디렉토리 계산

`correlation_heatmap.py` CIF 루트를 필수로 받고, CIF 및 JSON을 각 루트 아래에서 재귀 검색합니다. `--json-root`를 생략하면 CIF 루트에서 JSON도 찾습니다. `.cif`/`.json` 확장자는 대소문자를 구분하지 않습니다. CIF와 JSON은 파일 stem이 같거나 지원되는 샘플 ID(예: `074_S1`)가 같으면 매칭됩니다. CIF와 JSON 각각에서 확인한 공간군이 모두 160, 166, 194 중 하나인 구조만 계산합니다.

```bash
.venv/bin/python scripts/correlation_heatmap.py /path/to/cif_root \
  --json-root /path/to/json_root \
  -o correlation_output
```

`--limit 10`은 정렬된 CIF 중 처음 10개만 처리합니다. 결과 디렉토리에는 다음 파일이 생성됩니다.

- `features_all.csv`: 성공적으로 계산된 구조의 전체 feature와 CIF/JSON별 공간군 번호. 히트맵에서 제외된 feature도 포함합니다.
- `spearman_correlation.csv`, `spearman_heatmap.png`, `spearman_heatmap.html`: 히트맵 결과
- `extraction_errors.csv`, `excluded_space_groups.csv`, `warnings.log`, `run_metadata.json`: 계산 오류, 제외한 공간군, 실행 정보

히트맵은 숫자형 feature ID `001–114` 중 `101–106`을 제외하고, 값이 전부 비었거나 상수인 열도 제외합니다. Local feature ID `201` 이상은 히트맵에 포함되지 않습니다. 일부 행만 비어 있는 열은 히트맵 후보로 남고, 상관계수는 유효한 값 쌍으로 계산합니다.

## LightGBM 5-fold 데모

전체 데이터로 `features_all.csv`를 만든 다음 기존 `split_data_v1`의 `summary.json`과 `fold_membership.csv`를 사용합니다. `--data-root`는 feature CSV 생성에 사용한 CIF 루트여야 합니다.

```bash
.venv/bin/python scripts/train_lightgbm_demo.py \
  --features correlation_output/features_all.csv \
  --data-root /path/to/cif_root \
  --split-root split_data_v1 \
  -o ml_output/lightgbm_demo
```

모델은 공간군이 160, 166, 194인 구조만 사용합니다. 입력은 코드에 지정된 12개 feature입니다: `002`, `003`, `008`, `010`, `016`, `107`, `108`, `110`, `111`, `112`, `113`, `114`. `106_O_Na_O_to_O_TM_O_ratio`는 예측 대상이며 입력 feature가 아닙니다. 선택된 입력값이나 타깃이 하나라도 비어 있는 행은 제외됩니다. 따라서 유효 행이 적거나 특정 fold의 테스트 행이 없어지면 학습/평가가 실패할 수 있습니다.

결과는 `train.log`, `fold_metrics.csv`, `test_predictions.csv`, `feature_importance.csv`, `metrics.json`으로 저장됩니다. 그래프는 다음 명령으로 생성할 수 있습니다.

```bash
.venv/bin/python scripts/plot_lightgbm_results.py ml_output/lightgbm_demo
```

## Feature 구성

히트맵에 포함될 수 있는 feature는 `001`, `002`, `003`, `008`, `010`, `016`, `107–114`입니다. 실제 포함 여부는 데이터에 따라 달라지며, 값이 전부 비었거나 상수인 feature는 제외됩니다.

Google Colab에서 Drive 데이터를 연결해 feature 계산, 히트맵, 결측 점검, LightGBM 실행을 하는 절차는 [COLAB_GUIDE.md](COLAB_GUIDE.md)를 참고하세요.
