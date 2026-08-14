# SIB Structure Features

부분점유(disordered) CIF를 ordered structure로 변환하지 않고, Rietveld 점유율을 가중치로 사용해 feature를 계산하는 프로젝트입니다.

## 현재 구현 범위

`feature_v1_260718.txt`에서 현재 구현된 feature를 제공합니다.

구현상 정의:

- Na entropy는 공간군과 관계없이 Na층의 조성을 `Na_y X_x TM O2`로 환산하고, Na층 전체를 1로 둔 원소별 분율과 Vacancy를 사용합니다. Na site를 여러 원소가 공동점유하면 각 원소를 별도 entropy 항으로 포함합니다.
- TM layer는 전이금속 원소가 한 종 이상 포함된 crystallographic site이며, 그 site의 전이금속·알칼리·알칼리토금속 점유만 TM 조성에 포함합니다.
- 조성의 Na 함량은 구조 조성을 O=2로 정규화합니다.
- 원자 반지름과 전기음성도는 이온값이 아닌 pymatgen `Element`의 원자 물성으로 TM-layer 점유율 가중 평균과 표준편차를 냅니다.
- JSON이 제공되면 CIF에서 잃기 쉬운 Wyckoff label을 JSON에서 사용합니다. JSON 없이 CIF만 주면 Wyckoff 기반 feature는 제한됩니다.
- [104]-[106]은 제공된 기존 스프레드시트의 `P63/mmc`, `R-3m` 좌표식을 사용합니다. 지원하지 않는 공간군 또는 비정상 z 좌표는 세 feature 모두 `null`입니다.
- [201]-[228]은 Na층 site 및 TM 중심마다 pymatgen의 주기적 이웃에서 가장 가까운 O/F 6개를 배위권으로 선택합니다. Na층과 TM site의 공동점유 원소는 하나의 중심 site로 취급하며, [201]-[228]의 각도와 결합길이는 점유율을 무시하고 동일 가중치로 계산합니다.
- 배위 O/F 6개가 확보되지 않은 중심은 local 통계에서 제외하며, 유효한 중심이 없으면 해당 feature는 `null`입니다.

## 설치 및 실행

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/sib-extract /path/to/input.cif --json /path/to/input.json -o features_stage1.json

# 산화수 추정이 필요할 때만 별도 출력
```

출력 JSON의 key는 feature 번호와 의미를 함께 담아 후속 batch extractor에서 그대로 병합할 수 있도록 했습니다.

각 feature의 정의, 수식, 부분점유 처리와 검증 포인트는 [FEATURE_METHODS.md](FEATURE_METHODS.md)에 정리했습니다.

외부 CIF 디렉토리의 feature 상관 heatmap은 `scripts/correlation_heatmap.py`로 생성할 수 있습니다. 저장소에는 원본 데이터를 포함할 필요가 없습니다. CIF와 JSON이 같은 디렉토리 트리 아래 있으면 `data_root`만 지정하고, 서로 다른 위치에 있으면 `--json-root`를 추가합니다. 이 작업은 GPU가 필요하지 않지만, 많은 구조의 local neighbor 계산이 CPU를 사용할 수 있으므로 먼저 `--limit 10`으로 시험하는 것을 권장합니다.

```bash
.venv/bin/python scripts/correlation_heatmap.py /path/to/cif_root \
  --limit 10 \
  -o correlation_output_pilot

# CIF와 JSON이 서로 다른 디렉토리에 있는 경우
.venv/bin/python scripts/correlation_heatmap.py /path/to/cif_root \
  --json-root /path/to/json_root \
  -o correlation_output

# 전체 실행
.venv/bin/python scripts/correlation_heatmap.py /path/to/cif_root \
  -o correlation_output
```

`cif_root`와 `json_root` 아래의 파일은 CIF stem과 JSON stem이 같거나, 현재 지원하는
샘플 ID 형식(예: `074_S1`)이 같으면 자동으로 매칭됩니다. JSON이 없는 CIF도 처리되며,
이 경우 Wyckoff metadata를 사용하는 feature는 제한될 수 있습니다.

결과 파일은 `features_all.csv`, `pearson_correlation.csv`, `pearson_heatmap.png`, `extraction_errors.csv`, `run_metadata.json`입니다. 피쳐 추출 결과에는 모든 피쳐가 포함되며, Pearson 상관계수와 heatmap에서는 [004], [005], [104], [105], [106]만 제외됩니다. 산화수 추정은 heatmap 대상에서 제외됩니다.
