# Feature 정의 및 구현 방법

이 문서는 `feature_v1_260718.txt`와 현재 코드의 feature를 대조하기 위한 검증 문서입니다.

## 공통 규칙

- JSON이 제공되면 JSON의 `sites`, `species[].occ`, `wyckoff`, `coords`를 사용합니다.
- CIF는 pymatgen `Structure.from_file()`로 읽고, 부분점유 site를 ordered structure로 변환하지 않습니다.
- 조성 feature는 구조 전체 조성을 O=2가 되도록 정규화합니다.
- Na layer는 Na 점유율이 0보다 큰 site입니다.
- TM layer는 전이금속 점유율이 0보다 큰 site입니다. TM layer 조성에는 해당 site에 함께 존재하는 전이금속, 알칼리금속, 알칼리토금속만 포함합니다.
- [001] Na층 entropy의 Vacancy는 Na층 전체를 1로 두고 `1 - sum(모든 Na층 원소 점유율)`로 계산하며, 음수이면 0으로 제한합니다.
- [201]-[228] local bond/angle feature는 점유율을 무시하고 동일 가중치로 순수한 배위기하 통계를 계산합니다.
- 계산 불가능한 값은 Python `None`, JSON에서는 `null`입니다.

## 조성 Feature [001]-[016]

### [001] `001_Na_layer_entropy`

화학식을 `Na_y X_x TM O2` 꼴로 환산했을 때 Na층의 원소별 분율과 Vacancy의 Shannon
entropy입니다. Na가 포함된 site들의 원소 점유율을 Wyckoff multiplicity와 formula unit
수로 환산하여 합산하고, Na층 전체를 1로 둡니다. 따라서 Vacancy는
`max(0, 1 - sum(모든 Na층 원소 점유율))`입니다. Na층을 점유하는 원소가 여러 개이면
각 원소를 독립적인 entropy 항으로 포함합니다. 이 계산은 P6_3/mmc, R-3m을 포함한
공간군 종류와 관계없이 적용합니다. JSON의 Wyckoff label이 `3a`, `6c`처럼 주어지면
앞의 숫자를 multiplicity로 사용하고, CIF에서 이미 모든 site가 펼쳐진 경우에는 각
site의 multiplicity를 1로 사용합니다.

```text
Na층 분율 = {Na: y, X1: x1, X2: x2, ..., Vacancy: 1 - y - x1 - x2 - ...}
S = -sum(p_i * ln(p_i))
```

자연로그를 사용하며 p는 합산 후 정규화된 비율입니다.

### [002] `002_TM_layer_entropy`

TM site의 전이금속, 알칼리금속, 알칼리토금속 점유율 분포에 대한 Shannon entropy입니다. TM site의 다른 원소와 vacancy는 포함하지 않습니다.

### [003] `003_Na_content_per_O2`

구조 전체 Na 함량을 O=2 기준으로 정규화한 값입니다. 예를 들어 `Na0.84...O4`는 Na 함량 `0.42`가 됩니다.

### [004] `004_TM_layer_element_fractions`

TM layer 조성의 원소별 분율 벡터입니다. 원자번호 순서로 H부터 Og까지 118개 원소 key를 항상 출력하고, 없는 원소는 0입니다.

```text
fraction(element) = included element occupancy / included TM-layer occupancy total
```

### [005] `005_Na_layer_element_fractions`

Na layer 조성의 원소별 분율 벡터입니다. [004]와 같이 원자번호 순서로 H부터 Og까지
118개 원소 key를 항상 출력하고, 없는 원소는 0입니다. Na layer site에 점유된 원소만
포함해 정규화하며 Vacancy는 원소가 아니므로 벡터에서 제외합니다.

```text
fraction(element) = Na-layer element occupancy / total occupied Na-layer element occupancy
```

### [006] `006_Na_TM_entropy_product`

`feature_001 * feature_002`입니다.

### [007] `007_Li_fraction_in_TM_layer`

TM layer 조성 벡터의 Li 분율입니다.

### [008] `008_alkali_fraction_in_TM_layer`

TM layer 조성 벡터에서 Li, Na, K, Rb, Cs, Fr 분율의 합입니다.

### [009] `009_Mg_fraction_in_TM_layer`

TM layer 조성 벡터의 Mg 분율입니다.

### [010] `010_alkaline_earth_fraction_in_TM_layer`

TM layer 조성 벡터에서 Be, Mg, Ca, Sr, Ba, Ra 분율의 합입니다.

### [011] `011_alkali_plus_alkaline_earth_fraction_in_TM_layer`

TM layer 조성 벡터에서 알칼리금속과 알칼리토금속 분율의 합입니다.

### [012], [013] TM 원자 반지름

- [012] `012_TM_atomic_radius_mean`: TM layer 분율로 가중한 pymatgen `Element.atomic_radius` 평균
- [013] `013_TM_atomic_radius_std`: 가중 분산의 제곱근

이온 반지름이 아니라 원자 반지름을 사용합니다.

### [014], [015] TM 전기음성도

- [014] `014_TM_electronegativity_mean`: TM layer 분율로 가중한 pymatgen `Element.X` 평균
- [015] `015_TM_electronegativity_std`: 가중 분산의 제곱근

### [016] `016_Ni_Co_Mn_Fe_required_charge`

Ni, Co, Mn, Fe의 개별 산화수를 정하지 않고, 고정 원소의 산화수와 전하중성으로부터 이 원소군이 담당해야 하는 총 전하 기여량을 계산합니다.

```text
TM_group_total_charge = -sum(composition_i * oxidation_state_i)
                           for i not in {Ni, Co, Mn, Fe}
```

조성은 O/F 음이온 총량이 2가 되도록 정규화합니다. 고정 산화수는 feature 코드에 내장되어 있으며, 목록에 없는 Na는 SIB 기본값 `Na=+1`을 사용합니다. 이 feature는 평균 산화수가 아니라 `원소량 × 산화수`의 합이며, Ni/Co/Mn/Fe가 없거나 고정 원소의 산화수를 알 수 없으면 `null`입니다.

## 전역 구조 Feature [101]-[106]

### [101] `101_space_group_number`

JSON의 `space_group.number`를 우선 사용하고, 없으면 pymatgen 공간군 분석 결과를 사용합니다.

### [102] `102_cell_c_over_a`

```text
c / a
```

### [103] `103_cell_volume_A3`

pymatgen lattice의 unit-cell volume이며 단위는 `Angstrom^3`입니다.

### [104] `104_O_TM_O_distance_A`

제공된 `4-1-1-2_R_spreadsheet_calculator_v3f.py`의 좌표식을 그대로 사용한 TM slab O-TM-O 거리입니다.

`P63/mmc`에서 대표 O/F site의 z가 다음 기준에 가까우면 계산합니다. 허용 오차는 각 기준에 대해 `+-0.05`입니다.

```text
z ~ 0.41: d_TM = 2 * (0.5 - z) * c
z ~ 0.91: d_TM = 2 * (1.0 - z) * c
z ~ 0.08: d_TM = 2 * z * c
z ~ 0.58: d_TM = 2 * (z - 0.5) * c
```

`R-3m`에서는 Na 위치에 따라 다음 식을 사용합니다.

```text
Na at z=0:   d_Na = 2 * (1/3 - z_O) * c
Na at z=0.5: d_Na = 2 * (z_O - 1/6) * c
d_TM = c/3 - d_Na
```

### [105] `105_O_Na_O_distance_A`

Na slab O-Na-O 거리입니다. `P63/mmc`에서는 `d_Na = c/2 - d_TM`을 사용합니다.

### [106] `106_O_Na_O_to_O_TM_O_ratio`

```text
d_O-Na-O / d_O-TM-O
```

지원하지 않는 공간군, O/F site 부재, 잘못된 z 좌표 또는 계산 실패 시 [104]-[106]은 모두 `null`입니다.

## 국소 구조 Feature [201]-[228]

### 배위권 선택

각 Na 또는 TM 중심 site에 대해 pymatgen `Structure.get_neighbors()`를 사용합니다.

- 이웃 탐색 cutoff: `3.2 Angstrom`
- O 또는 F 이웃만 사용
- 거리순 최근접 6개를 배위권으로 선택
- [201]-[228]의 bond length와 angle은 점유율을 무시하고 각 측정값에 동일한 가중치 `1`을 사용

예를 들어 `Mn0.531 Fe0.469` 중심은 중심 가중치 합이 1.0이고, `Na0.19 Ca0.07` site의 Na-O 값에는 Na 점유율 0.19만 반영됩니다.

### [201]-[207] O-Na-O angle

Na가 포함되거나 Na와 같은 crystallographic site를 공동점유하는 site를 하나의 중심으로
취급합니다. 중심 원소별로 중복 계산하지 않고, 해당 site가 완전히 점유되어 있다고
가정하여 최근접 O/F 이웃 6개에서 가능한 15개 이웃쌍의 각도를 계산합니다. 각도는
중심 원소, 공동점유 원소, 공공, O/F 점유율과 관계없이 동일한 가중치로 통계 처리합니다.

```text
angle = acos[(r_i-r_c) dot (r_j-r_c) / (|r_i-r_c| |r_j-r_c|)]
```

각도 단위는 degree입니다. 배위 O/F 이웃이 6개 미만이면 해당 중심의 angle feature 계산에서 제외하며, 유효한 중심이 하나도 없으면 `null`입니다.

- [201] `201_O_Na_O_angle_deg_arithmetic_mean`: 가중 산술평균
- [202] `202_O_Na_O_angle_deg_harmonic_mean`: 가중 조화평균
- [203] `203_O_Na_O_angle_deg_geometric_mean`: 가중 기하평균
- [204] `204_O_Na_O_angle_deg_max`: 최대값
- [205] `205_O_Na_O_angle_deg_min`: 최소값
- [206] `206_O_Na_O_angle_deg_std`: 가중 표준편차
- [207] `207_O_Na_O_angle_deg_skew`: 가중 3차 표준화 중심모멘트

### [208]-[214] O-TM-O angle

[201]-[207]과 동일한 방식으로 TM 중심에서 계산합니다. TM이 포함된 crystallographic
site를 하나의 중심으로 취급하고, 중심 TM의 종류나 공동점유 원소를 구분하지 않습니다.
중심 site와 O/F 이웃이 완전히 점유되어 있다고 가정하여 점유율을 각도 가중치에
사용하지 않으며, 모든 각도를 동일한 가중치로 통계 처리합니다.

- [208] `208_O_TM_O_angle_deg_arithmetic_mean`
- [209] `209_O_TM_O_angle_deg_harmonic_mean`
- [210] `210_O_TM_O_angle_deg_geometric_mean`
- [211] `211_O_TM_O_angle_deg_max`
- [212] `212_O_TM_O_angle_deg_min`
- [213] `213_O_TM_O_angle_deg_std`
- [214] `214_O_TM_O_angle_deg_skew`

### [215]-[228] Na-O 및 TM-O bond length

Na 또는 TM 중심과 최근접 O/F 6개 사이의 bond length를 계산합니다. 이웃이 6개 미만인 중심은 제외하며, 유효한 중심이 하나도 없으면 `null`입니다.
Na층 site와 TM site는 완전히 점유되어 있다고 가정하고, O/F 이웃도 점유율을 무시합니다. 따라서 각 bond length에 동일한 가중치 `1`을 사용하여 순수한 배위기하 통계를 계산합니다.

- [215] `215_Na_O_bond_length_A_arithmetic_mean`
- [216] `216_TM_O_bond_length_A_arithmetic_mean`
- [217] `217_Na_O_bond_length_A_harmonic_mean`
- [218] `218_TM_O_bond_length_A_harmonic_mean`
- [219] `219_Na_O_bond_length_A_geometric_mean`
- [220] `220_TM_O_bond_length_A_geometric_mean`
- [221] `221_Na_O_bond_length_A_max`
- [222] `222_TM_O_bond_length_A_max`
- [223] `223_Na_O_bond_length_A_min`
- [224] `224_TM_O_bond_length_A_min`
- [225] `225_Na_O_bond_length_A_std`
- [226] `226_TM_O_bond_length_A_std`
- [227] `227_Na_O_bond_length_A_skew`
- [228] `228_TM_O_bond_length_A_skew`

출력 key는 feature 번호와 측정 대상 및 통계 방법을 포함합니다.

## 검증 시 확인할 차이점

1. [201]-[228]은 feature 번호와 측정 대상 및 통계 방법을 조합한 key를 사용합니다. 예를 들어 [201]은 `201_O_Na_O_angle_deg_arithmetic_mean`, [219]는 `219_Na_O_bond_length_A_geometric_mean`입니다.
2. 현재 local geometry는 최근접 6개 O/F를 고정 배위권으로 사용합니다. cutoff 안에 6개가 없는 중심은 제외하고, 유효한 중심이 하나도 없는 feature는 `null`입니다.
