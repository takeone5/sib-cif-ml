# Feature 정의 및 구현 방법

이 문서는 `feature_v1_260718.txt`와 현재 코드의 feature를 대조하기 위한 검증 문서입니다.

## 공통 규칙

- JSON이 제공되면 JSON의 `sites`, `species[].occ`, `wyckoff`, `coords`를 사용합니다.
- CIF는 pymatgen `Structure.from_file()`로 읽고, 부분점유 site를 ordered structure로 변환하지 않습니다.
- 조성 feature는 구조 전체 조성을 O=2가 되도록 정규화합니다.
- Na layer는 Na 점유율이 0보다 큰 site입니다.
- TM layer는 전이금속 또는 란타넘족(Tb 포함) 점유율이 0보다 큰 site입니다. TM layer 조성에는 해당 site에 함께 존재하는 전이금속·란타넘족, 알칼리금속, 알칼리토금속만 포함합니다.
- 전이금속 판정은 pymatgen의 `Element.is_transition_metal`을 사용하되, pymatgen에서 전이금속으로 분류하지 않는 La–Lu 란타넘족을 TM-like 중심으로 추가 포함합니다.
- [001] Na층 entropy의 Vacancy는 Na층 전체를 1로 두고 `1 - sum(모든 Na층 원소 점유율)`로 계산하며, 음수이면 0으로 제한합니다.
- [201], [206]-[207], [208], [213]-[214], [215]-[220], [225]-[228] local bond/angle feature는 점유율을 무시하고 동일 가중치로 순수한 배위기하 통계를 계산합니다.
- 계산 불가능한 값은 Python `None`, JSON에서는 `null`입니다.

## 조성 Feature [001]-[016], [107]-[112]

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

### [006] `006_Na_TM_entropy_sum`

`feature_001 + feature_002`입니다. 엔트로피 관련 출력값([001], [002], [006])은
소수점 이하 셋째 자리까지 반올림합니다. 합산은 반올림 전 엔트로피 값으로 수행합니다.

### [007] `007_Li_fraction_in_TM_layer`

TM layer 조성 벡터의 Li 분율입니다.

### [008] `008_alkali_fraction_in_TM_layer`

TM layer 조성 벡터에서 Li, Na, K, Rb, Cs, Fr 분율의 합입니다.

### [009] `009_Mg_fraction_in_TM_layer`

TM layer 조성 벡터의 Mg 분율입니다.

### [010] `010_alkaline_earth_fraction_in_TM_layer`

TM layer 조성 벡터에서 Be, Mg, Ca, Sr, Ba, Ra 분율의 합입니다.

### [016] `016_Ni_Co_Mn_Fe_required_charge`

Ni, Co, Mn, Fe의 개별 산화수를 정하지 않고, 고정 원소의 산화수와 전하중성으로부터 이 원소군이 담당해야 하는 총 전하 기여량을 계산합니다.

```text
TM_group_total_charge = -sum(composition_i * oxidation_state_i)
                           for i not in {Ni, Co, Mn, Fe}
```

조성은 O/F 음이온 총량이 2가 되도록 정규화합니다. 고정 산화수는 feature 코드에 내장되어 있으며, 목록에 없는 Na는 SIB 기본값 `Na=+1`을 사용합니다. 이 feature는 평균 산화수가 아니라 `원소량 × 산화수`의 합이며, Ni/Co/Mn/Fe가 없거나 고정 원소의 산화수를 알 수 없으면 `null`입니다.

### Matminer 기반 조성 Feature [107]-[112]

TM layer 조성의 원소별 분율을 가중치로 사용해 Matminer의 원소 물성값 통계를 계산합니다.
`AtomicRadius`, `Electronegativity`, `NdValence`는 `MagpieData`에서, `CohesiveEnergy`는
`CohesiveEnergyData`에서 가져옵니다. 해당 물성값이 없는 원소는 그 물성의 평균에서 제외하며,
유효한 원소가 없으면 `null`입니다.

- [107] `107_AtomicRadius_mean`: AtomicRadius의 TM 조성비 가중 평균
- [108] `108_AtomicRadius_avg_dev`: AtomicRadius의 TM 조성비 가중 평균 절대편차
- [109] `109_Electronegativity_mean`: Electronegativity의 TM 조성비 가중 평균
- [110] `110_Electronegativity_avg_dev`: Electronegativity의 TM 조성비 가중 평균 절대편차
- [111] `111_NdValence`: NdValence의 TM 조성비 가중 평균
- [112] `112_CohesiveEnergy`: CohesiveEnergy의 TM 조성비 가중 평균

## 전역 구조 Feature [101]-[106], [113]-[114]

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

`R3m`(국제 공간군 번호 160)에서는 Wyckoff position이 `3a`인 O/F site 두 개의
`(0, 0, z)` 좌표를 사용합니다. 두 좌표가 다음 조건 중 하나를 만족할 때 계산합니다.

1. 두 z 좌표가 부호만 반대이고 절댓값이 같으면, 양수 좌표를 `z`로 둡니다. `z > 1/3`
   일 때 다음 식을 사용합니다.

   ```text
   d_O-TM-O = 2 * (z - 1/3) * c
   d_O-Na-O = c/3 - d_O-TM-O
   ```

2. 두 z 좌표가 모두 양수이고 `z1 > z2`, `z1 + z2 = 1`이면 다음 식을 사용합니다.

   ```text
   d_O-TM-O = (z1 - z2) * c
   d_O-Na-O = c/3 - d_O-TM-O
   ```

조건을 만족하는 두 개의 O/F `3a` site를 찾지 못하거나 좌표 조건을 만족하지 않으면
[104]–[106]은 `null`입니다.

### [105] `105_O_Na_O_distance_A`

Na slab O-Na-O 거리입니다. `P63/mmc`에서는 `d_Na = c/2 - d_TM`을 사용합니다.

### [106] `106_O_Na_O_to_O_TM_O_ratio`

```text
d_O-Na-O / d_O-TM-O
```

지원하지 않는 공간군, O/F site 부재, 잘못된 z 좌표 또는 계산 실패 시 [104]-[106]은 모두 `null`입니다.

### [113] `113_is_Prismatic`

공간군 번호에 따른 prismatic 구조 여부입니다.

- 공간군 166 (`R-3m`): `0`
- 공간군 160 (`R3m`): `1`
- 공간군 194 (`P6_3/mmc`): `1`
- 그 외 공간군 또는 공간군 번호를 확인할 수 없는 경우: `null`

### [114] `114_is_2Layer`

공간군 번호에 따른 2-layer 구조 여부입니다.

- 공간군 166 (`R-3m`): `0`
- 공간군 160 (`R3m`): `0`
- 공간군 194 (`P6_3/mmc`): `1`
- 그 외 공간군 또는 공간군 번호를 확인할 수 없는 경우: `null`

## 국소 구조 Feature [201], [206]-[208], [213]-[220], [225]-[228]

### 배위권 선택

각 Na 또는 TM 중심 site에 대해 pymatgen `Structure.get_neighbors()`를 사용합니다.

- 이웃 탐색 cutoff: `3.2 Angstrom`
- O 또는 F 이웃만 사용
- 거리순 최근접 6개를 배위권으로 선택
- [201], [206]-[208], [213]-[220], [225]-[228]의 bond length와 angle은 점유율을 무시하고 각 측정값에 동일한 가중치 `1`을 사용
- 이 feature들은 CIF의 평균 결정구조에서 얻은 이상적인 6배위 기하의 통계이며, 원자 점유율이나 vacancy 확률을 반영한 통계가 아닙니다.
- O/F site는 O 또는 F 점유율이 `0`보다 크면 이웃으로 포함하고, 점유율이 `0`인 site만 제외합니다. 따라서 `O0.7 Vacancy0.3`과 같은 부분점유 site도 완전히 존재하는 O 이웃처럼 취급되며, vacancy 확률은 각도와 결합길이의 가중치에 반영되지 않습니다.
- 유효한 O/F 이웃이 6개 미만인 중심은 local feature 계산에서 제외합니다. 6개가 확보되면 부분점유 여부와 관계없이 6개 이웃을 사용합니다.

예를 들어 `Mn0.531 Fe0.469` 중심과 `Na0.19 Ca0.07` 중심은 각각 하나의 완전 점유 중심으로 취급되며, 두 경우 모두 점유율은 각도와 결합길이 통계의 가중치에 반영되지 않습니다.

### [201], [206]-[207] O-Na-O angle

Na가 포함되거나 Na와 같은 crystallographic site를 공동점유하는 site를 하나의 중심으로
취급합니다. 중심 원소별로 중복 계산하지 않고, 해당 site가 완전히 점유되어 있다고
가정하여 최근접 O/F 이웃 6개에서 가능한 15개 이웃쌍의 각도를 계산합니다. 각도는
중심 원소, 공동점유 원소, 공공, O/F 점유율과 관계없이 동일한 가중치로 통계 처리합니다.

```text
angle = acos[(r_i-r_c) dot (r_j-r_c) / (|r_i-r_c| |r_j-r_c|)]
```

각도 단위는 degree입니다. 배위 O/F 이웃이 6개 미만이면 해당 중심의 angle feature 계산에서 제외하며, 유효한 중심이 하나도 없으면 `null`입니다. 조화평균과 기하평균은 제외하고 산술평균, 최대값, 최소값만 출력합니다.
각도 feature의 출력값은 소수점 이하 둘째 자리까지 반올림합니다. 각도 계산 자체는 반올림 전 값으로 수행합니다.

- [201] `201_O_Na_O_angle_deg_arithmetic_mean`: 산술평균
- [206] `206_O_Na_O_angle_deg_std`: 표준편차
- [207] `207_O_Na_O_angle_deg_skew`: skew

### [208]-[214] O-TM-O angle

[201], [206]-[207]과 동일한 방식으로 TM 중심에서 계산합니다. TM이 포함된 crystallographic
site를 하나의 중심으로 취급하고, 중심 TM의 종류나 공동점유 원소를 구분하지 않습니다.
중심 site와 O/F 이웃이 완전히 점유되어 있다고 가정하여 점유율을 각도 가중치에
사용하지 않으며, 모든 각도를 동일한 가중치로 통계 처리합니다. 즉 O/F 점유율이
0보다 크기만 하면 해당 이웃을 포함하고, `O0.7 Vacancy0.3`과 같은 부분점유도
완전 점유 이웃처럼 취급합니다. O/F 점유율이 0인 site만 제외합니다.
각도 feature의 출력값은 소수점 이하 둘째 자리까지 반올림합니다. 각도 계산 자체는 반올림 전 값으로 수행합니다.

- [208] `208_O_TM_O_angle_deg_arithmetic_mean`
- [213] `213_O_TM_O_angle_deg_std`
- [214] `214_O_TM_O_angle_deg_skew`

### [215]-[228] Na-O 및 TM-O bond length

Na 또는 TM 중심과 최근접 O/F 6개 사이의 bond length를 계산합니다. 이웃이 6개 미만인 중심은 제외하며, 유효한 중심이 하나도 없으면 `null`입니다.
Na층 site와 TM site는 완전히 점유되어 있다고 가정하고, O/F 이웃도 점유율을 무시합니다. 따라서 각 bond length에 동일한 가중치 `1`을 사용하여 순수한 배위기하 통계를 계산합니다.
Bond length의 단위는 Å이며, 통계 계산 전에 소수점 이하 셋째 자리까지 반올림합니다. 따라서 0.001 Å 단위까지 같은 값은 동일한 결합길이로 취급하며, 그보다 작은 부동소수점 차이는 무시합니다.

- [215] `215_Na_O_bond_length_A_arithmetic_mean`
- [216] `216_TM_O_bond_length_A_arithmetic_mean`
- [217] `217_Na_O_bond_length_A_harmonic_mean`
- [218] `218_TM_O_bond_length_A_harmonic_mean`
- [219] `219_Na_O_bond_length_A_geometric_mean`
- [220] `220_TM_O_bond_length_A_geometric_mean`
- [225] `225_Na_O_bond_length_A_std`
- [226] `226_TM_O_bond_length_A_std`
- [227] `227_Na_O_bond_length_A_skew`
- [228] `228_TM_O_bond_length_A_skew`

출력 key는 feature 번호와 측정 대상 및 통계 방법을 포함합니다.

## 검증 시 확인할 차이점

1. [201], [206]-[208], [213]-[220], [225]-[228]은 feature 번호와 측정 대상 및 통계 방법을 조합한 key를 사용합니다. 예를 들어 [201]은 `201_O_Na_O_angle_deg_arithmetic_mean`, [219]는 `219_Na_O_bond_length_A_geometric_mean`입니다.
2. 현재 local geometry는 최근접 6개 O/F를 고정 배위권으로 사용합니다. cutoff 안에 6개가 없는 중심은 제외하고, 유효한 중심이 하나도 없는 feature는 `null`입니다.
