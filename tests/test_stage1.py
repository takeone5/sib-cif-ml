import json
import unittest
from pathlib import Path

from sib_features import FeatureExtractor


ROOT = Path(__file__).parents[1]


class StageOneTest(unittest.TestCase):
    def test_disordered_example(self):
        result = FeatureExtractor.from_files(
            ROOT / "074_S1_NPD_P2-NaxCayFe0.5Mn0.5O2.cif", ROOT / "074_S1_NPD.json"
        ).extract()
        self.assertAlmostEqual(result["003_Na_content_per_O2"], 0.42, places=2)
        self.assertEqual(result["101_space_group_number"], 194)
        self.assertAlmostEqual(result["102_cell_c_over_a"], 3.851, places=2)
        self.assertAlmostEqual(result["016_Ni_Co_Mn_Fe_required_charge"], 3.44, places=2)
        self.assertAlmostEqual(result["104_O_TM_O_distance_A"], 1.922, places=2)
        self.assertAlmostEqual(result["105_O_Na_O_distance_A"], 3.699, places=2)
        self.assertAlmostEqual(
            result["106_O_Na_O_to_O_TM_O_ratio"],
            result["105_O_Na_O_distance_A"] / result["104_O_TM_O_distance_A"],
            places=6,
        )
        self.assertAlmostEqual(result["001_Na_layer_entropy"], 0.8939041632, places=8)
        self.assertAlmostEqual(sum(result["004_TM_layer_element_fractions"].values()), 1.0)
        self.assertEqual(len(result["005_Na_layer_element_fractions"]), 118)
        self.assertAlmostEqual(sum(result["005_Na_layer_element_fractions"].values()), 1.0)
        self.assertAlmostEqual(result["005_Na_layer_element_fractions"]["Na"], 0.42 / 0.49, places=8)
        self.assertAlmostEqual(result["005_Na_layer_element_fractions"]["Ca"], 0.07 / 0.49, places=8)
        self.assertTrue(json.dumps(result, allow_nan=False))
        self.assertNotIn("oxidation_state_inference", result)
        self.assertAlmostEqual(result["215_Na_O_bond_length_A_arithmetic_mean"], 2.502, places=2)
        self.assertAlmostEqual(result["216_TM_O_bond_length_A_arithmetic_mean"], 1.940, places=2)
        required_prefixes = [
            "201_O_Na_O_angle_deg_", "208_O_TM_O_angle_deg_", "215_Na_O_bond_length_A_",
            "216_TM_O_bond_length_A_",
        ]
        for prefix in required_prefixes:
            self.assertTrue(any(key.startswith(prefix) for key in result), prefix)


if __name__ == "__main__":
    unittest.main()
