from __future__ import annotations

import unittest

from drug_discovery.evaluation.swissadme_proxy import SwissADMEProxy


class SwissADMEProxyTests(unittest.TestCase):
    def setUp(self):
        self.proxy = SwissADMEProxy(use_rdkit=False)

    def test_predict_returns_rich_profile(self):
        result = self.proxy.predict("CCO")

        self.assertEqual(result.smiles, "CCO")
        self.assertIn("logP", result.admet_profile)
        self.assertIn("gi_absorption", result.admet_profile)
        self.assertIn("lipinski", result.rule_checks)
        self.assertIn("veber", result.rule_checks)
        self.assertGreaterEqual(result.violations, 0)
        self.assertIsInstance(result.developable, bool)

    def test_predict_rejects_empty_smiles(self):
        with self.assertRaises(ValueError):
            self.proxy.predict("  ")


if __name__ == "__main__":
    unittest.main()