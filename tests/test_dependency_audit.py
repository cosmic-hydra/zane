from __future__ import annotations

import unittest

from dependency_audit import audit_missing_modules, format_missing_modules


class DependencyAuditTests(unittest.TestCase):
    def test_audit_missing_modules_reports_absent_packages(self):
        missing = audit_missing_modules(["this_module_should_not_exist_12345", "requests"])

        self.assertIn("this_module_should_not_exist_12345", missing)
        self.assertNotIn("requests", missing)

    def test_format_missing_modules(self):
        self.assertEqual(format_missing_modules(["torch", "pandas"]), "torch, pandas")
        self.assertEqual(format_missing_modules([]), "none")


if __name__ == "__main__":
    unittest.main()
