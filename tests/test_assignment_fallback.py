"""Regression coverage for the assignment OOM fallback mode."""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AssignmentFallbackTests(unittest.TestCase):
    def test_assignment_oom_retry_switches_to_cpu_mode(self):
        source = (ROOT / "yolox" / "models" / "yolo_head_obb_kld.py").read_text()
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_assignments"
        ]
        self.assertGreaterEqual(len(calls), 2)
        fallback_mode = calls[1].args[-1]
        self.assertIsInstance(fallback_mode, ast.Constant)
        self.assertEqual(fallback_mode.value, "cpu")


if __name__ == "__main__":
    unittest.main()
