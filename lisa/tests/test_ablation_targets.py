"""CPU-only target contract tests; extract pure helpers without importing GPU dependencies."""
import ast
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'utils/brain_seg_dataset.py'
names = {'CLASSES', 'CLASS_PHRASE', 'NEUTRAL_ANSWER', 'SEG_SENTENCE', 'REASONING_MODES',
         'render_a1', 'validate_reasoning_mode', 'build_answer'}
module = ast.parse(SOURCE.read_text())
nodes = [n for n in module.body if
         (isinstance(n, ast.FunctionDef) and n.name in names) or
         (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
helpers = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), helpers)
build_answer = helpers['build_answer']


class Targets(unittest.TestCase):
    def setUp(self):
        self.row = dict(my_id='test', Q1='localization question', Q2='morph question', A2='morph answer',
                        Q3='stage question', A3='stage answer', Q4='impact question',
                        A4='verbatim conclusion', Status='DO NOT SUBSTITUTE', A1='[]')

    def test_exact_flows_and_mask_independence(self):
        for mode, step in [('localization_only', 1), ('morph_only', 2), ('malig_only', 3)]:
            with self.subTest(mode=mode):
                expected = '\n'.join(f'{k}: {self.row[k]}' for k in
                                     (f'Q{step}', f'A{step}', 'Q4', 'A4'))
                expected += '\n' + helpers['SEG_SENTENCE']
                for present in ({}, dict.fromkeys(helpers['CLASSES'], True),
                                dict.fromkeys(helpers['CLASSES'], False)):
                    actual = build_answer(self.row, present, reasoning_mode=mode)
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual.count('[SEG]'), 3)

    def test_missing_selected_annotation_fails(self):
        for mode, key in [('localization_only', 'Q1'), ('localization_only', 'A1'), ('morph_only', 'Q2'), ('malig_only', 'A3'), ('morph_only', 'A4')]:
            row = dict(self.row, **{key: ' '})
            with self.assertRaisesRegex(ValueError, key):
                build_answer(row, {}, reasoning_mode=mode)

    def test_conflicting_flags_fail(self):
        for flags in [dict(use_q1=True), dict(order='seg_first'), dict(order='a4_seg'), dict(neutral=True)]:
            with self.assertRaises(ValueError):
                build_answer(self.row, {}, reasoning_mode='morph_only', **flags)

    def test_legacy_default(self):
        answer = build_answer(self.row, dict.fromkeys(helpers['CLASSES'], False))
        self.assertIn('Findings: morph answer', answer)
        self.assertIn('Severity: stage answer.', answer)
        self.assertIn('Status (DO NOT SUBSTITUTE)', answer)
        self.assertEqual(answer.count('[SEG]'), 3)


if __name__ == '__main__':
    unittest.main()
