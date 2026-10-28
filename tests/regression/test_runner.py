"""Fast self-tests for the regression harness, without models or Pillow."""
import copy
import json
import unittest

from run import ROOT, baseline_errors, baseline_in_original_pixels, contract_errors, extract


VALID = '''MRZ (OCR):
P<UTOTESTER<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<<<<
X123456785UTO7408122F1204159ZE184226B<<<<<12
MRZ quadrangle: [(10.0, 20.0), (110.0, 20.0), (110.0, 40.0), (10.0, 40.0)]

Passport data:
  Format:           Td3
  Surname:          TESTER
  Passport number:  X12345678

Check digits:
  Passport number:  OK
  Date of birth:    OK
  Expiration date:  OK
  Personal number:  OK
  Composite:        OK
Checksum status: OK
'''


class RunnerTests(unittest.TestCase):
    def test_extracts_sections_and_ignores_presentation(self):
        actual = extract(VALID, 0)
        self.assertEqual(actual['fields']['Document number'], 'X12345678')
        self.assertEqual(actual['checks']['Document number'], 'OK')
        self.assertFalse(actual['parse_failed'])
        self.assertEqual(len(actual['ocr_lines']), 2)
        self.assertEqual(len(actual['polygon']), 4)
        self.assertEqual(actual, extract(VALID.replace('Passport', 'Document'), 0))
        self.assertEqual([], contract_errors({'id': 'valid', 'category': 'valid'}, actual))

    def test_original_pixel_label_is_parsed(self):
        labelled = VALID.replace(')]\n', ')] (original image pixels)\n')
        self.assertNotEqual(labelled, VALID)
        self.assertEqual(extract(VALID, 0), extract(labelled, 0))

    def test_historical_polygons_are_converted_without_changing_other_data(self):
        for size, crop, expected_points in [
            ([8, 5], False, [[1, 2], [2, 3], [6, 2], [5, 1]]),
            ([8, 5], True, [[2, 2], [3, 3], [4, 2], [3, 1]]),
            ([5, 10], True, [[1, 4], [2, 5], [3, 4], [2, 3]]),
        ]:
            for rotation, point in zip([0, 90, 180, 270], expected_points):
                old = extract(VALID, 0)
                old.update(polygon=[[1, 2]], rotation=rotation, center_crop=crop)
                saved = copy.deepcopy(old)
                converted = baseline_in_original_pixels(old, size, 'attempt')
                self.assertEqual(converted, {**old, 'polygon': [point]})
                self.assertEqual(old, saved)
                self.assertIs(baseline_in_original_pixels(converted, size, 'original'), converted)
        with self.assertRaises(ValueError):
            baseline_in_original_pixels(old, size, 'unknown')

    def test_usage_error_is_not_a_candidate(self):
        actual = extract('', 2)
        self.assertFalse(actual['candidate'])
        self.assertEqual(['exit 2 != 1'], contract_errors(
            {'id': 'usage', 'category': 'usage'}, actual))

    def test_absent_region_contract(self):
        self.assertEqual([], contract_errors({'id': 'white', 'category': 'absent'}, extract('', 1)))
        self.assertIn('candidate presence violates contract', contract_errors(
            {'id': 'white', 'category': 'absent'}, extract(VALID, 2)))

    def test_parse_and_checksum_failure_contracts(self):
        unparseable = 'MRZ (OCR):\nABC\n\nCould not parse MRZ: invalid\nChecksum status: NOT VERIFIED\n'
        self.assertEqual([], contract_errors({'id': 'parse-failed', 'category': 'unverified'},
                                            extract(unparseable, 2)))
        failed = VALID.replace('Checksum status: OK', 'Checksum status: FAILED — OCR errors')
        self.assertEqual([], contract_errors({'id': 'checksum-failed', 'category': 'unverified'},
                                            extract(failed, 2)))

    def test_fields_checks_and_ocr_changes_are_detected(self):
        expected = extract(VALID, 0)
        for key in ('fields', 'checks', 'ocr_lines', 'normalized_lines'):
            actual = copy.deepcopy(expected)
            actual[key] = {} if isinstance(actual[key], dict) else []
            self.assertIn(f'{key} changed', baseline_errors(actual, expected, [640, 480]))

    def test_legacy_empty_personal_number_can_be_omitted(self):
        expected = extract(VALID, 0)
        actual = copy.deepcopy(expected)
        expected['fields']['Personal number'] = ''
        saved = copy.deepcopy(expected)
        self.assertEqual([], baseline_errors(actual, expected, [640, 480]))
        self.assertEqual(expected, saved)
        expected['fields']['Personal number'] = 'SYNTHETIC123'
        self.assertIn('fields changed', baseline_errors(actual, expected, [640, 480]))
        actual['fields']['Personal number'] = 'SYNTHETIC123'
        self.assertEqual([], baseline_errors(actual, expected, [640, 480]))

    def test_polygon_tolerance_and_corner_order(self):
        expected = extract(VALID, 0)
        actual = copy.deepcopy(expected)
        actual['polygon'] = [[x + 1, y - 1] for x, y in reversed(expected['polygon'])]
        self.assertEqual([], baseline_errors(actual, expected, [640, 480]))
        actual['polygon'][0][0] += 100
        self.assertTrue(baseline_errors(actual, expected, [640, 480]))
        actual['polygon'][0][0] = float('nan')
        self.assertIn('invalid polygon', baseline_errors(actual, expected, [640, 480]))

    def test_performance_is_not_an_assertion(self):
        expected = extract(VALID, 0)
        expected['metrics'] = {'elapsed_seconds': 1, 'max_rss_kib': 100}
        actual = copy.deepcopy(expected)
        actual['metrics'] = {'elapsed_seconds': 10, 'max_rss_kib': 1000}
        self.assertEqual([], baseline_errors(actual, expected, [640, 480]))

    def test_repair_rotation_and_crop_are_recorded(self):
        text = VALID.replace('MRZ quadrangle:', 'Image rotation applied: 90°\n'
                             'Center crop applied: yes\nMRZ quadrangle:')
        text = text.replace('Passport data:', 'MRZ after checksum-guided normalization/repair:\n'
                            'REPAIRED\nTEXT\n\nPassport data:')
        text += '  Recovery:         checksum-guided OCR repair was applied\n'
        actual = extract(text, 0)
        self.assertEqual(actual['normalized_lines'], ['REPAIRED', 'TEXT'])
        self.assertEqual(actual['rotation'], 90)
        self.assertTrue(actual['center_crop'])
        self.assertTrue(actual['recovered'])

    def test_baseline_and_known_failures_are_consistent(self):
        cases = json.loads((ROOT / 'cases.json').read_text())
        baseline = json.loads((ROOT / 'baseline.json').read_text())['cases']
        known = json.loads((ROOT / 'known-failures.json').read_text())
        self.assertEqual(len(cases), len({case['id'] for case in cases}))
        self.assertTrue(set(known) <= set(baseline))
        for case in cases:
            with self.subTest(case=case['id']):
                self.assertEqual(case['sha256'], baseline[case['id']]['sha256'])
                errors = contract_errors(case, baseline[case['id']]['result'])
                self.assertEqual(bool(errors), case['id'] in known)


if __name__ == '__main__':
    unittest.main()
