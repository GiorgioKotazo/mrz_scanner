#!/usr/bin/env python3
"""Black-box model regression tests. Standard library only; no network access."""
import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
MODELS = ('mrz_detection_20250222_fp32.onnx', 'mrz_recognition_20250221_fp32.onnx')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(stdout, code):
    """Inspect semantic sections, not a golden copy of the entire stdout."""
    lines = stdout.splitlines()
    ocr = []
    if 'MRZ (OCR):' in lines:
        for line in lines[lines.index('MRZ (OCR):') + 1:]:
            if not re.fullmatch('[A-Z0-9<]+', line):
                break
            ocr.append(line)
    result = dict(exit=code, candidate='MRZ (OCR):' in lines, ocr_lines=ocr,
                  fields={}, checks={}, polygon=[], rotation=0, center_crop=False,
                  recovered=False, parse_failed='Could not parse MRZ:' in stdout)
    section = None
    for line in lines:
        if line in ('Passport data:', 'Document data:'):
            section = 'fields'
        elif line == 'Check digits:':
            section = 'checks'
        elif line.startswith('Checksum status:'):
            result['checksum_status'] = line.split(':', 1)[1].strip().split(' —')[0]
        elif line.startswith('MRZ quadrangle:'):
            result['polygon'] = [[float(x), float(y)] for x, y in
                                 re.findall(r'\(([-\d.]+), ([-\d.]+)\)', line)]
        elif line.startswith('Image rotation applied:'):
            result['rotation'] = int(re.search(r'(\d+)°', line)[1])
        elif line == 'Center crop applied: yes':
            result['center_crop'] = True
        elif line.startswith('  Recovery:'):
            result['recovered'] = True
        elif section and line.startswith('  ') and ':' in line:
            key, value = line.strip().split(':', 1)
            key = key.replace('Passport number', 'Document number')
            result[section][key] = value.strip()
    marker = 'MRZ after checksum-guided normalization/repair:\n'
    result['normalized_lines'] = (stdout.split(marker, 1)[1].split('\n\n', 1)[0].splitlines()
                                  if marker in stdout else ocr)
    return result


def invoke(binary, arguments, models, threads, timeout):
    with tempfile.TemporaryDirectory(prefix='mrz-regression-') as tmp:
        metrics = Path(tmp) / 'time'
        command = ['/usr/bin/time', '-f', '%e %M', '-o', str(metrics), str(binary),
                   *arguments, '--models-dir', str(models)]
        if '--threads' not in arguments:
            command += ['--threads', str(threads)]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        elapsed, rss = metrics.read_text().splitlines()[-1].split()
    if proc.returncode not in (0, 1, 2):
        raise RuntimeError(f'scanner crashed: exit {proc.returncode}: {proc.stderr}')
    result = extract(proc.stdout, proc.returncode)
    result['metrics'] = dict(elapsed_seconds=float(elapsed), max_rss_kib=int(rss))
    return result


def contract_errors(case, actual):
    errors = []
    category = case['category']
    expected_code = {'valid': 0, 'absent': 1, 'unverified': 2, 'usage': 1}[category]
    if actual['exit'] != expected_code:
        errors.append(f"exit {actual['exit']} != {expected_code}")
    if actual['candidate'] != (category in ('valid', 'unverified')):
        errors.append('candidate presence violates contract')
    if category == 'valid':
        if actual.get('checksum_status') != 'OK':
            errors.append('checksums not OK')
        if 'source_lines' in case and actual['normalized_lines'] != case['source_lines']:
            errors.append('recognized data differs from synthetic source')
        if 'format' in case and actual['fields'].get('Format') != case['format']:
            errors.append('wrong format')
        if case.get('require_crop') and not actual['center_crop']:
            errors.append('center-crop fallback not exercised')
    if case['id'] == 'checksum-failed' and actual.get('checksum_status') != 'FAILED':
        errors.append('expected parseable checksum failure')
    if case['id'] == 'parse-failed' and not actual['parse_failed']:
        errors.append('expected unparseable OCR')
    return errors


def baseline_in_original_pixels(expected, size, space):
    """Migrate historical attempt-space geometry without re-recording OCR baselines."""
    if space == 'original':
        return expected
    if space != 'attempt':
        raise ValueError(f'unknown polygon coordinate space: {space}')
    width, height = size
    dx = dy = 0
    if expected['center_crop']:
        side = min(width, height)
        dx, dy = (width - side) // 2, (height - side) // 2
        width = height = side
    polygon = []
    for x, y in expected['polygon']:
        x, y = {
            0: (x, y),
            90: (y, height - 1 - x),
            180: (width - 1 - x, height - 1 - y),
            270: (width - 1 - y, x),
        }[expected['rotation']]
        polygon.append([x + dx, y + dy])
    return {**expected, 'polygon': polygon}


def baseline_errors(actual, expected, size):
    errors = []
    for key in expected:
        if key not in ('polygon', 'metrics') and actual.get(key) != expected[key]:
            errors.append(f'{key} changed')
    # Unordered corners, tolerance in original-file pixels.
    a, b = actual['polygon'], expected['polygon']
    tolerance = max(3.0, 0.02 * max(size))
    if len(a) != len(b) or any(not math.isfinite(v) for p in a for v in p):
        errors.append('invalid polygon')
    elif a and max(max(min(math.dist(p, q) for q in b) for p in a),
                   max(min(math.dist(p, q) for q in a) for p in b)) > tolerance:
        errors.append(f'polygon moved by more than {tolerance:.1f}px')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=PROJECT / 'target/release/mrz_scanner')
    parser.add_argument('--models-dir', type=Path, default=PROJECT / 'models')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--include-slow', action='store_true')
    parser.add_argument('--strict-contracts', action='store_true', help='fail on known defects too')
    parser.add_argument('--record', action='store_true', help='explicitly replace baseline')
    parser.add_argument('--private', action='store_true', help='use the three existing local files only')
    parser.add_argument('--only', nargs='+', help='case IDs')
    parser.add_argument('--report', type=Path, help='write observed results (may contain personal data)')
    args = parser.parse_args()
    if args.private and args.report and not args.report.resolve().is_relative_to(PROJECT / 'files'):
        parser.error('private reports must remain under the ignored files/ directory')
    baseline_path = ROOT / 'baseline.json'
    cases = json.loads((ROOT / 'cases.json').read_text())
    cases += [dict(id='usage-no-argument', category='usage', arguments=[]),
              dict(id='usage-unknown-flag', category='usage', arguments=['--unknown']),
              dict(id='usage-nonnumeric-threads', category='usage', arguments=['--threads', 'oops'])]
    if args.private:
        baseline_path = PROJECT / 'files/regression-baseline.json'
        cases = [dict(id=f'local-{i}', category='valid', image=str(PROJECT / 'files' / name),
                      sha256=sha256(PROJECT / 'files' / name), size=size)
                 for i, (name, size) in enumerate([
                     ('1299944263-5304.jpg', [1080, 1080]),
                     ('photo_2026-10-28_13-10-29.jpg', [960, 1280]),
                     ('photo_2026-10-28_13-10-42.jpg', [725, 1024])], 1)]
    if args.record and (args.only or (not args.include_slow and not args.private)):
        parser.error('--record requires the complete suite (--include-slow for public cases)')
    baseline = {} if args.record else json.loads(baseline_path.read_text())
    known = json.loads((ROOT / 'known-failures.json').read_text()) if not args.private else {}
    report = dict(polygon_space='original',
                  environment=dict(platform=platform.platform(), python=platform.python_version(),
                                  threads=args.threads, binary_sha256=sha256(args.binary),
                                  source_sha256={str(path.relative_to(PROJECT)): sha256(path)
                                                 for path in [*sorted((PROJECT / 'src').glob('*.rs')),
                                                              PROJECT / 'Cargo.lock',
                                                              PROJECT / 'Cargo.toml']},
                                  models={name: sha256(args.models_dir / name) for name in MODELS}),
                  cases={})
    if baseline and report['environment']['models'] != baseline['environment']['models']:
        parser.error('model hashes differ from baseline')
    failures = 0
    selected = [c for c in cases if (args.include_slow or not c.get('slow')) and
                (not args.only or c['id'] in args.only)]
    if not selected or (args.only and set(args.only) != {c['id'] for c in selected}):
        parser.error('unknown or excluded case ID (slow cases require --include-slow)')
    for case in selected:
        name = case['id']
        if 'image' in case:
            path = ROOT / case['image']
            if sha256(path) != case['sha256']:
                raise RuntimeError(f'{name}: fixture hash mismatch')
            arguments = [str(path)]
        else:
            arguments = case['arguments']
        actual = invoke(args.binary.resolve(), arguments, args.models_dir.resolve(),
                        args.threads, args.timeout)
        report['cases'][name] = dict(sha256=case.get('sha256'), size=case.get('size', [0, 0]),
                                     result=actual)
        violations = contract_errors(case, actual)
        differences = []
        if baseline:
            previous = baseline['cases'][name]
            if previous['sha256'] != case.get('sha256'):
                differences.append('fixture differs from baseline')
            size = case.get('size', [0, 0])
            expected = baseline_in_original_pixels(previous['result'], size,
                                                   baseline.get('polygon_space', 'attempt'))
            differences += baseline_errors(actual, expected, size)
        if differences or (violations and (args.strict_contracts or name not in known)):
            status = 'FAIL'
            failures += 1
        elif violations:
            status = 'XFAIL'
        else:
            status = 'PASS'
        print(f"{status:5} {name}: {actual['metrics']}"
              + (f"; {'; '.join(differences + violations)}" if differences or violations else ''),
              flush=True)
    if args.record:
        baseline_path.write_text(json.dumps(report, indent=2) + '\n')
        if args.private:
            baseline_path.chmod(0o600)
        print(f'Recorded {baseline_path}')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
        if args.private:
            args.report.chmod(0o600)
    print(f'{len(selected)} cases; {failures} failures')
    return int(failures > 0)


if __name__ == '__main__':
    sys.exit(main())
