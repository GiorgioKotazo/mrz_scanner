#!/usr/bin/env python3
"""Rebuild synthetic fixtures; never reads files/ or personal photographs."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FONT = Path('/usr/share/fonts/truetype/ubuntu/UbuntuMono[wght].ttf')
FONT_SHA256 = 'fbf1e748836994f730e602f7dcf2525564d6d78aa336080cbb73af909d0e08ee'


def digit(text):
    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return str(sum((0 if c == '<' else alphabet.index(c)) * (7, 3, 1)[i % 3]
                   for i, c in enumerate(text)) % 10)


def specimens():
    # Invented UTO/TESTER specimen data, not a real identity.
    number, birth, expiry = 'X12345678', '740812', '120415'
    doc = number + digit(number)
    dob, exp = birth + digit(birth), expiry + digit(expiry)
    names = 'TESTER<<ANNA<MARIA'
    personal = 'ZE184226B'.ljust(14, '<')
    p = personal + digit(personal)
    td3 = doc + 'UTO' + dob + 'F' + exp + p
    td3 += digit(doc + dob + exp + p)
    td2 = doc + 'UTO' + dob + 'F' + exp + '<' * 7
    td2 += digit(doc + dob + exp + '<' * 7)
    td1a = 'I<UTO' + doc + '<' * 15
    td1b = dob + 'F' + exp + 'UTO' + '<' * 11
    td1b += digit(td1a[5:] + dob + exp + '<' * 11)
    return {
        'td1': [td1a, td1b, names.ljust(30, '<')],
        'td2': [('I<UTO' + names).ljust(36, '<'), td2],
        'td3': [('P<UTO' + names).ljust(44, '<'), td3],
        'mrv-a': [('V<UTO' + names).ljust(44, '<'),
                  (doc + 'UTO' + dob + 'F' + exp + 'VISATEST').ljust(44, '<')],
        'mrv-b': [('V<UTO' + names).ljust(36, '<'),
                  (doc + 'UTO' + dob + 'F' + exp).ljust(36, '<')],
    }


def render(lines):
    font = ImageFont.truetype(str(FONT), 32)
    width = round(font.getlength(lines[0])) + 80
    image = Image.new('RGB', (width, 640), '#eee9dd')
    draw = ImageDraw.Draw(image)
    draw.text((40, 40), 'UTO / SPECIMEN — NOT A DOCUMENT', font=font, fill='#555555')
    draw.rectangle((40, 110, 210, 340), fill='#aaa59b')
    for i, line in enumerate(lines):
        draw.text((40, 440 + i * 42), line, font=font, fill='black', stroke_width=0)
    return image


def main():
    assert hashlib.sha256(FONT.read_bytes()).hexdigest() == FONT_SHA256, 'font changed'
    cases = []

    def add(name, image, category, lines=None, **extra):
        path = ROOT / 'fixtures' / (name + '.png')
        image.save(path)
        case = dict(id=name, image='fixtures/' + path.name, category=category,
                    size=list(image.size), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if lines:
            case['source_lines'] = lines
        case.update(extra)
        cases.append(case)

    samples = specimens()
    for name, lines in samples.items():
        assert all(len(line) == {'td1': 30, 'td2': 36, 'td3': 44,
                                 'mrv-a': 44, 'mrv-b': 36}[name] for line in lines)
        add(name, render(lines), 'valid', lines,
            format={'td1': 'Td1', 'td2': 'Td2', 'td3': 'Td3',
                    'mrv-a': 'MrvA', 'mrv-b': 'MrvB'}[name])
    base = render(samples['td3'])
    for rotation in (90, 180, 270):
        add(f'td3-rotate-{rotation}', base.rotate(rotation, expand=True), 'valid',
            samples['td3'], format='Td3')
    odd = Image.new('RGB', (base.width + 1, base.height), '#eee9dd')
    odd.paste(base)
    add('odd-size', odd, 'valid', samples['td3'], format='Td3')
    # The document fits the center square, but is tiny in the full tall frame.
    crop = Image.new('RGB', (base.width, base.width * 4), '#b0b0b0')
    crop.paste(base, (0, (crop.height - base.height) // 2))
    add('center-crop', crop, 'valid', samples['td3'], format='Td3', require_crop=True)
    add('tiny', base.resize((232, 160), Image.Resampling.LANCZOS), 'valid',
        samples['td3'], format='Td3')
    add('stretched', base.resize((1800, 320), Image.Resampling.BICUBIC), 'valid',
        samples['td3'], format='Td3')
    add('large', base.resize((6000, 4000), Image.Resampling.BICUBIC), 'valid',
        samples['td3'], format='Td3', slow=True)
    panorama = Image.new('RGB', (6000, 800), '#b0b0b0')
    panorama.paste(base.resize((1100, 760)), (2450, 20))
    add('panorama', panorama, 'valid', samples['td3'], format='Td3', slow=True)
    for name, color in [('white', 'white'), ('black', 'black'), ('solid', '#3878a8')]:
        add(name, Image.new('RGB', (640, 480), color), 'absent')
    bad = samples['td3'].copy()
    # Unambiguous but wrong document number; leave check digits unchanged.
    bad[1] = 'Y12345678' + bad[1][9:]
    add('checksum-failed', render(bad), 'unverified', bad)
    junk = ['ABCDEFGHIJKLMNOPQRSTUVWXYZ<<<<<<<<<<<<<<<<<<'] * 2
    add('parse-failed', render(junk), 'unverified', junk)
    for name, category, sha in [
        ('docsaid', 'valid', 'c1a672b074a708cede498490eca999dc42ee41a79af5d97b4dff12df725daa74'),
        ('photo', 'absent', 'ffe89a0ab0e94114e10777e7313d7fa83d634e34ebc2ea7479085cffa504c920'),
    ]:
        path = ROOT / 'fixtures' / (name + '.jpg')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
        cases.append(dict(id=name, image='fixtures/' + path.name, category=category,
                          size=list(Image.open(path).size), sha256=sha))
    (ROOT / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')


if __name__ == '__main__':
    main()
