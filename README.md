# mrz_scanner

Локальный CLI на Rust для извлечения данных из машиночитаемой зоны (MRZ) паспортов и других документов ICAO 9303. Python и внешние OCR-сервисы не используются.

## Возможности

- DOCSAID detection + recognition ONNX models через ONNX Runtime (`ort`).
- OpenCV-совместимая предобработка, поиск MRZ и коррекция перспективы на Rust.
- Автоматические повороты на 0/90/180/270 градусов и fallback с центральной обрезкой.
- Разбор TD1/TD2/TD3/MRV-A/MRV-B и проверка контрольных цифр через `mrz`.
- Вывод исходной MRZ, структурированных полей и результата каждой проверки.
- Остановка на первом checksum-valid TD3; при неудаче выбирается лучший кандидат.

## Сборка

Требуется Rust 1.92 или новее, `curl` и `sha256sum`.

```bash
cd mrz_scanner
./scripts/download_models.sh
cargo build --release
```

ONNX Runtime линкуется в бинарник. Для запуска нужны исполняемый файл и две внешние ONNX-модели; Python не требуется.

## Запуск

```bash
./target/release/mrz_scanner /path/to/passport.jpg
```

По умолчанию модели читаются из `./models`. Другой каталог и число CPU-потоков:

```bash
./target/release/mrz_scanner /path/to/passport.jpg \
  --models-dir /opt/mrz_scanner/models \
  --threads 2
```

Поддерживаются JPEG и PNG. Флаг `--no-rotate` отключает автоматические повороты; центральная обрезка всё равно может использоваться как fallback. Для максимальной точности лучше запускать программу без этого флага.

Коды завершения:

- `0` — MRZ разобрана, все применимые контрольные цифры совпали;
- `2` — получен кандидат, но разбор или хотя бы одна контрольная цифра не прошли;
- `1` — модель, изображение или MRZ не удалось обработать.

При коде `2` данные нельзя считать подтверждёнными: необходимо другое фото или ручная проверка.

## Отладка

```bash
./target/release/mrz_scanner /path/to/passport.jpg --debug-dir ./debug
```

Сохраняются вход detection-модели, heatmap, бинаризация и выровненная MRZ. **Эти файлы содержат персональные данные** и не должны попадать в логи или систему контроля версий.

## Тесты

```bash
cargo test
cargo clippy --all-targets -- -D warnings
```

## Модели

Используются официальные DOCSAID MRZScanner 1.0.7 two-stage weights:

- `mrz_detection_20250222_fp32.onnx`;
- `mrz_recognition_20250221_fp32.onnx`.

Скрипт загрузки проверяет SHA-256 из официального каталога DOCSAID. Лицензия кода и официальных весов — Apache-2.0. Источники:

- https://github.com/DocsaidLab/MRZScanner
- https://docsaid.org/en/models/mrzscanner
- https://github.com/pykeio/ort
- https://github.com/ruledicaprio/SynthPass/tree/main/crates/mrz

Контрольные цифры подтверждают checksum consistency, а не подлинность документа. В TD3 имя и фамилия не покрываются контрольной цифрой, поэтому при критичных операциях их также нужно проверять по визуальной зоне.
