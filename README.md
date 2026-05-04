# OCR Act Extraction MVP

FastAPI-приложение для OCR-распознавания актов, счетов и других первичных документов из PDF-файлов и изображений.

Проект умеет:
- загружать документы через UI и API;
- извлекать реквизиты документа, контрагентов и табличную часть;
- определять наличие подписей и печатей;
- экспортировать результат в Excel;
- поддерживать ручную корректировку распознанных данных;
- работать с шаблонами документов;
- ограничивать доступ по ролям `admin` и `operator`.

## Сценарий работы

Типовой поток обработки документа:
1. Загрузить PDF или изображение через UI или API.
2. Запустить OCR-обработку выбранного документа.
3. Проверить извлечённые поля, табличную часть, подписи и печати.
4. При необходимости внести ручные исправления.
5. Скачать исходный файл или выгрузить нормализованный результат в Excel.

## Стек

- `FastAPI` + `Uvicorn`
- `SQLAlchemy` + SQLite
- `PaddleOCR` / `PaddleX`
- `OpenCV`, `PyMuPDF`, `Pillow`, `NumPy`
- `openpyxl`

Основные модули:
- `app/main.py` - точка входа и подключение роутеров
- `app/api/auth.py` - авторизация
- `app/api/documents.py` - загрузка, обработка, предпросмотр, экспорт, ручные правки
- `app/api/templates.py` - CRUD шаблонов
- `app/services/paddle_doc_service.py` - OCR и извлечение таблиц
- `app/services/parser_service.py` - разбор полей документа
- `app/services/document_refinement_service.py` - дообогащение результата по изображению
- `app/services/visual_detection_service.py` - детекция подписей и печати

## Поддерживаемые входные данные

Поддерживаемые форматы файлов:
- `PDF`
- `JPG`
- `JPEG`
- `PNG`

Ограничения:
- до `10 MB` на файл
- до `50` файлов в пакетной загрузке

Результат обработки включает:
- тип документа
- номер и дату
- продавца / исполнителя
- покупателя / заказчика
- суммы без НДС / НДС / с НДС
- строки табличной части
- статус обработки и confidence
- признаки наличия печати и подписей
- Excel-выгрузку

## Запуск через Docker

Требования:
- Docker Desktop
- Docker Compose

Запуск:

```bash
docker compose up --build
```

Приложение будет доступно по адресу:

```text
http://localhost:8001
```

`docker-compose.yml` также:
- монтирует `./data` в `/data`
- сохраняет кэш PaddleX в `./data/paddlex_cache`
- задаёт `STORAGE_DIR=/data/storage`
- задаёт `EXPORT_DIR=/data/exports`
- отключает лишнюю проверку источников моделей через `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`

Примечания:
- при первом запуске OCR-модели могут скачиваться заметно долго;
- при следующих запусках используется кэш из `data/paddlex_cache`.

## Локальный запуск

Требования:
- Python `3.11`
- установленный `tesseract-ocr`
- пакет `tesseract-ocr-rus` для русского языка

Установка и запуск:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Локальный адрес:

```text
http://localhost:8000
```

## Авторизация

Приложение использует HTTP Basic auth.

Пользователи по умолчанию создаются автоматически при старте:

```text
admin / admin123
operator / operator123
```

Их можно переопределить переменными окружения:
- `OCR_ADMIN_USERNAME`
- `OCR_ADMIN_PASSWORD`
- `OCR_OPERATOR_USERNAME`
- `OCR_OPERATOR_PASSWORD`

## Интерфейс

Встроенный интерфейс доступен по адресу:

```text
GET /
```

В UI доступны:
- вход в систему;
- одиночная и пакетная загрузка;
- список документов;
- запуск OCR-обработки;
- предпросмотр страниц;
- просмотр и ручная корректировка результата;
- скачивание исходного файла;
- выгрузка в Excel.

## Основные API-роуты

### Авторизация

- `POST /api/auth/login`
- `GET /api/auth/me`

### Документы

- `POST /api/documents/upload`
- `POST /api/documents/upload-batch`
- `GET /api/documents`
- `GET /api/documents/{document_id}`
- `GET /api/documents/{document_id}/preview/{page_number}`
- `POST /api/documents/{document_id}/process`
- `PUT /api/documents/{document_id}/corrections`
- `GET /api/documents/{document_id}/download`
- `GET /api/documents/{document_id}/export/xlsx`
- `GET /api/documents/export/xlsx`
- `DELETE /api/documents/{document_id}` - только для `admin`

### Шаблоны

- `GET /api/templates`
- `GET /api/templates/{template_id}`
- `POST /api/templates`
- `DELETE /api/templates/{template_id}` - только для `admin`

## Структура данных документа

Для каждого документа хранятся:
- метаданные исходного файла;
- `raw_text` после OCR;
- `parsed_json` с результатом автоматического распознавания;
- `updated_json` с ручными правками;
- путь к сформированному Excel-файлу;
- статус обработки, confidence и текст ошибки.

UI и API возвращают `effective_json`:
- `updated_json`, если есть ручные корректировки;
- иначе `parsed_json`.

Пример фрагмента ответа:

```json
{
  "document_type": "act",
  "document_number": "47",
  "document_date": "2024-03-01",
  "seller": {
    "name": "ООО \"ДВФ\"",
    "inn": "7707337629",
    "kpp": ""
  },
  "buyer": {
    "name": "ООО \"ЛЕНТА-ЦЕНТР\"",
    "inn": "7721511903",
    "kpp": ""
  },
  "totals": {
    "amount_without_vat": "62500.00",
    "vat_amount": "12500.00",
    "amount_with_vat": "75000.00"
  },
  "line_items": [
    {
      "row_no": "1",
      "name": "Услуги по механизированной очистке территории от снега",
      "quantity": "5",
      "unit": "усл.",
      "price": "3500.00",
      "amount": "17500.00",
      "vat_rate": "20%"
    }
  ],
  "signatures": {
    "seal_present": true,
    "contractor_signature_present": true,
    "customer_signature_present": false
  }
}
```

## Рабочие директории и данные

Во время работы проект использует:
- `data/app.db` - SQLite-базу
- `data/storage/` - загруженные файлы и превью
- `data/exports/` - Excel-файлы
- `data/paddlex_cache/` - кэш OCR-моделей

Эти данные не нужно коммитить в Git. Они уже добавлены в `.gitignore`.
