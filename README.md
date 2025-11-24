# AML_TestTask — классификация наименований ТРУ

Решение тестового задания на позицию ML-инженера.

## 1.Идея

По колонке `DESCRIPTION` (текст наименования ТРУ) предсказываю категорию  
(например, «Продукты питания и напитки», «Обувь и средства индивидуальной защиты (СИЗ)» и т.п.).

Подход:
- предобработка текста (lowercase, нормализация пробелов);
- TF-IDF по символьным n-граммам (3–5);
- линейный классификатор `SGDClassifier (loss="log_loss")`.

Обучение выполняется на вручную размеченном сэмпле `esf_label_sample_5000.csv` (21 категория).

## 2.Установка

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

## 3. Подготовка данных (локально)
Исходный большой файл esf_fulll_2025....csv находится локально и не выкладывается в GitHub.
Путь к нему указывается в константе SRC_PATH в make_sample.py.

# 1) Сделать облегчённый сэмпл (200k строк) из большого CSV
python make_sample.py
# → создаётся файл esf_sample_200k.csv (колонка DESCRIPTION)

# 2) Сформировать выборку для ручной разметки категорий (5k строк)
python make_label_sample.py
# → создаётся файл esf_label_sample_5000.csv с колонками
#    DESCRIPTION и CATEGORY (CATEGORY по умолчанию пустая)





