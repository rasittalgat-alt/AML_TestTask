import pandas as pd
import joblib
import re
from pathlib import Path

MODEL_PATH = Path("models/tfidf_sgd_model.pkl")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def predict_for_texts(texts):
    model = load_model()
    texts_clean = [clean_text(t) for t in texts]
    preds = model.predict(texts_clean)
    return preds

def predict_for_csv(input_path: str, output_path: str):
    print(f"Читаем входной файл: {input_path}")
    df = pd.read_csv(input_path)

    if "DESCRIPTION" not in df.columns:
        raise ValueError("Входной CSV должен содержать колонку 'DESCRIPTION'")

    print("Количество строк во входном файле:", len(df))

    df["DESCRIPTION_CLEAN"] = df["DESCRIPTION"].astype(str).map(clean_text)

    model = load_model()
    print("Делаем предсказания...")
    df["PREDICTED_CATEGORY"] = model.predict(df["DESCRIPTION_CLEAN"])

    print("Сохраняем результат в:", output_path)
    df.to_csv(output_path, index=False)
    print("✅ Готово!")

if __name__ == "__main__":
    # Примеры одиночных предсказаний
    example_texts = [
        "Мыло хозяйственное 72% 200 гр",
        "Принтер цветной лазерный формат А3",
        "Услуги по сопровождению SAP в феврале 2025",
        "Кроссовки мужские Adidas черные 42 размера",
    ]

    print("=== Примеры предсказаний по одиночным строкам ===")
    preds = predict_for_texts(example_texts)
    for txt, cat in zip(example_texts, preds):
        print(f"{txt} -> {cat}")

    # Пример применения к CSV-сэмплу
    input_csv = "esf_sample_200k.csv"         # можно поменять при необходимости
    output_csv = "esf_sample_200k_predicted.csv"
    print("\n=== Применяем модель к CSV ===")
    predict_for_csv(input_csv, output_csv)
