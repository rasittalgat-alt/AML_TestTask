import pandas as pd

SRC_PATH = "esf_sample_200k.csv"
DST_PATH = "esf_label_sample_5000.csv"
N_ROWS = 5000

def main():
    print("Читаем файл:", SRC_PATH)
    df = pd.read_csv(SRC_PATH)

    print("Полный размер:", df.shape)

    # перемешаем строки, чтобы выборка была более разнообразной
    df_sample = df.sample(n=N_ROWS, random_state=42).reset_index(drop=True)

    print("Размер сэмпла для разметки:", df_sample.shape)

    # добавим пустую колонку CATEGORY для будущей разметки
    df_sample["CATEGORY"] = ""

    print("Сохраняем файл для разметки:", DST_PATH)
    df_sample.to_csv(DST_PATH, index=False)

    print("✅ Готово! Теперь можно открыть", DST_PATH, "в Excel/Google Sheets и размечать категории.")

if __name__ == "__main__":
    main()
