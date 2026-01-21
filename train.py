import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

DATA_PATH  = Path("project_dataa.xlsx")
OUT_PATH   = Path("model_diet_proyek_RETRAIN.pkl")

COL_DATE   = "Tanggal"
COL_IMT    = "Status Gizi (IMT)"
COL_ASUHAN = "Asuhan Gizi"
COL_DIET   = "Jenis Diet"
COL_DIAG_1 = "Diagnosis Medis"
COL_DIAG_2 = "Diagnosa Medis"

MAX_DATE = pd.Timestamp("2024-06-30")


def clean_for_model(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s,\/\-\+]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s,\/\-\+]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ====== TARGET 4 KELAS (sesuai proyek Anda) ======
def diet_terapeutik_4kelas(diet: str) -> str:
    d = normalize_text(diet)

    if ("diet diabetes" in d) or ("diabetes" in d) or ("diabet" in d) or re.search(r"\bdm\b", d):
        return "Diet DM"

    if ("rendah garam" in d) or re.search(r"\brg\b", d) or ("jantung" in d) or ("hipertensi" in d):
        return "Diet Rendah Garam/Jantung"

    if ("tinggi protein" in d) or ("rendah protein" in d) or re.search(r"\btkp\b", d) or re.search(r"\brp\b", d) or ("tktp" in d):
        return "Diet Protein"

    return "Diet Standar/Umum"


def load_data():
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip()

    diag_col = COL_DIAG_1 if COL_DIAG_1 in df.columns else (COL_DIAG_2 if COL_DIAG_2 in df.columns else None)
    if diag_col is None:
        raise ValueError("Kolom diagnosis tidak ditemukan (Diagnosis Medis / Diagnosa Medis).")

    # tanggal
    dt = pd.to_datetime(df[COL_DATE], errors="coerce", dayfirst=True)
    dt = dt.fillna(dt.dropna().median())
    df[COL_DATE] = dt
    df.loc[df[COL_DATE] > MAX_DATE, COL_DATE] = MAX_DATE

    # cleaning
    for col in [COL_IMT, COL_ASUHAN, diag_col, COL_DIET]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)

    df.replace({"": np.nan, "nan": np.nan, "None": np.nan}, inplace=True)

    df[COL_IMT]    = df[COL_IMT].fillna("Tidak diketahui")
    df[COL_ASUHAN] = df[COL_ASUHAN].fillna("Tidak diketahui")
    df[diag_col]   = df[diag_col].fillna("Tidak diketahui")
    df[COL_DIET]   = df[COL_DIET].fillna("Diet Standar/Umum")

    df["Diagnosis_Raw"] = df[diag_col].astype(str).fillna("Tidak diketahui").str.strip()
    df["Diagnosis_Model"] = df["Diagnosis_Raw"].apply(clean_for_model)

    df["month"] = df[COL_DATE].dt.month.astype(int)
    df["year"]  = df[COL_DATE].dt.year.astype(int)

    # >>> ini kunci: buat target 4 kelas
    df["Diet_Terapeutik_4Kelas"] = df[COL_DIET].apply(diet_terapeutik_4kelas)

    return df


def main():
    df = load_data()

    X = df[[COL_IMT, COL_ASUHAN, "Diagnosis_Model", "month", "year"]].copy()
    y = df["Diet_Terapeutik_4Kelas"].copy()

    print("Distribusi target 4 kelas:")
    print(y.value_counts())

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), [COL_IMT, COL_ASUHAN]),
            ("txt", TfidfVectorizer(ngram_range=(1,2), max_features=4000), "Diagnosis_Model"),
            ("num", "passthrough", ["month", "year"]),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    pipe = Pipeline(steps=[
        ("prep", pre),
        ("smote", SMOTE(random_state=42)),
        ("rf", clf),
    ])

    # stratify aman karena sekarang cuma 4 kelas (harusnya > 2 tiap kelas)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    diagnosis_options = sorted(df["Diagnosis_Raw"].unique().tolist())
    bundle = {
        "pipeline": pipe,
        "diagnosis_options": diagnosis_options,
        "feature_cols": list(X.columns),
        "target_col": "Diet_Terapeutik_4Kelas",
    }

    joblib.dump(bundle, OUT_PATH)
    print(f"\nSaved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
