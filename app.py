import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Klasifikasi Jenis Diet Terapeutik - RSUD Sawahlunto",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "project_dataa.xlsx"
LOGO_PATH = BASE_DIR / "Lambang_Kota_Sawahlunto.png"

MODEL_PATH: Optional[str] = None  # biarkan None agar auto-detect

COL_DATE = "Tanggal"
COL_IMT = "Status Gizi (IMT)"
COL_ASUHAN = "Asuhan Gizi"
COL_DIET = "Jenis Diet"
COL_DIAG_1 = "Diagnosis Medis"
COL_DIAG_2 = "Diagnosa Medis"

MAX_DATE = pd.Timestamp("2024-06-30")


# =========================================================
# COLORS (JANGAN DIUBAH — sesuai punya kamu)
# =========================================================
C = {
    # wallpaper / app bg
    "bg_top": "#D7E2EA",
    "bg_mid": "#D2E7E4",
    "bg_bot": "#E6D9C8",

    # glass
    "card": "rgba(255,255,255,0.28)",
    "card2": "rgba(255,255,255,0.24)",

    # text
    "text": "#0B1220",
    "muted": "rgba(11,18,32,0.72)",

    # borders / grid
    "border": "rgba(255,255,255,0.22)",
    "border_strong": "rgba(255,255,255,0.26)",
    "grid": "rgba(11,18,32,0.10)",

    # accent
    "accent": "#3B82F6",

    # sidebar (dark)
    "sb_bg_top": "#0B1220",
    "sb_bg_mid": "#13233A",
    "sb_bg_bot": "#12233B",
    "sb_text": "rgba(255,255,255,0.92)",
    "sb_muted": "rgba(255,255,255,0.72)",
    "sb_border": "rgba(255,255,255,0.12)",
}

# =========================================================
# PALETTE (SOFT & KONSISTEN — biar ga rame)
# =========================================================
PALETTE = [
    "#4C78A8",  # soft blue (utama)
    "#72B7B2",  # soft teal
    "#54A24B",  # soft green
    "#ECA82C",  # soft amber
    "#B279A2",  # soft purple
]


# =========================================================
# CSS
# =========================================================
def inject_css():
    st.markdown(
        f"""
        <style>
        /* ---- APP BACKGROUND ---- */
        .stApp {{
          padding-top: env(safe-area-inset-top);
          background:
radial-gradient(1200px 520px at 50% 72%, rgba(13,28,50,0.22), transparent 72%),
            radial-gradient(900px 480px at 86% 10%, rgba(84,162,75,0.10), transparent 70%),
            radial-gradient(900px 540px at 50% 102%, rgba(236,168,44,0.10), transparent 72%),
            linear-gradient(180deg, {C["bg_top"]} 0%, {C["bg_mid"]} 55%, {C["bg_bot"]} 100%);
          color: {C["text"]};
        }}

        /* ruang atas biar ga kepotong */
        div[data-testid="stAppViewContainer"] .block-container {{
  padding-top: 2.0rem !important;
  padding-bottom: 2.0rem !important;

  /* ✅ bikin lebar penuh (hilangkan sisa kiri-kanan) */
  max-width: 100% !important;

  /* ✅ rapikan padding kiri-kanan */
  padding-left: 1.8rem !important;
  padding-right: 1.8rem !important;
        }}

        /* ===== SIDEBAR: DARK AESTHETIC + FIX TOP ===== */
        section[data-testid="stSidebar"] {{
          background:
            radial-gradient(720px 420px at 25% 8%, rgba(59,130,246,0.18), transparent 70%),
            radial-gradient(600px 420px at 80% 22%, rgba(72,184,166,0.14), transparent 72%),
            linear-gradient(180deg, {C["sb_bg_top"]} 0%, {C["sb_bg_mid"]} 55%, {C["sb_bg_bot"]} 100%);
          border-right: 1px solid {C["sb_border"]};
          padding-top: 18px !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {{
          padding-top: 18px !important;
        }}
        section[data-testid="stSidebar"] * {{
          color: {C["sb_text"]} !important;
        }}
        section[data-testid="stSidebar"] .stMarkdown p {{
          color: {C["sb_muted"]} !important;
        }}

        .sidebar-line {{
          height: 1px;
          background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
          margin: 14px 0;
          border-radius: 999px;
        }}

        /* ===== TITLE CARD (JUDUL TENGAH) ===== */
        .title-wrap {{
          background: rgba(255,255,255,0.20);
          border: 1px solid rgba(255,255,255,0.24);
          border-radius: 22px;
          padding: 16px 18px;
          box-shadow: 0 18px 42px rgba(2,6,23,0.12);
          backdrop-filter: blur(14px);
          text-align: center;
          margin-top: 0.25rem;
        }}
        .title-main {{
          font-size: 28px;
          font-weight: 950;
          margin: 0;
          letter-spacing: -0.02em;
          text-align: center;
        }}
        .title-line {{
          height: 2px;
          width: 200px;
          margin: 10px auto 0 auto;
          background: linear-gradient(90deg, rgba(59,130,246,0.0), rgba(59,130,246,0.70), rgba(44,177,161,0.55), rgba(59,130,246,0.0));
          border-radius: 999px;
        }}

        /* ===== CARDS ===== */
        .card {{
          background: {C["card"]};
          border: 1px solid {C["border_strong"]};
          border-radius: 22px;
          padding: 14px 14px;
          box-shadow: 0 16px 34px rgba(2,6,23,0.12);
          backdrop-filter: blur(14px);
        }}
        .card-h {{
          font-weight: 900;
          font-size: 15px;
          color: {C["muted"]};
          margin-bottom: 10px;
          display: flex;
          align-items: center;
          gap: 10px;
        }}

        /* emoji/icon chip bigger + glass */
        .chip {{
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 30px;
          height: 25px;
          border-radius: 12px;
          background: rgba(255,255,255,0.16);
          border: 1px solid rgba(255,255,255,0.26);
          font-size: 25px;
          line-height: 1;
        }}

        /* =====================================================
           KPI: LABEL + VALUE DI TENGAH (INI YANG KAMU MAU)
           ===================================================== */
        .kpi-box {{
          background: rgba(255,255,255,0.22) !important;
          border: 1px solid rgba(255,255,255,0.26) !important;
          border-radius: 22px !important;
          padding: 14px 16px !important;
          box-shadow: 0 16px 34px rgba(2,6,23,0.12) !important;
          backdrop-filter: blur(14px) !important;

          /* ✅ center semua isi KPI */
          text-align: center !important;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;

          /* ✅ biar tinggi KPI sejajar */
          min-height: 84px;
        }}
        .kpi-label {{
          font-size: 13px !important;
          font-weight: 900 !important;
          color: {C["muted"]} !important;

          /* ✅ center */
          width: 100%;
          text-align: center !important;
          line-height: 1.15;
        }}
        .kpi-value {{
          font-size: 24px;
          font-weight: 950;
          margin-top: 6px;

          /* ✅ center */
          width: 100%;
          text-align: center !important;
          line-height: 1;
          white-space: nowrap;
        }}

        /* ===== Rentang tanggal aesthetic (vertical + arrow down) ===== */
        .kpi-range {{
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 6px;
          padding: 10px 10px;
          border-radius: 18px;
          background: rgba(255,255,255,0.14);
          border: 1px solid rgba(255,255,255,0.22);
          box-shadow: 0 14px 30px rgba(2,6,23,0.10);
          backdrop-filter: blur(14px);
        }}
        .kpi-range-label {{
          font-size: 14px !important;
          font-weight: 950 !important;
          color: rgba(11,18,32,0.70) !important;
          letter-spacing: 0.2px;
          text-align: center;
          width: 100%;
        }}
        .kpi-range-value {{
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          flex-direction: column;
          font-size: 20px !important;
          font-weight: 950 !important;
          color: rgba(11,18,32,0.95) !important;
          text-align: center;
          width: 100%;
        }}
        .kpi-date {{
          padding: 6px 10px;
          border-radius: 999px;
          background: rgba(255,255,255,0.16);
          border: 1px solid rgba(255,255,255,0.24);
        }}
        .kpi-arrow {{
          font-size: 20px !important;
          opacity: 0.70 !important;
          line-height: 1 !important;
        }}

        /* ===== BUTTONS ===== */
        div.stButton > button {{
          border-radius: 14px;
          padding: 10px 14px;
          border: 1px solid rgba(59,130,246,0.34);
          background: rgba(59,130,246,0.14);
          color: {C["text"]};
          font-weight: 900;
        }}
        div.stButton > button:hover {{
          background: rgba(59,130,246,0.22);
        }}

        /* ===== INPUTS MAIN ===== */
        div[data-testid="stAppViewContainer"] .stTextInput input,
        div[data-testid="stAppViewContainer"] .stDateInput input,
        div[data-testid="stAppViewContainer"] .stTextArea textarea {{
          border-radius: 14px !important;
          border: 1px solid rgba(11,18,32,0.14) !important;
          background: rgba(255,255,255,0.55) !important;
        }}
        div[data-testid="stAppViewContainer"] div[data-baseweb="select"] > div {{
          border-radius: 14px !important;
          border: 1px solid rgba(11,18,32,0.14) !important;
          background: rgba(255,255,255,0.55) !important;
        }}

        /* ===== INPUTS SIDEBAR (dark) ===== */
        section[data-testid="stSidebar"] .stTextInput input,
        section[data-testid="stSidebar"] .stDateInput input,
        section[data-testid="stSidebar"] .stTextArea textarea {{
          border-radius: 14px !important;
          border: 1px solid rgba(255,255,255,0.14) !important;
          background: rgba(255,255,255,0.08) !important;
          color: rgba(255,255,255,0.92) !important;
        }}
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
          border-radius: 14px !important;
          border: 1px solid rgba(255,255,255,0.14) !important;
          background: rgba(255,255,255,0.08) !important;
          color: rgba(255,255,255,0.92) !important;
        }}
        section[data-testid="stSidebar"] input::placeholder {{
          color: rgba(255,255,255,0.55) !important;
        }}

        /* ===== DATAFRAME glass (nyatu wallpaper) ===== */
        div[data-testid="stDataFrame"] {{
          background: rgba(255,255,255,0.16) !important;
          border: 1px solid rgba(255,255,255,0.22) !important;
          border-radius: 18px !important;
          box-shadow: 0 14px 30px rgba(2,6,23,0.10) !important;
          backdrop-filter: blur(14px) !important;
          overflow: hidden !important;
        }}
        div[data-testid="stDataFrame"] [role="columnheader"] {{
          background: rgba(255,255,255,0.18) !important;
          color: rgba(11,18,32,0.78) !important;
          font-weight: 900 !important;
          border-bottom: 1px solid rgba(11,18,32,0.10) !important;
        }}
        div[data-testid="stDataFrame"] [role="gridcell"] {{
          background: rgba(255,255,255,0.10) !important;
          color: rgba(11,18,32,0.86) !important;
          border-bottom: 1px solid rgba(11,18,32,0.06) !important;
        }}
        div[data-testid="stDataFrame"] [role="row"]:nth-child(even) [role="gridcell"] {{
          background: rgba(255,255,255,0.14) !important;
        }}

        /* hide default menus + "Deploy" toolbar */
        footer {{visibility: hidden;}}
        #MainMenu {{visibility: hidden;}}
        header[data-testid="stHeader"] {{display: none !important;}}
        div[data-testid="stToolbar"] {{display: none !important;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# =========================================================
# CLEANING + MAPPING 4 KELAS
# =========================================================
def clean_for_model(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s,\/\-\+]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def diet_terapeutik_4kelas(diet: str) -> str:
    d = clean_for_model(diet)
    if ("diet diabetes" in d) or ("diabetes" in d) or ("diabet" in d) or re.search(r"\bdm\b", d):
        return "Diet DM"
    if ("rendah garam" in d) or re.search(r"\brg\b", d) or ("jantung" in d) or ("hipertensi" in d):
        return "Diet Rendah Garam/Jantung"
    if ("tinggi protein" in d) or ("rendah protein" in d) or re.search(r"\btkp\b", d) or re.search(r"\brp\b", d) or ("tktp" in d):
        return "Diet Protein"
    return "Diet Standar/Umum"


def normalize_asuhan_2kelas(x: str) -> str:
    s = clean_for_model(x)
    if s in {"ya", "y", "yes"}:
        return "Ya"
    if s in {"tidak", "t", "no"}:
        return "Tidak"
    return "Tidak"


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        st.error(f"File data tidak ditemukan: {path}")
        st.stop()

    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    diag_col = COL_DIAG_1 if COL_DIAG_1 in df.columns else (COL_DIAG_2 if COL_DIAG_2 in df.columns else None)
    if diag_col is None:
        st.error("Kolom diagnosis tidak ditemukan (Diagnosis Medis / Diagnosa Medis).")
        st.stop()

    required = [COL_DATE, COL_IMT, COL_ASUHAN, COL_DIET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Kolom wajib tidak ditemukan: {missing}\nKolom tersedia: {list(df.columns)}")
        st.stop()

    for c in [COL_IMT, COL_ASUHAN, diag_col, COL_DIET]:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].str.replace(r"\s+", " ", regex=True)

    df.replace({"": np.nan, "nan": np.nan, "None": np.nan}, inplace=True)

    dt = pd.to_datetime(df[COL_DATE], errors="coerce", dayfirst=True)
    if dt.dropna().empty:
        dt = dt.fillna(MAX_DATE)
    else:
        dt = dt.fillna(dt.dropna().median())
    df[COL_DATE] = dt
    df.loc[df[COL_DATE] > MAX_DATE, COL_DATE] = MAX_DATE

    df[COL_IMT] = df[COL_IMT].fillna("Tidak diketahui")
    df[COL_DIET] = df[COL_DIET].fillna("Diet Standar/Umum")
    df[diag_col] = df[diag_col].fillna("Tidak diketahui")
    df[COL_ASUHAN] = df[COL_ASUHAN].fillna("Tidak").apply(normalize_asuhan_2kelas)

    df["Diagnosis_Raw"] = df[diag_col].astype(str).fillna("Tidak diketahui").str.strip()
    df["Diagnosis_Model"] = df["Diagnosis_Raw"].apply(clean_for_model)

    df["month"] = df[COL_DATE].dt.month.astype(int)
    df["year"] = df[COL_DATE].dt.year.astype(int)

    df["Diet_Terapeutik_4Kelas"] = df[COL_DIET].apply(diet_terapeutik_4kelas)
    return df, diag_col


# =========================================================
# MODEL DETECT + LOAD
# =========================================================
def detect_model_path() -> Optional[Path]:
    if MODEL_PATH is not None:
        p = Path(MODEL_PATH)
        if p.exists():
            return p

    candidates = list(BASE_DIR.glob("*RETRAIN*.pkl")) + list(BASE_DIR.glob("*.pkl"))
    seen = set()
    uniq = []
    for c in candidates:
        if c.name not in seen:
            uniq.append(c)
            seen.add(c.name)
    return uniq[0] if uniq else None


@st.cache_resource
def load_model(model_path: Path):
    import joblib
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and "pipeline" in bundle:
        return bundle["pipeline"]
    return bundle


def classify_one(model, imt: str, asuhan: str, diagnosis: str, tgl: pd.Timestamp):
    X = pd.DataFrame([{
        COL_IMT: str(imt),
        COL_ASUHAN: str(asuhan),
        "Diagnosis_Model": clean_for_model(diagnosis),
        "month": int(tgl.month),
        "year": int(tgl.year),
    }])

    y = model.predict(X)[0]

    prob_df = None
    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        try:
            proba = model.predict_proba(X)[0]
            classes = list(model.classes_)
            prob_df = pd.DataFrame({"Kelas": classes, "Probabilitas": proba}).sort_values("Probabilitas", ascending=False)
        except Exception:
            prob_df = None

    return y, X, prob_df


# =========================================================
# PLOTLY HELPERS (lebih profesional & tidak rame)
# =========================================================
def style_fig(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=54, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], size=12),
        title_font=dict(size=15, color=C["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=C["grid"], zeroline=False,
        title_font=dict(size=13, color=C["text"], family="Arial Black"),
        tickfont=dict(size=11, color="rgba(11,18,32,0.78)"),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=C["grid"], zeroline=False,
        title_font=dict(size=13, color=C["text"], family="Arial Black"),
        tickfont=dict(size=11, color="rgba(11,18,32,0.78)"),
    )
    return fig


def plot_bar_h_with_labels(df_in, x, y, title, x_title="Jumlah Layanan", y_title="Kategori"):
    fig = px.bar(
        df_in,
        x=x,
        y=y,
        orientation="h",
        text=x,
        color_discrete_sequence=[PALETTE[0]],
    )
    fig.update_layout(title=title, showlegend=False)
    fig.update_traces(
        marker_color=PALETTE[0],
        marker_line_width=0,
        opacity=0.78,
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=12, color=C["text"]),
    )
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    return style_fig(fig, height=390)


def plot_donut_with_labels(labels, values, title, label_map=None):
    labels = list(labels)
    values = list(values)

    display_labels = labels
    if label_map:
        display_labels = [label_map.get(x, x) for x in labels]

    colors = (PALETTE * 10)[:len(display_labels)]

    fig = go.Figure(data=[
        go.Pie(
            labels=display_labels,
            values=values,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color="rgba(255,255,255,0.85)", width=2)),
            textinfo="label+percent",
            textposition="outside",
            textfont=dict(size=12, color=C["text"]),
            outsidetextfont=dict(size=12, color=C["text"]),
            automargin=True,
            sort=False,
        )
    ])
    fig.update_layout(title=title, uniformtext_minsize=10, uniformtext_mode="hide")
    return style_fig(fig, height=380)


def plot_line_with_labels(df_in, x, y, title, x_title="Bulan (YYYY-MM)", y_title="Jumlah Layanan"):
    fig = px.line(df_in, x=x, y=y, markers=True, color_discrete_sequence=[PALETTE[0]])
    fig.update_layout(title=title)
    fig.update_traces(line=dict(width=3.2), marker=dict(size=8))
    fig.update_xaxes(tickangle=45, title=x_title)
    fig.update_yaxes(title=y_title)
    return style_fig(fig, height=370)


def plot_prob_bar(prob_df):
    fig = px.bar(
        prob_df,
        x="Probabilitas",
        y="Kelas",
        orientation="h",
        text="Probabilitas",
        color_discrete_sequence=[PALETTE[1]],
    )
    fig.update_layout(title="Probabilitas per Kelas", showlegend=False)
    fig.update_traces(
        marker_color=PALETTE[1],
        texttemplate="%{text:.2%}",
        textposition="outside",
        cliponaxis=False,
        opacity=0.80,
    )
    fig.update_xaxes(title="Probabilitas")
    fig.update_yaxes(title="Kelas")
    return style_fig(fig, height=370)
# ✅ TEMPEL DI SINI (di bawah plot_prob_bar)
def plot_feature_importance(model, top_n: int = 20):
    prep = getattr(model, "named_steps", {}).get("prep", None)
    rf = getattr(model, "named_steps", {}).get("rf", None)

    if prep is None or rf is None:
        return None, "Pipeline tidak punya step 'prep' dan/atau 'rf'."
    if not hasattr(rf, "feature_importances_"):
        return None, "Model tidak mendukung feature_importances_."

    try:
        feat_names = prep.get_feature_names_out()
    except Exception:
        feat_names = np.array([f"f{i}" for i in range(len(rf.feature_importances_))])

    fi = pd.DataFrame({
        "Feature": feat_names.astype(str),
        "Importance": rf.feature_importances_.astype(float),
    }).sort_values("Importance", ascending=False).head(top_n)

    fi["Feature"] = (
        fi["Feature"]
        .str.replace(r"^(cat|txt|num)__", "", regex=True)
        .str.replace("Diagnosis_Model", "Diagnosis", regex=False)
    )

    fig = px.bar(
        fi.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        color_discrete_sequence=[PALETTE[2]],
    )
    fig.update_layout(title=None, showlegend=False)
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside", cliponaxis=False)
    fig = style_fig(fig, height=560)

    return fig, None

# =========================================================
# UI HELPERS
# =========================================================
def card_open(title: str, chip: str = ""):
    chip_html = f'<span class="chip">{chip}</span>' if chip else ""
    st.markdown(f'<div class="card"><div class="card-h">{chip_html}{title}</div>', unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def kpi(label: str, value: str):
    st.markdown(
        f"""
        <div class="kpi-box">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# INSIGHT HELPERS (MENU BARU)
# =========================================================
def pct(a: float) -> str:
    try:
        return f"{a*100:.1f}%"
    except Exception:
        return "-"


def safe_mode(series: pd.Series):
    if series is None or len(series) == 0:
        return None
    vc = series.value_counts()
    if len(vc) == 0:
        return None
    return vc.index[0], int(vc.iloc[0])


def last_trend_direction(df_in: pd.DataFrame) -> tuple[str, float]:
    if df_in is None or len(df_in) < 6:
        return "Data belum cukup untuk menghitung tren.", 0.0

    tmp = df_in.copy()
    tmp["bulan"] = tmp[COL_DATE].dt.to_period("M").dt.to_timestamp()
    g = tmp.groupby("bulan").size().sort_index().reset_index(name="Jumlah")
    if len(g) < 3:
        return "Data belum cukup untuk menghitung tren.", 0.0

    g3 = g.tail(3).copy()
    x = np.arange(len(g3), dtype=float)
    y = g3["Jumlah"].values.astype(float)
    slope = float(np.polyfit(x, y, 1)[0])

    if abs(slope) < 0.15:
        return "Tren 3 bulan terakhir cenderung stabil.", slope
    if slope > 0:
        return "Tren 3 bulan terakhir cenderung meningkat.", slope
    return "Tren 3 bulan terakhir cenderung menurun.", slope


# =========================================================
# LOAD EVERYTHING
# =========================================================
df, diag_col = load_data(DATA_PATH)

model_path = detect_model_path()
if model_path is None:
    st.error("File model (*.pkl) tidak ditemukan di folder project. Pastikan file model ada di folder yang sama dengan app.py.")
    st.stop()

model = load_model(model_path)

min_d = df[COL_DATE].min()
max_d = df[COL_DATE].max()


# =========================================================
# TITLE
# =========================================================
st.markdown(
    """
    <div class="title-wrap">
        <h1 class="title-main">KLASIFIKASI JENIS DIET TERAPEUTIK DI RSUD SAWAHLUNTO</h1>
        <div class="title-line"></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("")


# =========================================================
# SIDEBAR (logo center)
# =========================================================
import base64

def sidebar_logo_center(path: Path, width_px: int = 115):
    if not path.exists():
        return
    img_bytes = path.read_bytes()
    b64 = base64.b64encode(img_bytes).decode()

    st.sidebar.markdown(
        f"""
        <div style="width:100%; display:flex; justify-content:center; align-items:center; margin:10px 0 12px 0;">
            <img src="data:image/png;base64,{b64}" style="width:{width_px}px; height:auto; display:block; margin:0 auto;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

sidebar_logo_center(LOGO_PATH, width_px=100)

st.sidebar.markdown(
    """
    <div style="text-align:center; margin-top:6px;">
      <div style="font-size:20px; font-weight:900;">Dashboard</div>
      <div style="opacity:0.82;"> Kota Sawahlunto</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown('<div class="sidebar-line"></div>', unsafe_allow_html=True)

page = st.sidebar.radio("Menu", ["🏠 Home", "📊 EDA", "🧠 Klasifikasi", "📌 Ringkasan/Insight"], index=0)

st.sidebar.markdown('<div class="sidebar-line"></div>', unsafe_allow_html=True)
st.sidebar.markdown("### 🎛️ Filter Data")

# (biar tidak error: semua string)
years = sorted(df["year"].unique().tolist())
year_options = ["(Semua)"] + [str(y) for y in years]
year_sel = st.sidebar.selectbox("Tahun", year_options, index=0)

imt_vals = sorted(df[COL_IMT].astype(str).unique().tolist())
imt_sel = st.sidebar.selectbox("Status Gizi (IMT)", ["(Semua)"] + imt_vals, index=0)

asuhan_sel = st.sidebar.selectbox("Asuhan Gizi", ["(Semua)", "Ya", "Tidak"], index=0)

dff_base = df.copy()
if year_sel != "(Semua)":
    dff_base = dff_base[dff_base["year"] == int(year_sel)]
if imt_sel != "(Semua)":
    dff_base = dff_base[dff_base[COL_IMT].astype(str) == str(imt_sel)]
if asuhan_sel != "(Semua)":
    dff_base = dff_base[dff_base[COL_ASUHAN] == asuhan_sel]

diag_options = sorted(dff_base["Diagnosis_Raw"].dropna().unique().tolist())
diag_sel = st.sidebar.selectbox("Diagnosis Medis", ["(Semua)"] + diag_options, index=0)

dff = dff_base.copy()
if diag_sel != "(Semua)":
    dff = dff[dff["Diagnosis_Raw"] == diag_sel]


# =========================================================
# HOME
# =========================================================
if page == "🏠 Home":
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        kpi("Total Data (hasil filter)", f"{len(dff):,}")
    with c2:
        kpi("IMT Kategori", f"{dff[COL_IMT].nunique():,}")

    with c3:
        st.markdown(
            f"""
            <div class="kpi-range">
              <div class="kpi-range-label">Rentang Tanggal</div>
              <div class="kpi-range-value">
                <span class="kpi-date">{min_d.date()}</span>
                <span class="kpi-arrow">↓</span>
                <span class="kpi-date">{max_d.date()}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        kpi("Asuhan (2 Kelas)", f"{dff[COL_ASUHAN].nunique():,}")
    with c5:
        kpi("Target", "4 Kelas")

    st.markdown("")
    a, b, c = st.columns([1.15, 1.25, 1.05])

    st.markdown("""
<style>
/* Sembunyikan title/axis label Plotly (tulisan bawah/atas yang duplikat) */
.js-plotly-plot .gtitle { display: none !important; }
.js-plotly-plot .xtitle { display: none !important; }
.js-plotly-plot .ytitle { display: none !important; }
</style>
""", unsafe_allow_html=True)

    with a:
        card_open("Trend layanan per bulan", chip="📈")
        tmp = dff.copy()
        tmp["bulan"] = tmp[COL_DATE].dt.to_period("M").astype(str)
        trend = tmp.groupby("bulan").size().reset_index(name="Jumlah Layanan")
        fig = plot_line_with_labels(trend, "bulan", "Jumlah Layanan", "Trend layanan per bulan", "Bulan", "Jumlah Layanan")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        card_close()

    with b:
        card_open("Distribusi diet terapeutik (4 kelas)", chip="📊")
        dist4 = dff["Diet_Terapeutik_4Kelas"].value_counts().reset_index()
        dist4.columns = ["Kelas", "Jumlah Layanan"]
        dist4 = dist4.sort_values("Jumlah Layanan")
        fig = plot_bar_h_with_labels(dist4, "Jumlah Layanan", "Kelas", "Distribusi diet terapeutik (4 kelas)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        card_close()

    with c:
        card_open("Komposisi Asuhan Gizi (Ya/Tidak)", chip="🥗")
        asu = dff[COL_ASUHAN].value_counts().reset_index()
        asu.columns = ["Asuhan", "Jumlah Layanan"]
        fig = plot_donut_with_labels(asu["Asuhan"], asu["Jumlah Layanan"], "Asuhan Gizi (Ya/Tidak)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        card_close()

    st.markdown("")
    card_open("Preview data (hasil filter sesuai pilihan)", chip="🧾")

    show_cols = [COL_DATE, COL_IMT, COL_ASUHAN, diag_col, COL_DIET, "Diet_Terapeutik_4Kelas"]
    st.caption(f"Hasil filter: **{len(dff):,} baris**")

    if len(dff) == 0:
        st.warning("Tidak ada data yang cocok dengan filter yang dipilih.")
    else:
        max_rows = int(min(len(dff), 5000))
        default_rows = int(min(len(dff), 200))
        n_show = st.slider("Tampilkan jumlah baris", 1, max_rows, default_rows, 1)

        csv_bytes = dff[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Data (CSV)",
            data=csv_bytes,
            file_name="preview_data_filter.csv",
            mime="text/csv",
            use_container_width=False,
        )

        st.dataframe(dff[show_cols].head(n_show), use_container_width=True, height=460)

    card_close()


# =========================================================
# EDA
# =========================================================
elif page == "📊 EDA":
    r1, r2 = st.columns([1.25, 1.0])

    with r1:
        card_open("Top 12 jenis diet", chip="🍽️")
        topdiet = dff[COL_DIET].value_counts().head(12).reset_index()
        topdiet.columns = ["Jenis Diet", "Jumlah Layanan"]
        topdiet = topdiet.sort_values("Jumlah Layanan")
        fig = plot_bar_h_with_labels(topdiet, "Jumlah Layanan", "Jenis Diet", "Top 12 jenis diet")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        card_close()

    with r2:
        card_open("Distribusi Status Gizi (IMT)", chip="🧍")
        imt_counts = dff[COL_IMT].value_counts().reset_index()
        imt_counts.columns = ["IMT", "Jumlah Layanan"]
        fig = plot_donut_with_labels(imt_counts["IMT"], imt_counts["Jumlah Layanan"], "Distribusi Status Gizi (IMT)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        card_close()

    st.markdown("")
    r3, r4 = st.columns([1.25, 1.0])

    with r3:
        card_open("Top 15 Diagnosis Medis", chip="🩺")
        top_diag = dff[diag_col].value_counts().head(15).reset_index()
        top_diag.columns = ["Diagnosis", "Jumlah Layanan"]
        top_diag = top_diag.sort_values("Jumlah Layanan")
        fig = plot_bar_h_with_labels(top_diag, "Jumlah Layanan", "Diagnosis", "Top 15 Diagnosis Medis")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        card_close()

    with r4:
        card_open("Distribusi diet terapeutik (4 kelas)", chip="🍩")
        dist4 = dff["Diet_Terapeutik_4Kelas"].value_counts().reset_index()
        dist4.columns = ["Kelas", "Jumlah Layanan"]
        wrap_map = {"Diet Rendah Garam/Jantung": "Diet Rendah<br>Garam/Jantung"}
        fig = plot_donut_with_labels(dist4["Kelas"], dist4["Jumlah Layanan"], "Distribusi diet (4 kelas)", label_map=wrap_map)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        card_close()
        


# =========================================================
# 📌 RINGKASAN / INSIGHT (MENU BARU)
# =========================================================
elif page == "📌 Ringkasan/Insight":
    card_open("Ringkasan Insight (berdasarkan filter aktif)", chip="📌")

    if len(dff) == 0:
        st.warning("Tidak ada data yang cocok dengan filter yang dipilih. Coba ubah filter di sidebar.")
        card_close()
    else:
        total = int(len(dff))

        kelas_top = safe_mode(dff["Diet_Terapeutik_4Kelas"])
        kelas_top_name, kelas_top_cnt = (kelas_top if kelas_top else ("-", 0))
        kelas_top_pct = kelas_top_cnt / total if total else 0

        diet_top = safe_mode(dff[COL_DIET])
        diet_top_name, diet_top_cnt = (diet_top if diet_top else ("-", 0))
        diet_top_pct = diet_top_cnt / total if total else 0

        diag_top = safe_mode(dff[diag_col])
        diag_top_name, diag_top_cnt = (diag_top if diag_top else ("-", 0))
        diag_top_pct = diag_top_cnt / total if total else 0

        imt_top = safe_mode(dff[COL_IMT])
        imt_top_name, imt_top_cnt = (imt_top if imt_top else ("-", 0))
        imt_top_pct = imt_top_cnt / total if total else 0

        asu_vc = dff[COL_ASUHAN].value_counts()
        ya_pct = (float(asu_vc.get("Ya", 0)) / total) if total else 0
        tidak_pct = (float(asu_vc.get("Tidak", 0)) / total) if total else 0

        trend_text, slope = last_trend_direction(dff)

        st.markdown(
            f"""
**Hasil utama dari data terfilter ({total:,} baris):**
- **Kelas diet terapeutik dominan:** **{kelas_top_name}** — {kelas_top_cnt:,} ({pct(kelas_top_pct)})
- **Jenis diet terbanyak (asli data):** **{diet_top_name}** — {diet_top_cnt:,} ({pct(diet_top_pct)})
- **Diagnosis medis terbanyak:** **{diag_top_name}** — {diag_top_cnt:,} ({pct(diag_top_pct)})
- **Status gizi (IMT) dominan:** **{imt_top_name}** — {imt_top_cnt:,} ({pct(imt_top_pct)})
- **Komposisi asuhan:** Ya {pct(ya_pct)} • Tidak {pct(tidak_pct)}
- **Arah tren layanan:** {trend_text}
            """.strip()
        )

        card_close()


# =========================================================
# KLASIFIKASI
# =========================================================
else:
    left, right = st.columns([1.05, 1.0])

    with left:
        card_open("Input Data Pasien (Klasifikasi)", chip="🧠")
        imt_list = sorted(df[COL_IMT].astype(str).unique().tolist())
        imt_in = st.selectbox("Status Gizi (IMT)", imt_list)

        asuhan_in = st.selectbox("Asuhan Gizi", ["Ya", "Tidak"], index=1)

        all_diag_options = sorted(df["Diagnosis_Raw"].dropna().unique().tolist())
        st.caption("Diagnosis Medis (pilih dari data Excel — bisa lebih dari satu)")
        diag_multi = st.multiselect("Pilih diagnosis", options=all_diag_options, default=[])

        diagnosis_in = ", ".join(diag_multi).strip() if diag_multi else "Tidak diketahui"

        tgl = st.date_input("Tanggal layanan", value=max_d.date(), min_value=min_d.date(), max_value=max_d.date())

        run = st.button("🔎 Klasifikasikan Jenis Diet", use_container_width=True)
        card_close()

    with right:
        card_open("Hasil Klasifikasi", chip="✅")
        if run:
            y, X, prob_df = classify_one(model, imt_in, asuhan_in, diagnosis_in, pd.Timestamp(tgl))
            st.success(f"✅ Hasil Klasifikasi: **{y}**")

            if prob_df is not None:
                fig = plot_prob_bar(prob_df)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.caption("Input yang dikirim ke model")
            st.dataframe(X, use_container_width=True)
        else:
            st.info("Isi input di kiri, lalu klik **Klasifikasikan Jenis Diet**.")
        card_close()
