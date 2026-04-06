import streamlit as st
import pandas as pd
import time
import math
import re
from io import BytesIO

st.set_page_config(page_title="Completed Slate Analyser", layout="wide")

# =========================================================
# PROGRESS TRACKER
# =========================================================
class ProgressTracker:
    def __init__(self, progress_bar, status_box, detail_box):
        self.progress_bar = progress_bar
        self.status_box = status_box
        self.detail_box = detail_box
        self.start_time = time.time()

    def update(self, done, total, detail=""):
        total = max(total, 1)
        frac = min(max(done / total, 0), 1)
        elapsed = time.time() - self.start_time

        if frac > 0:
            est_total = elapsed / frac
            eta = max(0, est_total - elapsed)
        else:
            eta = 0

        self.progress_bar.progress(int(frac * 100))
        self.status_box.markdown(
            f"**Progress:** {frac * 100:,.1f}%  \n"
            f"**Processed:** {done:,} / {total:,} lineups  \n"
            f"**Elapsed:** {format_seconds(elapsed)}  \n"
            f"**ETA:** {format_seconds(eta)}"
        )
        if detail:
            self.detail_box.caption(detail)


def format_seconds(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# =========================================================
# HELPERS
# =========================================================
def clean_numeric(series):
    return (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", "0")
        .fillna("0")
        .astype(float)
    )


def detect_total_score_column(df):
    preferred = [
        "TotalScore",
        "LineupScore",
        "Total Score",
        "Lineup Score",
        "Score",
        "Points",
        "Pts",
    ]
    for col in preferred:
        if col in df.columns:
            return col

    # fallback: look for exact-ish score-like names
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if col_lower in {"totalscore", "lineupscore", "score", "points", "pts"}:
            return col

    return None


def build_total_score_column(df):
    df = df.copy()

    total_score_col = detect_total_score_column(df)
    if total_score_col:
        df["TotalScore_CALC"] = clean_numeric(df[total_score_col])
        return df, total_score_col, "existing"

    # fallback: sum score-like columns
    score_like_cols = []
    for col in df.columns:
        col_lower = str(col).lower()
        if "score" in col_lower or col_lower.endswith("pts") or "_pts" in col_lower:
            score_like_cols.append(col)

    if score_like_cols:
        for col in score_like_cols:
            df[col] = clean_numeric(df[col])
        df["TotalScore_CALC"] = df[score_like_cols].sum(axis=1)
        return df, score_like_cols, "summed"

    raise ValueError(
        "Could not detect a total score column or score-per-slot columns in the uploaded CSV."
    )


def detect_total_salary_column(df):
    preferred = [
        "TotalSalary",
        "SalaryTotal",
        "Total Salary",
        "Salary",
    ]
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if col_lower in {"totalsalary", "salarytotal", "salary"}:
            return col
    return None


def make_excel_bytes(summary_df, detailed_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        detailed_df.to_excel(writer, index=False, sheet_name="Detailed")
    output.seek(0)
    return output.getvalue()


# =========================================================
# APP
# =========================================================
st.title("Completed Slate Analyser")
st.write("Upload a single CSV, then enter the slate's top score and prize cutoff manually.")

uploaded_file = st.file_uploader("Upload lineup results CSV", type=["csv"])

col1, col2, col3 = st.columns(3)
with col1:
    top_score = st.number_input("Historical top score", min_value=0.0, value=850.0, step=0.1)
with col2:
    prize_cutoff = st.number_input("Minimum score that won a prize", min_value=0.0, value=750.0, step=0.1)
with col3:
    salary_cap = st.number_input("Salary cap (optional check)", min_value=0, value=100000, step=1000)

drop_invalid_salary = st.checkbox("Exclude lineups above salary cap", value=False)

run = st.button("Run analysis", type="primary", disabled=uploaded_file is None)

if run and uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        if df_raw.empty:
            raise ValueError("Uploaded CSV is empty.")

        progress_bar = st.progress(0)
        status_box = st.empty()
        detail_box = st.empty()
        live_counts = st.empty()

        tracker = ProgressTracker(progress_bar, status_box, detail_box)

        tracker.update(1, 100, "Loading CSV and detecting score columns...")

        df, score_source, mode = build_total_score_column(df_raw)

        salary_col = detect_total_salary_column(df)
        if salary_col:
            df["TotalSalary_CALC"] = clean_numeric(df[salary_col])
        else:
            df["TotalSalary_CALC"] = pd.NA

        tracker.update(5, 100, f"Score source detected: {score_source}")

        results = []
        total_rows = len(df)

        beat_top = 0
        cashed = 0
        valid_rows = 0
        invalid_salary_rows = 0

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            total_score_val = float(row["TotalScore_CALC"])
            total_salary_val = row["TotalSalary_CALC"]

            salary_valid = True
            if pd.notna(total_salary_val):
                salary_valid = float(total_salary_val) <= salary_cap

            if not salary_valid:
                invalid_salary_rows += 1

            counted = (salary_valid or not drop_invalid_salary)

            beat_top_flag = False
            cash_flag = False

            if counted:
                valid_rows += 1
                beat_top_flag = total_score_val > top_score
                cash_flag = total_score_val >= prize_cutoff

                if beat_top_flag:
                    beat_top += 1
                if cash_flag:
                    cashed += 1

            results.append({
                "LineupIndex": i,
                "TotalScore": round(total_score_val, 2),
                "TotalSalary": None if pd.isna(total_salary_val) else round(float(total_salary_val), 2),
                "SalaryValid": salary_valid,
                "CountedInAnalysis": counted,
                "BeatTopScore": beat_top_flag,
                "WonPrize": cash_flag,
                "TopScoreThreshold": top_score,
                "PrizeCutoffThreshold": prize_cutoff,
            })

            if i % 25 == 0 or i == total_rows:
                tracker.update(
                    5 + int((i / total_rows) * 90),
                    100,
                    f"Analysing lineup {i:,} of {total_rows:,}"
                )
                live_counts.markdown(
                    f"""
**Live counts**
- Counted lineups: **{valid_rows:,}**
- Beat top score: **{beat_top:,}**
- Won prize: **{cashed:,}**
- Invalid salary lineups: **{invalid_salary_rows:,}**
"""
                )

        detailed_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

        tracker.update(97, 100, "Building summary tables...")

        counted_df = detailed_df[detailed_df["CountedInAnalysis"] == True].copy()

        summary_df = pd.DataFrame([{
            "UploadedRows": len(df_raw),
            "CountedLineups": len(counted_df),
            "ExcludedForSalary": invalid_salary_rows if drop_invalid_salary else 0,
            "TopScoreThreshold": top_score,
            "PrizeCutoffThreshold": prize_cutoff,
            "LineupsBeatingTopScore": int(counted_df["BeatTopScore"].sum()) if not counted_df.empty else 0,
            "LineupsWinningPrize": int(counted_df["WonPrize"].sum()) if not counted_df.empty else 0,
            "PctBeatingTopScore": round((counted_df["BeatTopScore"].mean() * 100), 2) if not counted_df.empty else 0.0,
            "PctWinningPrize": round((counted_df["WonPrize"].mean() * 100), 2) if not counted_df.empty else 0.0,
            "HighestLineupScore": round(counted_df["TotalScore"].max(), 2) if not counted_df.empty else None,
            "LowestLineupScore": round(counted_df["TotalScore"].min(), 2) if not counted_df.empty else None,
            "AverageLineupScore": round(counted_df["TotalScore"].mean(), 2) if not counted_df.empty else None,
        }])

        tracker.update(100, 100, "Analysis complete.")

        st.success("Analysis complete.")

        st.subheader("Summary")
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("Detailed results")
        st.dataframe(detailed_df, use_container_width=True, height=500)

        summary_csv = summary_df.to_csv(index=False).encode("utf-8-sig")
        detailed_csv = detailed_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            "Download summary CSV",
            data=summary_csv,
            file_name="completed_slate_summary.csv",
            mime="text/csv"
        )

        st.download_button(
            "Download detailed CSV",
            data=detailed_csv,
            file_name="completed_slate_detailed.csv",
            mime="text/csv"
        )

        # Optional Excel export
        try:
            excel_bytes = make_excel_bytes(summary_df, detailed_df)
            st.download_button(
                "Download Excel workbook",
                data=excel_bytes,
                file_name="completed_slate_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception:
            st.info("Excel export skipped because openpyxl is not installed. CSV downloads still work.")

    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.exception(e)
