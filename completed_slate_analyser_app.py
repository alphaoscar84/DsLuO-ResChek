import streamlit as st
import pandas as pd
import re
import json
import time
import io
import zipfile
import traceback
from pathlib import Path
from collections import deque

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Completed Slate Analyser",
    layout="wide"
)

# =========================================================
# CONFIG
# =========================================================
SALARY_CAP = 100000
APP_TITLE = "Completed Slate Analyser"
APP_SUBTITLE = "Analyse generated lineups against uploaded post-game players CSVs and historical winning scores"

# =========================================================
# TEAM NORMALISATION
# =========================================================
TEAM_ALIASES = {
    "ADE": "ADELAIDE", "ADL": "ADELAIDE", "ADELAIDE": "ADELAIDE",
    "BL": "BRISBANE", "BRL": "BRISBANE", "BRIS": "BRISBANE", "BRISBANE": "BRISBANE", "LIONS": "BRISBANE",
    "CARL": "CARLTON", "CAR": "CARLTON", "CARLTON": "CARLTON",
    "COLL": "COLLINGWOOD", "COL": "COLLINGWOOD", "COLLINGWOOD": "COLLINGWOOD",
    "ESS": "ESSENDON", "ESSENDON": "ESSENDON",
    "FRE": "FREMANTLE", "FREMANTLE": "FREMANTLE",
    "GC": "GOLD COAST", "GCS": "GOLD COAST", "GOLDCOAST": "GOLD COAST", "GOLD COAST": "GOLD COAST",
    "GEEL": "GEELONG", "GEE": "GEELONG", "GEELONG": "GEELONG",
    "GWS": "GWS", "GIANTS": "GWS",
    "HAW": "HAWTHORN", "HAWTHORN": "HAWTHORN",
    "MELB": "MELBOURNE", "MEL": "MELBOURNE", "MELBOURNE": "MELBOURNE",
    "NM": "NORTH MELBOURNE", "NTH": "NORTH MELBOURNE", "NORTH": "NORTH MELBOURNE", "NORTH MELBOURNE": "NORTH MELBOURNE",
    "PORT": "PORT ADELAIDE", "PA": "PORT ADELAIDE", "PORT ADELAIDE": "PORT ADELAIDE",
    "RICH": "RICHMOND", "RICHMOND": "RICHMOND",
    "STK": "ST KILDA", "STKILDA": "ST KILDA", "ST KILDA": "ST KILDA", "SAINTS": "ST KILDA",
    "SYD": "SYDNEY", "SYDNEY": "SYDNEY",
    "WCE": "WEST COAST", "WEST COAST": "WEST COAST",
    "WB": "WESTERN BULLDOGS", "WBD": "WESTERN BULLDOGS", "BULLDOGS": "WESTERN BULLDOGS", "WESTERN BULLDOGS": "WESTERN BULLDOGS",
}

# =========================================================
# PROGRESS TRACKER
# =========================================================
class StreamlitProgressTracker:
    def __init__(self):
        self.start_time = time.time()
        self.progress_bar = st.progress(0, text="Starting analysis...")
        self.status_box = st.empty()
        self.metrics_box = st.empty()
        self.log_box = st.empty()
        self.logs = deque(maxlen=15)

        self.current_percent = 0.0
        self.current_stage = "Starting"
        self.current_item = ""
        self.completed = 0
        self.total = 0

    def _format_seconds(self, seconds):
        if seconds is None or seconds < 0:
            return "—"
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        if m > 0:
            return f"{m}m {s}s"
        return f"{s}s"

    def update(self, percent, stage="", item="", completed=None, total=None, extra=""):
        self.current_percent = max(0.0, min(100.0, float(percent)))
        if stage:
            self.current_stage = stage
        if item:
            self.current_item = item
        if completed is not None:
            self.completed = completed
        if total is not None:
            self.total = total

        elapsed = time.time() - self.start_time
        eta = None
        if self.current_percent > 0:
            estimated_total = elapsed / (self.current_percent / 100.0)
            eta = estimated_total - elapsed

        progress_text = f"{self.current_stage} — {self.current_percent:.1f}%"
        if self.current_item:
            progress_text += f" | {self.current_item}"

        self.progress_bar.progress(int(self.current_percent), text=progress_text)

        self.status_box.markdown(
            f"""
**Stage:** {self.current_stage}  
**Current:** {self.current_item or "—"}  
**Processed:** {self.completed}/{self.total if self.total else "—"}  
{extra}
"""
        )

        self.metrics_box.markdown(
            f"""
**Elapsed:** {self._format_seconds(elapsed)}  
**ETA:** {self._format_seconds(eta)}
"""
        )

    def log(self, message):
        elapsed = time.time() - self.start_time
        self.logs.appendleft(f"[{self._format_seconds(elapsed)}] {message}")
        self.log_box.code("\n".join(self.logs), language="text")

    def done(self, message="Analysis complete"):
        elapsed = time.time() - self.start_time
        self.progress_bar.progress(100, text=message)
        self.status_box.success(f"{message} in {self._format_seconds(elapsed)}")

# =========================================================
# HELPERS
# =========================================================
def normalise_team_name(team):
    s = str(team).strip().upper()
    s = re.sub(r"[^A-Z ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s_nospace = s.replace(" ", "")

    if s in TEAM_ALIASES:
        return TEAM_ALIASES[s]
    if s_nospace in TEAM_ALIASES:
        return TEAM_ALIASES[s_nospace]
    return s

def clean_numeric(series):
    return (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", "0")
        .astype(float)
    )

def normalise_position(pos):
    pos = str(pos).strip().upper()
    if pos == "RK":
        return "RK"
    if pos.startswith("DEF"):
        return "DEF"
    if pos.startswith("MID"):
        return "MID"
    if pos.startswith("FWD"):
        return "FWD"
    return pos

def extract_primary_position(pos):
    pos = str(pos).strip().upper()
    if "(" in pos:
        pos = pos.split("(")[0].strip()
    return normalise_position(pos)

def extract_all_eligible_positions(pos):
    pos = str(pos).strip().upper()
    positions = []

    main = extract_primary_position(pos)
    if main in {"DEF", "MID", "FWD", "RK"}:
        positions.append(main)

    extras = re.findall(r"\((.*?)\)", pos)
    for extra in extras:
        for token in extra.split("/"):
            token = normalise_position(token.strip())
            if token in {"DEF", "MID", "FWD", "RK"} and token not in positions:
                positions.append(token)

    return positions

def detect_lineup_columns(df):
    preferred = ["DEF_1", "DEF_2", "MID_1", "MID_2", "MID_3", "MID_4", "FWD_1", "FWD_2", "RK_1"]
    if all(c in df.columns for c in preferred):
        return preferred
    raise ValueError(
        "Lineup columns not detected. Expected columns: "
        + ", ".join(preferred)
    )

def parse_rank_code(code):
    m = re.fullmatch(r"([DMFR])(\d+)", str(code).upper())
    if not m:
        return None, None
    return m.group(1), int(m.group(2))

def extract_contest_id_from_filename(filename):
    stem = Path(filename).stem
    m = re.match(r"players_(.+)$", stem, re.IGNORECASE)
    return m.group(1) if m else stem

def make_contest_url(contest_id):
    return f"https://www.playup.com.au/fantasy/contest/{contest_id}"

def get_winning_score(contest_url, score_cache):
    value = score_cache.get(contest_url)
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None

def prepare_players_df(df):
    # fallbacks
    if "Position" not in df.columns and "Pos" in df.columns:
        df["Position"] = df["Pos"]

    if "Score" not in df.columns and "Pts" in df.columns:
        df["Score"] = df["Pts"]

    required = ["Name", "Salary", "Position", "Score", "Team", "Opponent"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Players file missing required columns: {missing}")

    df["Salary_num"] = clean_numeric(df["Salary"])
    df["Score_num"] = clean_numeric(df["Score"])
    df["Team_NORM"] = df["Team"].apply(normalise_team_name)
    df["Opponent_NORM"] = df["Opponent"].apply(normalise_team_name)
    return df

def detect_contest_teams(players_df):
    teams = set(players_df["Team_NORM"].dropna().unique()) | set(players_df["Opponent_NORM"].dropna().unique())
    return sorted(list(teams))

def build_rank_tables(players_df):
    rank_lookup = {}
    available_ranks = {"D": [], "M": [], "F": [], "R": []}
    pos_map = {"DEF": "D", "MID": "M", "FWD": "F", "RK": "R"}

    for pos in ["DEF", "MID", "FWD", "RK"]:
        eligible = []
        for _, row in players_df.iterrows():
            if pos in extract_all_eligible_positions(row["Position"]):
                eligible.append(row)

        if not eligible:
            continue

        pos_df = pd.DataFrame(eligible).sort_values(
            by=["Salary_num", "Score_num", "Name"],
            ascending=[False, False, True]
        )

        prefix = pos_map[pos]

        for i, (_, row) in enumerate(pos_df.iterrows(), start=1):
            code = f"{prefix}{i}"
            rank_lookup[code] = row.to_dict()
            available_ranks[prefix].append(i)

    return rank_lookup, available_ranks

def resolve_rank_code_not_used(code, rank_lookup, available_ranks, used):
    prefix, _ = parse_rank_code(code)
    if prefix is None:
        return None

    for n in available_ranks.get(prefix, []):
        candidate = f"{prefix}{n}"
        player = rank_lookup.get(candidate)
        if not player:
            continue

        name = str(player["Name"]).upper()
        if name not in used:
            return candidate

    return None

def make_zip_file(file_map):
    """
    file_map = {filename: bytes}
    """
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, content in file_map.items():
            zf.writestr(filename, content)
    mem_zip.seek(0)
    return mem_zip.getvalue()

# =========================================================
# CORE ANALYSIS
# =========================================================
def analyse_contests_streamlit(lineups_df, uploaded_player_files, score_cache, tracker):
    lineup_cols = detect_lineup_columns(lineups_df)

    tracker.update(3, "Loading inputs", completed=0, total=len(uploaded_player_files))
    tracker.log(f"Loaded {len(lineups_df):,} lineups")
    tracker.log(f"Detected lineup columns: {', '.join(lineup_cols)}")
    tracker.log(f"Detected {len(uploaded_player_files):,} player files")

    summary_rows = []
    detailed_rows = []
    live_summary_placeholder = st.empty()

    total_files = len(uploaded_player_files)
    total_lineups = len(lineups_df)

    # Stage 1: preparation
    tracker.update(8, "Preparing uploads", item="Validating files", completed=0, total=total_files)

    for file_index, uploaded_file in enumerate(uploaded_player_files, start=1):
        contest_id = extract_contest_id_from_filename(uploaded_file.name)
        contest_url = make_contest_url(contest_id)

        tracker.update(
            percent=10 + (70 * (file_index - 1) / max(total_files, 1)),
            stage="Preparing contest",
            item=f"{contest_id} ({file_index}/{total_files})",
            completed=file_index - 1,
            total=total_files
        )
        tracker.log(f"Reading {uploaded_file.name}")

        uploaded_file.seek(0)
        players_df = pd.read_csv(uploaded_file)
        players_df = prepare_players_df(players_df)

        winning_score = get_winning_score(contest_url, score_cache)
        contest_teams = detect_contest_teams(players_df)
        teams_text = " vs ".join(contest_teams) if contest_teams else ""

        rank_lookup, available_ranks = build_rank_tables(players_df)

        valid = 0
        invalid = 0
        beat = 0
        beating_scores = []

        # Roughly 100 progress updates max per contest
        update_every = max(1, total_lineups // 100)

        for lineup_idx, (_, row) in enumerate(lineups_df.iterrows(), start=1):
            used = set()
            total_score = 0.0
            total_salary = 0.0
            valid_lineup = True
            beat_top = False
            resolved_names = []
            resolved_codes = []

            for col in lineup_cols:
                code = str(row[col]).upper()
                resolved = resolve_rank_code_not_used(code, rank_lookup, available_ranks, used)

                if resolved is None:
                    valid_lineup = False
                    break

                p = rank_lookup[resolved]
                player_name = str(p["Name"]).upper()

                used.add(player_name)
                resolved_names.append(str(p["Name"]))
                resolved_codes.append(resolved)
                total_score += float(p["Score_num"])
                total_salary += float(p["Salary_num"])

            if total_salary > SALARY_CAP:
                valid_lineup = False

            if valid_lineup:
                valid += 1
                if winning_score is not None and total_score > winning_score:
                    beat += 1
                    beat_top = True
                    beating_scores.append(round(total_score, 2))
            else:
                invalid += 1

            detailed_rows.append({
                "ContestID": contest_id,
                "ContestURL": contest_url,
                "Teams": teams_text,
                "HistoricalWinningScore": winning_score,
                "LineupNumber": lineup_idx,
                "ValidLineup": valid_lineup,
                "BeatTopScore": beat_top,
                "LineupScore": round(total_score, 2),
                "LineupSalary": round(total_salary, 2),
                "ResolvedCodes": " | ".join(resolved_codes),
                "Players": " | ".join(resolved_names)
            })

            if lineup_idx % update_every == 0 or lineup_idx == total_lineups:
                contest_progress = lineup_idx / max(total_lineups, 1)
                overall = 10 + (70 * ((file_index - 1 + contest_progress) / max(total_files, 1)))

                tracker.update(
                    percent=overall,
                    stage="Analysing lineups",
                    item=f"{contest_id}: lineup {lineup_idx:,}/{total_lineups:,}",
                    completed=file_index - 1,
                    total=total_files,
                    extra=(
                        f"**Contest:** {contest_id}  \n"
                        f"**Valid so far:** {valid:,}  \n"
                        f"**Invalid so far:** {invalid:,}  \n"
                        f"**Beating top score:** {beat:,}"
                    )
                )

        summary_rows.append({
            "ContestID": contest_id,
            "ContestURL": contest_url,
            "Teams": teams_text,
            "ValidLineups": valid,
            "InvalidLineups": invalid,
            "LineupsBeatingTopScore": beat,
            "HistoricalWinningScore": winning_score,
            "WinningScoresBeaten": ", ".join(map(str, beating_scores))
        })

        tracker.log(
            f"{contest_id}: {beat}/{valid} valid lineups beat top score"
            + (f" | Historical winning score: {winning_score}" if winning_score is not None else " | No historical winning score")
        )

        # Live contest summary table
        live_summary_df = pd.DataFrame(summary_rows)
        live_summary_placeholder.dataframe(live_summary_df, use_container_width=True, height=260)

    tracker.update(90, "Building outputs", item="Creating result tables", completed=total_files, total=total_files)
    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.DataFrame(detailed_rows)

    tracker.update(96, "Finalising", item="Preparing downloads", completed=total_files, total=total_files)

    return summary_df, detailed_df

# =========================================================
# UI
# =========================================================
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.expander("What this app does", expanded=False):
    st.markdown(
        """
This app:
- reads a generated **lineups CSV**
- reads one or more post-game **`players_*.csv`** files
- optionally reads a **`winning_score_cache.json`**
- resolves your rank codes (e.g. `D1`, `M4`, `F2`, `R1`) to real players for each contest
- calculates lineup score and salary
- identifies which valid lineups beat the uploaded historical winning score

This version is designed to be **stable on Streamlit Cloud** and uses the cache file for historical winning scores rather than scraping live by default.
"""
    )

col1, col2 = st.columns(2)

with col1:
    lineups_file = st.file_uploader(
        "Upload lineups CSV",
        type=["csv"],
        help="Expected columns: DEF_1, DEF_2, MID_1, MID_2, MID_3, MID_4, FWD_1, FWD_2, RK_1"
    )

    player_files = st.file_uploader(
        "Upload one or more players_*.csv files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload the post-game Draftstars/PlayUp players CSVs"
    )

with col2:
    cache_file = st.file_uploader(
        "Upload winning_score_cache.json (optional)",
        type=["json"],
        help="If provided, the app will use historical winning scores from this cache"
    )

    show_detailed_preview_rows = st.number_input(
        "Detailed preview rows",
        min_value=50,
        max_value=5000,
        value=500,
        step=50
    )

st.divider()

if lineups_file is not None:
    st.info(f"Lineups file loaded: **{lineups_file.name}**")

if player_files:
    st.info(f"Player files loaded: **{len(player_files)}**")

if cache_file is not None:
    st.info(f"Winning score cache loaded: **{cache_file.name}**")

run_clicked = st.button("Start analysis", type="primary", use_container_width=True)

if run_clicked:
    try:
        if lineups_file is None:
            st.error("Please upload the lineups CSV.")
            st.stop()

        if not player_files:
            st.error("Please upload at least one players_*.csv file.")
            st.stop()

        tracker = StreamlitProgressTracker()
        tracker.update(1, "Starting analysis", item="Reading lineups file")

        # Read lineups
        lineups_file.seek(0)
        lineups_df = pd.read_csv(lineups_file, encoding="utf-8-sig")
        tracker.log(f"Loaded lineups file: {lineups_file.name}")

        # Read cache if provided
        score_cache = {}
        if cache_file is not None:
            cache_file.seek(0)
            score_cache = json.load(cache_file)
            tracker.log(f"Loaded {len(score_cache):,} cached winning scores")
        else:
            tracker.log("No cache provided; historical winning scores will be blank unless present in uploaded cache")

        summary_df, detailed_df = analyse_contests_streamlit(
            lineups_df=lineups_df,
            uploaded_player_files=player_files,
            score_cache=score_cache,
            tracker=tracker
        )

        tracker.update(99, "Complete", item="Rendering results")
        tracker.done("Analysis complete")

        st.success("Analysis complete")

        # Metrics
        total_contests = len(summary_df)
        total_valid = int(summary_df["ValidLineups"].fillna(0).sum()) if not summary_df.empty else 0
        total_invalid = int(summary_df["InvalidLineups"].fillna(0).sum()) if not summary_df.empty else 0
        total_beating = int(summary_df["LineupsBeatingTopScore"].fillna(0).sum()) if not summary_df.empty else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Contests", f"{total_contests:,}")
        m2.metric("Valid lineups", f"{total_valid:,}")
        m3.metric("Invalid lineups", f"{total_invalid:,}")
        m4.metric("Lineups beating top score", f"{total_beating:,}")

        st.subheader("Summary results")
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("Detailed results")
        st.dataframe(detailed_df.head(int(show_detailed_preview_rows)), use_container_width=True)

        summary_csv = summary_df.to_csv(index=False).encode("utf-8-sig")
        detailed_csv = detailed_df.to_csv(index=False).encode("utf-8-sig")

        zip_bytes = make_zip_file({
            "Lineup_BeatTopScore_Summary.csv": summary_csv,
            "Lineup_Results_Detailed.csv": detailed_csv
        })

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "Download summary CSV",
                data=summary_csv,
                file_name="Lineup_BeatTopScore_Summary.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d2:
            st.download_button(
                "Download detailed CSV",
                data=detailed_csv,
                file_name="Lineup_Results_Detailed.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d3:
            st.download_button(
                "Download both as ZIP",
                data=zip_bytes,
                file_name="completed_slate_analysis_outputs.zip",
                mime="application/zip",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.exception(e)
        st.code(traceback.format_exc(), language="python")
