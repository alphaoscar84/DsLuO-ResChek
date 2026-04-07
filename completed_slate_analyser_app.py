import math
import re
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import pulp


# =========================================================
# DEFAULTS
# =========================================================
DEFAULTS = {
    "SALARY_CAP": 100000,
    "LINEUP_SIZE": 9,
    "REQ_DEF": 2,
    "REQ_MID": 4,
    "REQ_RK": 1,
    "REQ_FWD": 2,
    "MIN_DIFFERENT_OPTIONS": [1, 2, 3, 4],
    "MAX_TEAM_OPTIONS": [4, 5, 6, 7, 8],
    "MAX_SHARE_OPTIONS": [0.60, 0.65, 0.70, 0.75],
    "TOP_N_LINEUPS": 50,
    "PLAYING_STATUS_REQUIRED_TEXT": "IN TEAM TO PLAY",
    "SOLVER_TIME_LIMIT": 20,
    "FALLBACK_PROJECTION": 41.22,
}

TEAM_ALIASES = {
    "adelaide": "Crows",
    "crows": "Crows",
    "brisbane": "Lions",
    "brisbane lions": "Lions",
    "lions": "Lions",
    "carlton": "Blues",
    "blues": "Blues",
    "collingwood": "Magpies",
    "magpies": "Magpies",
    "essendon": "Bombers",
    "bombers": "Bombers",
    "fremantle": "Dockers",
    "dockers": "Dockers",
    "geelong": "Cats",
    "cats": "Cats",
    "gold coast": "Suns",
    "gold coast suns": "Suns",
    "suns": "Suns",
    "greater western sydney": "Giants",
    "gws": "Giants",
    "giants": "Giants",
    "hawthorn": "Hawks",
    "hawks": "Hawks",
    "melbourne": "Demons",
    "demons": "Demons",
    "north melbourne": "Kangaroos",
    "kangaroos": "Kangaroos",
    "port adelaide": "Power",
    "power": "Power",
    "richmond": "Tigers",
    "tigers": "Tigers",
    "st kilda": "Saints",
    "saints": "Saints",
    "sydney": "Swans",
    "swans": "Swans",
    "west coast": "Eagles",
    "eagles": "Eagles",
    "western bulldogs": "Bulldogs",
    "bulldogs": "Bulldogs",
}

POS_PREFIX = {"DEF": "D", "MID": "M", "FWD": "F", "RK": "R"}


# =========================================================
# GENERAL HELPERS
# =========================================================
def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default



def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default



def find_first_existing_column(df, candidates, required=True):
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None



def normalise_team(team):
    if pd.isna(team):
        return ""
    key = str(team).strip().lower()
    return TEAM_ALIASES.get(key, str(team).strip())



def parse_positions(pos_value):
    if pd.isna(pos_value):
        return []
    s = str(pos_value).upper().replace(" ", "")
    s = s.replace("RUC", "RK")
    parts = [p for p in s.replace("/", ",").split(",") if p]
    out = []
    for p in parts:
        if p in {"DEF", "MID", "FWD", "RK"}:
            out.append(p)
    return list(dict.fromkeys(out))



def extract_contest_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    m = re.search(r"players[_\-]?([A-Za-z0-9]+)", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    return stem



def normalise_matchup(team_a: str, team_b: str) -> str:
    parts = sorted([normalise_team(team_a), normalise_team(team_b)])
    return " vs ".join(parts)



def build_team_to_matchup_map(df: pd.DataFrame) -> Dict[str, str]:
    mapping = {}
    for _, row in df[["TeamNick", "OpponentNick"]].drop_duplicates().iterrows():
        team = row["TeamNick"]
        opp = row["OpponentNick"]
        if team and opp:
            mapping[team] = normalise_matchup(team, opp)
    return mapping



def get_slate_shape(df: pd.DataFrame) -> Dict[str, object]:
    teams = sorted(t for t in df["TeamNick"].dropna().astype(str).unique() if t.strip())
    matchup_set = set()
    for _, row in df[["TeamNick", "OpponentNick"]].drop_duplicates().iterrows():
        if row["TeamNick"] and row["OpponentNick"]:
            matchup_set.add(normalise_matchup(row["TeamNick"], row["OpponentNick"]))
    return {
        "teams": teams,
        "team_count": len(teams),
        "matchups": sorted(matchup_set),
        "matchup_count": len(matchup_set),
        "is_multi_game": len(teams) >= 3,
    }



def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"



def update_progress(progress_bar, status_box, current_step: int, total_steps: int, started: float, message: str):
    ratio = 0.0 if total_steps <= 0 else min(max(current_step / total_steps, 0.0), 1.0)
    elapsed = time.time() - started
    eta = ((elapsed / current_step) * (total_steps - current_step)) if current_step > 0 else 0.0
    text = (
        f"{ratio * 100:,.1f}% complete | Step {current_step:,}/{total_steps:,} | "
        f"Elapsed: {format_seconds(elapsed)} | ETA: {format_seconds(eta)}"
    )
    progress_bar.progress(ratio, text=text)
    status_box.info(message)


# =========================================================
# PREP
# =========================================================
def prepare_players_df(players: pd.DataFrame, required_status_text: str) -> pd.DataFrame:
    name_col = find_first_existing_column(players, ["Name"])
    team_col = find_first_existing_column(players, ["Team"])
    salary_col = find_first_existing_column(players, ["Salary"])
    pos_col = find_first_existing_column(players, ["Position", "Pos"])

    opponent_col = find_first_existing_column(players, ["Opponent"], required=False)
    score_col = find_first_existing_column(players, ["Score", "Pts"], required=False)
    form_col = find_first_existing_column(players, ["Form"], required=False)
    fppg_col = find_first_existing_column(players, ["FPPG", "Avg", "Average"], required=False)
    status_col = find_first_existing_column(players, ["Playing Status", "Status"], required=False)

    df = players.copy().rename(columns={
        name_col: "Name",
        team_col: "Team",
        salary_col: "Salary",
        pos_col: "PositionRaw",
    })

    df["Opponent"] = df[opponent_col] if opponent_col else ""
    df["ActualScore"] = df[score_col] if score_col else pd.NA
    df["Form"] = df[form_col] if form_col else pd.NA
    df["FPPG"] = df[fppg_col] if fppg_col else pd.NA
    df["Playing Status"] = df[status_col] if status_col else ""

    df["Name"] = df["Name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["Team"] = df["Team"].astype(str).str.strip()
    df["Opponent"] = df["Opponent"].astype(str).str.strip()
    df["TeamNick"] = df["Team"].map(normalise_team)
    df["OpponentNick"] = df["Opponent"].map(normalise_team)
    df["Salary"] = df["Salary"].apply(safe_int)
    df["ActualScore"] = df["ActualScore"].apply(lambda x: safe_float(x, default=math.nan))
    df["Form"] = df["Form"].apply(lambda x: safe_float(x, default=math.nan))
    df["FPPG"] = df["FPPG"].apply(lambda x: safe_float(x, default=math.nan))
    df["Playing Status"] = df["Playing Status"].astype(str).str.strip()
    df["EligiblePositions"] = df["PositionRaw"].map(parse_positions)

    df = df[df["Playing Status"].str.contains(required_status_text, case=False, na=False)].copy()
    df = df[df["EligiblePositions"].map(len) > 0].copy()
    df = df[pd.notna(df["ActualScore"])].copy()
    return df



def prepare_merged_df(merged: pd.DataFrame) -> pd.DataFrame:
    player_col = find_first_existing_column(merged, ["Player", "Name"])
    team_col = find_first_existing_column(merged, ["Team", "Team_2026", "Team_2025"], required=False)
    avg_col = find_first_existing_column(merged, ["Average", "NewAverage", "MergedAverage"])
    games_col = find_first_existing_column(merged, ["TotalGames", "Total_Games", "Total Games", "Games", "Gm"], required=False)

    df = merged.copy()
    rename_map = {player_col: "Player", avg_col: "MergedAverage"}
    if team_col:
        rename_map[team_col] = "Team"
    df = df.rename(columns=rename_map)

    if "Team" not in df.columns:
        df["Team"] = ""

    df["Total_Games"] = df[games_col] if games_col else 0
    df["Player"] = df["Player"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["Team"] = df["Team"].astype(str).str.strip()
    df["TeamNick"] = df["Team"].map(normalise_team)
    df["MergedAverage"] = df["MergedAverage"].apply(safe_float)
    df["Total_Games"] = df["Total_Games"].apply(safe_int)

    return df[["Player", "Team", "TeamNick", "MergedAverage", "Total_Games"]].copy()


def match_players(players_df: pd.DataFrame, merged_df: pd.DataFrame, fallback_projection: float) -> pd.DataFrame:
    out = players_df.merge(
        merged_df,
        left_on="Name",
        right_on="Player",
        how="left",
        suffixes=("_players", "_merged"),
    )

    out["MatchMethod"] = out["MergedAverage"].notna().map(lambda x: "full-name" if x else "")
    unmatched_mask = out["MergedAverage"].isna()

    if unmatched_mask.any():
        fallback = players_df.merge(
            merged_df,
            left_on=["Name", "TeamNick"],
            right_on=["Player", "TeamNick"],
            how="left",
            suffixes=("_players", "_merged"),
        )
        out.loc[unmatched_mask, "MergedAverage"] = fallback.loc[unmatched_mask, "MergedAverage"].values
        out.loc[unmatched_mask, "Total_Games"] = fallback.loc[unmatched_mask, "Total_Games"].values

        fallback_match_mask = pd.notna(fallback.loc[unmatched_mask, "MergedAverage"]).values
        out.loc[unmatched_mask, "MatchMethod"] = [
            "full-name+team" if matched else ""
            for matched in fallback_match_mask
        ]

    if "Team_players" in out.columns:
        out["Team"] = out["Team_players"]
    if "TeamNick_players" in out.columns:
        out["TeamNick"] = out["TeamNick_players"]

    out["ProjectedAverage"] = out["MergedAverage"].fillna(float(fallback_projection))

    expanded_rows = []
    unique_names = sorted(out["Name"].unique())
    name_to_bit = {name: i for i, name in enumerate(unique_names)}

    for _, row in out.iterrows():
        for pos in row["EligiblePositions"]:
            rec = row.to_dict()
            rec["Position"] = pos
            rec["PlayerKey"] = rec["Name"]
            rec["PlayerInternalID"] = name_to_bit[rec["Name"]]
            expanded_rows.append(rec)

    expanded = pd.DataFrame(expanded_rows).reset_index(drop=True)
    expanded["RowID"] = expanded.index

    expanded["PosRankNum"] = expanded.groupby("Position")["ProjectedAverage"].rank(method="first", ascending=False).astype(int)
    expanded["RankLabel"] = expanded.apply(
        lambda r: f"{POS_PREFIX.get(r['Position'], r['Position'][0])}{int(r['PosRankNum'])}",
        axis=1,
    )
    expanded["ActualRankNum"] = expanded.groupby("Position")["ActualScore"].rank(method="first", ascending=False)
    expanded["ActualRankLabel"] = expanded.apply(
        lambda r: f"{POS_PREFIX.get(r['Position'], r['Position'][0])}{int(r['ActualRankNum'])}" if pd.notna(r["ActualRankNum"]) else "",
        axis=1,
    )

    team_to_matchup = build_team_to_matchup_map(expanded)
    expanded["MatchupKey"] = expanded["TeamNick"].map(team_to_matchup).fillna("")
    return expanded



def derive_projection(row: pd.Series, fallback_projection: float) -> float:
    form = safe_float(row.get("Form"), default=math.nan)
    fppg = safe_float(row.get("FPPG"), default=math.nan)

    valid = [x for x in [form, fppg] if pd.notna(x)]
    if valid:
        return round(sum(valid) / len(valid), 2)
    return float(fallback_projection)



def build_expanded_from_single_csv(players_df: pd.DataFrame, fallback_projection: float) -> pd.DataFrame:
    out = players_df.copy()
    out["ProjectedAverage"] = out.apply(lambda r: derive_projection(r, fallback_projection=fallback_projection), axis=1)
    out["MatchMethod"] = out.apply(
        lambda r: "form+fppg"
        if pd.notna(r.get("Form")) and pd.notna(r.get("FPPG"))
        else ("form" if pd.notna(r.get("Form")) else ("fppg" if pd.notna(r.get("FPPG")) else "fallback")),
        axis=1,
    )

    expanded_rows = []
    unique_names = sorted(out["Name"].unique())
    name_to_bit = {name: i for i, name in enumerate(unique_names)}

    for _, row in out.iterrows():
        for pos in row["EligiblePositions"]:
            rec = row.to_dict()
            rec["Position"] = pos
            rec["PlayerKey"] = rec["Name"]
            rec["PlayerInternalID"] = name_to_bit[rec["Name"]]
            expanded_rows.append(rec)

    expanded = pd.DataFrame(expanded_rows).reset_index(drop=True)
    expanded["RowID"] = expanded.index

    expanded["PosRankNum"] = expanded.groupby("Position")["ProjectedAverage"].rank(method="first", ascending=False).astype(int)
    expanded["RankLabel"] = expanded.apply(
        lambda r: f"{POS_PREFIX.get(r['Position'], r['Position'][0])}{int(r['PosRankNum'])}",
        axis=1,
    )
    expanded["ActualRankNum"] = expanded.groupby("Position")["ActualScore"].rank(method="first", ascending=False)
    expanded["ActualRankLabel"] = expanded.apply(
        lambda r: f"{POS_PREFIX.get(r['Position'], r['Position'][0])}{int(r['ActualRankNum'])}" if pd.notna(r["ActualRankNum"]) else "",
        axis=1,
    )

    team_to_matchup = build_team_to_matchup_map(expanded)
    expanded["MatchupKey"] = expanded["TeamNick"].map(team_to_matchup).fillna("")
    return expanded


# =========================================================
# SOLVER
# =========================================================
def solve_top_n_projected_lineups(expanded: pd.DataFrame, settings: Dict[str, object]) -> List[Dict[str, object]]:
    df = expanded.copy().reset_index(drop=True)
    row_ids = df["RowID"].tolist()
    row_to_record = df.set_index("RowID").to_dict("index")

    pos_to_rows = {
        "DEF": df.loc[df["Position"] == "DEF", "RowID"].tolist(),
        "MID": df.loc[df["Position"] == "MID", "RowID"].tolist(),
        "RK": df.loc[df["Position"] == "RK", "RowID"].tolist(),
        "FWD": df.loc[df["Position"] == "FWD", "RowID"].tolist(),
    }
    player_to_rows = df.groupby("PlayerKey")["RowID"].apply(list).to_dict()
    team_to_rows = df.groupby("TeamNick")["RowID"].apply(list).to_dict()
    matchup_to_rows = df.groupby("MatchupKey")["RowID"].apply(list).to_dict()

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=settings["SOLVER_TIME_LIMIT"])

    previous_lineups_playerkeys: List[List[str]] = []
    player_lineup_counts = {player_key: 0 for player_key in player_to_rows.keys()}
    solved = []

    max_player_lineups = max(
        math.floor(int(settings["TOP_N_LINEUPS"]) * float(settings["MAX_PLAYER_LINEUP_SHARE"])),
        1,
    )

    for lineup_idx in range(1, int(settings["TOP_N_LINEUPS"]) + 1):
        prob = pulp.LpProblem(f"ProjectedAverage_Lineup_{lineup_idx}", pulp.LpMaximize)
        x = {rid: pulp.LpVariable(f"x_{rid}", cat="Binary") for rid in row_ids}

        prob += pulp.lpSum(safe_float(row_to_record[rid]["ProjectedAverage"]) * x[rid] for rid in row_ids)
        prob += pulp.lpSum(safe_int(row_to_record[rid]["Salary"]) * x[rid] for rid in row_ids) <= int(settings["SALARY_CAP"])

        req_by_pos = {
            "DEF": int(settings["REQ_DEF"]),
            "MID": int(settings["REQ_MID"]),
            "FWD": int(settings["REQ_FWD"]),
            "RK": int(settings["REQ_RK"]),
        }
        for pos, req in req_by_pos.items():
            prob += pulp.lpSum(x[rid] for rid in pos_to_rows[pos]) == req

        for _, rids in player_to_rows.items():
            prob += pulp.lpSum(x[rid] for rid in rids) <= 1

        for player_key, used_count in player_lineup_counts.items():
            if used_count >= max_player_lineups:
                prob += pulp.lpSum(x[rid] for rid in player_to_rows[player_key]) == 0

        max_players_per_team = int(settings["MAX_PLAYERS_PER_TEAM_EFFECTIVE"])
        for _, rids in team_to_rows.items():
            prob += pulp.lpSum(x[rid] for rid in rids) <= max_players_per_team

        if bool(settings["USE_MATCHUP_CAP"]):
            matchup_cap = int(settings["MATCHUP_CAP"])
            for matchup_key, rids in matchup_to_rows.items():
                if matchup_key:
                    prob += pulp.lpSum(x[rid] for rid in rids) <= matchup_cap

        min_diff = int(settings["MIN_DIFFERENT_PLAYERS_FROM_PREVIOUS"])
        for prev_player_keys in previous_lineups_playerkeys:
            prev_rows = []
            for player_key in prev_player_keys:
                prev_rows.extend(player_to_rows[player_key])
            prob += pulp.lpSum(x[rid] for rid in prev_rows) <= int(settings["LINEUP_SIZE"]) - min_diff

        status = prob.solve(solver)
        if pulp.LpStatus[status] != "Optimal":
            break

        selected_rids = [rid for rid in row_ids if pulp.value(x[rid]) > 0.5]
        if len(selected_rids) != int(settings["LINEUP_SIZE"]):
            break

        players = [row_to_record[rid] for rid in selected_rids]
        player_keys = [p["PlayerKey"] for p in players]
        previous_lineups_playerkeys.append(player_keys)

        for player_key in set(player_keys):
            player_lineup_counts[player_key] += 1

        solved.append({
            "lineup_no": lineup_idx,
            "projected_score": round(sum(safe_float(p["ProjectedAverage"]) for p in players), 2),
            "actual_score": round(sum(safe_float(p["ActualScore"]) for p in players), 2),
            "salary": sum(safe_int(p["Salary"]) for p in players),
            "players": players,
        })

    return solved


# =========================================================
# SUMMARIES
# =========================================================
def build_lineup_rows(solved: List[Dict[str, object]], combo_label: str, slate_id: str) -> pd.DataFrame:
    rows = []
    for lineup in solved:
        by_pos = {"DEF": [], "MID": [], "RK": [], "FWD": []}
        for p in lineup["players"]:
            by_pos[p["Position"]].append(p)
        for pos in by_pos:
            by_pos[pos] = sorted(by_pos[pos], key=lambda x: safe_float(x["ProjectedAverage"]), reverse=True)

        row = {
            "SlateID": slate_id,
            "Combo": combo_label,
            "LineupNo": lineup["lineup_no"],
            "ProjectedScore": lineup["projected_score"],
            "ActualScore": lineup["actual_score"],
            "Salary": lineup["salary"],
        }
        for pos in ["DEF", "MID", "RK", "FWD"]:
            for idx, p in enumerate(by_pos[pos], start=1):
                row[f"{pos}{idx}"] = p["Name"]
                row[f"{pos}{idx}_Team"] = p["TeamNick"]
                row[f"{pos}{idx}_Proj"] = round(safe_float(p["ProjectedAverage"]), 2)
                row[f"{pos}{idx}_Actual"] = round(safe_float(p["ActualScore"]), 2)
                row[f"{pos}{idx}_ProjRank"] = p.get("RankLabel", "")
                row[f"{pos}{idx}_ActualRank"] = p.get("ActualRankLabel", "")
        rows.append(row)
    return pd.DataFrame(rows)



def summarise_vs_manual_thresholds(
    slate_id: str,
    combo_label: str,
    solved: List[Dict[str, object]],
    top_score: float,
    min_prize_score: float,
    settings: Dict[str, object],
    slate_shape: Dict[str, object],
    fallback_projection_count: int,
) -> Dict[str, object]:
    projected_scores = sorted([safe_float(x["projected_score"]) for x in solved], reverse=True)
    actual_scores = sorted([safe_float(x["actual_score"]) for x in solved], reverse=True)

    return {
        "SlateID": slate_id,
        "ContestID": slate_id,
        "Combo": combo_label,
        "MinDifferent": int(settings["MIN_DIFFERENT_PLAYERS_FROM_PREVIOUS"]),
        "MaxPlayersPerTeamInput": int(settings["MAX_PLAYERS_PER_TEAM"]),
        "MaxPlayersPerTeamEffective": int(settings["MAX_PLAYERS_PER_TEAM_EFFECTIVE"]),
        "MaxPlayerLineupShare": float(settings["MAX_PLAYER_LINEUP_SHARE"]),
        "UseMatchupCap": bool(settings["USE_MATCHUP_CAP"]),
        "MatchupCap": int(settings["MATCHUP_CAP"]) if bool(settings["USE_MATCHUP_CAP"]) else "",
        "TeamsOnSlate": slate_shape["team_count"],
        "MatchupsOnSlate": slate_shape["matchup_count"],
        "SolvedLineups": len(actual_scores),
        "BestProjectedGeneratedScore": projected_scores[0] if projected_scores else math.nan,
        "BestActualGeneratedScore": actual_scores[0] if actual_scores else math.nan,
        "MeanTop25ProjectedGeneratedScore": round(sum(projected_scores[:25]) / min(len(projected_scores), 25), 2) if projected_scores else math.nan,
        "MeanTop25ActualGeneratedScore": round(sum(actual_scores[:25]) / min(len(actual_scores), 25), 2) if actual_scores else math.nan,
        "Generated>=TopScore": sum(1 for s in actual_scores if s >= top_score),
        "Generated>=MinPrizeScore": sum(1 for s in actual_scores if s >= min_prize_score),
        "TopScore": top_score,
        "MinPrizeScore": min_prize_score,
        "Top25GeneratedActualScores": ", ".join(f"{x:.2f}" for x in actual_scores[:25]),
        "Top25GeneratedProjectedScores": ", ".join(f"{x:.2f}" for x in projected_scores[:25]),
        "FallbackProjectionPlayers": fallback_projection_count,
    }


# =========================================================
# MAIN RUNNER
# =========================================================
def build_combo_grid(is_multi_game: bool) -> List[Tuple[int, int, float, int]]:
    deduped = []
    seen = set()
    for min_diff in DEFAULTS["MIN_DIFFERENT_OPTIONS"]:
        for max_team_input in reversed(DEFAULTS["MAX_TEAM_OPTIONS"]):
            effective_team = min(max_team_input, 5) if is_multi_game else max_team_input
            for max_share in DEFAULTS["MAX_SHARE_OPTIONS"]:
                item = (min_diff, max_team_input, max_share, effective_team)
                key = (min_diff, effective_team, max_share)
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)
    return deduped



def analyse_one_slate(uploaded_file, merged_df_master: pd.DataFrame, required_status_text: str, fallback_projection: float, top_score: float, min_prize_score: float):
    players = pd.read_csv(uploaded_file)
    players_df = prepare_players_df(players, required_status_text=required_status_text)
    if players_df.empty:
        raise ValueError(f"No valid players remained after filtering in {uploaded_file.name}.")

    expanded = match_players(players_df, merged_df_master, fallback_projection=fallback_projection)
    if expanded.empty:
        raise ValueError(f"No expanded player-position rows were built for {uploaded_file.name}.")

    contest_id = extract_contest_id_from_filename(uploaded_file.name)
    slate_shape = get_slate_shape(expanded)
    combo_grid = build_combo_grid(slate_shape["is_multi_game"])
    fallback_projection_count = expanded.loc[expanded["MatchMethod"] == "", ["Name"]].drop_duplicates().shape[0]

    return players_df, expanded, contest_id, slate_shape, combo_grid, fallback_projection_count, top_score, min_prize_score


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Completed Slate Analyser", layout="wide")
st.title("Completed Slate Analyser")
st.caption("Upload one completed players_xxxx CSV plus your merged averages file, generate lineups from merged-average rankings, and compare generated actual scores against your manually entered slate thresholds.")

with st.sidebar:
    st.header("Inputs")
    players_file = st.file_uploader("Completed players_xxxx CSV file", type=["csv"], accept_multiple_files=False)
    merged_file = st.file_uploader("Merged averages file", type=["csv", "xlsx", "xls"], accept_multiple_files=False)
    top_score = st.number_input("Actual top score from slate", min_value=0.0, value=1000.0, step=0.1)
    min_prize_score = st.number_input("Minimum score that won a prize", min_value=0.0, value=850.0, step=0.1)
    required_status_text = st.text_input("Required Playing Status text", value=DEFAULTS["PLAYING_STATUS_REQUIRED_TEXT"])
    fallback_projection = st.number_input("Fallback projection (used if player missing from merged file)", min_value=0.0, value=float(DEFAULTS["FALLBACK_PROJECTION"]), step=0.1)

    st.header("Assumptions")
    st.write("• Single completed-slate CSV plus merged averages file")
    st.write("• Projection = merged average where matched by player name")
    st.write("• Fallback match also tries player name + team")
    st.write("• If still unmatched, fallback projection is used")
    st.write("• Top 25 lineups are generated per parameter combination")
    st.write("• Salary cap = 100,000")
    st.write("• Roster = 2 DEF / 4 MID / 2 FWD / 1 RK")
    st.write("• On 3+ team slates, max per team is capped at 5 and max per matchup is also capped at 5")

run_button = st.button("Analyse completed slate", type="primary")

if run_button:
    try:
        if players_file is None:
            st.error("Please upload one completed players_xxxx.csv file.")
            st.stop()
        if merged_file is None:
            st.error("Please upload the merged averages file.")
            st.stop()

        if merged_file.name.lower().endswith(".csv"):
            merged_raw = pd.read_csv(merged_file)
        else:
            merged_raw = pd.read_excel(merged_file)
        merged_df_master = prepare_merged_df(merged_raw)

        all_combo = []
        all_lineups = []
        all_fallback_rows = []

        started = time.time()
        progress = st.progress(0.0, text="0.0% complete | Step 0/1 | Elapsed: 0s | ETA: 0s")
        status_box = st.empty()

        total_steps = 3
        update_progress(progress, status_box, 1, total_steps, started, f"Reading merged averages from {merged_file.name}")
        update_progress(progress, status_box, 2, total_steps, started, f"Reading and matching {players_file.name}")
        players_df, expanded, contest_id, slate_shape, combo_grid, fallback_projection_count, top_score, min_prize_score = analyse_one_slate(
            uploaded_file=players_file,
            merged_df_master=merged_df_master,
            required_status_text=required_status_text,
            fallback_projection=float(fallback_projection),
            top_score=float(top_score),
            min_prize_score=float(min_prize_score),
        )

        fallback_detail = expanded.loc[
            expanded["MatchMethod"] == "", ["Name", "TeamNick"]
        ].drop_duplicates().sort_values(["TeamNick", "Name"])
        if not fallback_detail.empty:
            for _, row in fallback_detail.iterrows():
                all_fallback_rows.append({
                    "SlateID": contest_id,
                    "Name": row["Name"],
                    "TeamNick": row["TeamNick"],
                    "FallbackProjection": float(fallback_projection),
                })

        total_steps = 3 + len(combo_grid) + 1
        current_step = 3
        update_progress(progress, status_box, current_step, total_steps, started, f"Prepared slate {contest_id}. Solving {len(combo_grid)} parameter combinations...")

        for combo_index, (min_diff, max_team_input, max_share, max_team_effective) in enumerate(combo_grid, start=1):
            settings = {
                "SALARY_CAP": DEFAULTS["SALARY_CAP"],
                "LINEUP_SIZE": DEFAULTS["LINEUP_SIZE"],
                "REQ_DEF": DEFAULTS["REQ_DEF"],
                "REQ_MID": DEFAULTS["REQ_MID"],
                "REQ_RK": DEFAULTS["REQ_RK"],
                "REQ_FWD": DEFAULTS["REQ_FWD"],
                "TOP_N_LINEUPS": DEFAULTS["TOP_N_LINEUPS"],
                "SOLVER_TIME_LIMIT": DEFAULTS["SOLVER_TIME_LIMIT"],
                "MIN_DIFFERENT_PLAYERS_FROM_PREVIOUS": min_diff,
                "MAX_PLAYERS_PER_TEAM": max_team_input,
                "MAX_PLAYERS_PER_TEAM_EFFECTIVE": max_team_effective,
                "MAX_PLAYER_LINEUP_SHARE": max_share,
                "USE_MATCHUP_CAP": slate_shape["is_multi_game"],
                "MATCHUP_CAP": 5,
            }

            combo_label = f"{min_diff}/{max_team_effective}/{max_share:.2f}"
            update_progress(
                progress,
                status_box,
                current_step,
                total_steps,
                started,
                f"Solving combo {combo_index}/{len(combo_grid)}: {combo_label}"
            )
            solved = solve_top_n_projected_lineups(expanded, settings)

            all_combo.append(
                summarise_vs_manual_thresholds(
                    slate_id=contest_id,
                    combo_label=combo_label,
                    solved=solved,
                    top_score=float(top_score),
                    min_prize_score=float(min_prize_score),
                    settings=settings,
                    slate_shape=slate_shape,
                    fallback_projection_count=fallback_projection_count,
                )
            )
            if solved:
                all_lineups.append(build_lineup_rows(solved, combo_label=combo_label, slate_id=contest_id))

            current_step += 1

        update_progress(progress, status_box, total_steps, total_steps, started, "Finalising outputs...")

        combo_summary_df = pd.DataFrame(all_combo)
        detailed_lineups_df = pd.concat(all_lineups, ignore_index=True) if all_lineups else pd.DataFrame()
        fallback_df = pd.DataFrame(all_fallback_rows)

        if combo_summary_df.empty:
            st.error("No combo summaries were produced.")
            st.stop()

        progress.empty()
        status_box.empty()
        st.success(f"Finished in {format_seconds(time.time() - started)}")

        best_combo = (
            combo_summary_df
            .sort_values(
                [
                    "Generated>=TopScore",
                    "Generated>=MinPrizeScore",
                    "BestActualGeneratedScore",
                    "MeanTop25ActualGeneratedScore",
                ],
                ascending=[False, False, False, False],
            )
            .head(1)
            .reset_index(drop=True)
        )

        overall_combo_rank = (
            combo_summary_df
            .groupby("Combo", as_index=False)
            .agg({
                "Generated>=TopScore": "sum",
                "Generated>=MinPrizeScore": "sum",
                "BestActualGeneratedScore": "mean",
                "MeanTop25ActualGeneratedScore": "mean",
                "SolvedLineups": "sum",
            })
            .sort_values(
                [
                    "Generated>=TopScore",
                    "Generated>=MinPrizeScore",
                    "MeanTop25ActualGeneratedScore",
                ],
                ascending=[False, False, False],
            )
            .reset_index(drop=True)
        )

        st.subheader("Best parameter combination")
        st.dataframe(best_combo, use_container_width=True)

        st.subheader("Parameter combinations ranked")
        st.dataframe(overall_combo_rank, use_container_width=True)

        st.subheader("All combo summaries")
        st.dataframe(combo_summary_df, use_container_width=True)

        with st.expander("Detailed generated lineups"):
            st.dataframe(detailed_lineups_df, use_container_width=True)

        with st.expander("Players unmatched to merged averages and using fallback projection"):
            if fallback_df.empty:
                st.write("No fallback-projection players found.")
            else:
                st.dataframe(fallback_df, use_container_width=True)

        combo_csv = combo_summary_df.to_csv(index=False).encode("utf-8")
        lineups_csv = detailed_lineups_df.to_csv(index=False).encode("utf-8")
        best_csv = best_combo.to_csv(index=False).encode("utf-8")
        overall_csv = overall_combo_rank.to_csv(index=False).encode("utf-8")
        fallback_csv = fallback_df.to_csv(index=False).encode("utf-8")

        st.download_button("Download combo summary CSV", combo_csv, file_name="completed_slate_combo_summary.csv", mime="text/csv")
        st.download_button("Download best combo CSV", best_csv, file_name="completed_slate_best_combo.csv", mime="text/csv")
        st.download_button("Download ranked combos CSV", overall_csv, file_name="completed_slate_ranked_combos.csv", mime="text/csv")
        st.download_button("Download detailed lineups CSV", lineups_csv, file_name="completed_slate_lineups.csv", mime="text/csv")
        st.download_button("Download fallback projection players CSV", fallback_csv, file_name="completed_slate_fallback_projection_players.csv", mime="text/csv")

    except Exception as e:
        st.error(f"ERROR: {e}")
        st.code(traceback.format_exc())
