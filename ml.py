import pandas as pd
import numpy as np
import lightgbm as lgb
from rapidfuzz import process, fuzz
import re

# CHANGE 1: point to the pre-normalised CSV
CSV_FILE = "All_States_GE_normalized.csv"

def _normalize_name(name):
    """
    Cleans and standardizes a candidate's name.
    - Converts to lowercase.
    - Removes common titles (dr., smt., shri, etc.).
    - Removes punctuation and extra whitespace.
    """
    if not isinstance(name, str):
        return ""
    
    name = name.lower().strip()

    prefixes = [
        'dr.','dr ','mrs.','mrs ','ms.','ms ','mr.','mr ', 'adv.', 'adv ',
        'prof.','prof ','smt.','smt ','shri.','shri ','ku.','kum.',
        'col.','col ','capt.','capt ','major ','major.','advocate'
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):].lstrip()

    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def _prepare():
   
    df = pd.read_csv(CSV_FILE, low_memory=False)
    df.rename(columns={
        "Year": "year", "State_Name": "state_name", "Constituency_Name": "constituency_name",
        "Candidate": "candidate", "Party": "party", "Votes": "votes_polled",
        "Valid_Votes": "total_votes", "Electors": "total_electors", "Position": "position",
    }, inplace=True)
    df.columns = df.columns.str.lower().str.strip()
    
    for col in ["votes_polled", "total_votes", "total_electors", "position"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df.dropna(subset=["year", "state_name", "constituency_name", "candidate", "party", "votes_polled", "total_votes"], inplace=True)

    # CHANGE 2: use the pre-cleaned Normalized_Candidate column as the base for
    # candidate_normalized. After df.columns.str.lower(), it becomes
    # "normalized_candidate". Titles like Dr./Col./Retd. are already stripped
    # there, so _normalize_name only needs to lowercase and remove punctuation.
    df["candidate_normalized"] = df["normalized_candidate"].apply(_normalize_name)

    df = df[df["candidate_normalized"] != '']

    df["year"] = df["year"].astype(int)
    df["vote_share_%"] = df["votes_polled"] / df["total_votes"] * 100
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["vote_share_%"], inplace=True)
    df.sort_values(["state_name", "constituency_name", "year"], inplace=True)

    st = df.groupby(["year", "state_name", "party"])["vote_share_%"].mean().rename("avg_st").reset_index()
    st["st_prev"] = st.groupby(["state_name", "party"])["avg_st"].shift(1)
    st["state_party_wave"] = st["avg_st"] - st["st_prev"]
    df = df.merge(st[["year", "state_name", "party", "st_prev", "state_party_wave"]], on=["year", "state_name", "party"], how="left")

    nat = df.groupby(["year", "party"])["vote_share_%"].mean().rename("avg_nat").reset_index()
    nat["nat_prev"] = nat.groupby("party")["avg_nat"].shift(1)
    nat["nat_party_wave"] = nat["avg_nat"] - nat["nat_prev"]
    df = df.merge(nat[["year", "party", "nat_party_wave"]], on=["year", "party"], how="left")

    df.sort_values(["candidate_normalized", "year"], inplace=True)

    df["candidate_prev_win"] = df.groupby("candidate_normalized")["position"].shift(1).fillna(0).apply(lambda x: 1 if x == 1 else 0)
    df["seat_party_3cycle_avg"] = (df.groupby(["constituency_name", "party"])["vote_share_%"].rolling(window=3, min_periods=1).mean().reset_index(level=[0, 1], drop=True))
    
    features = ["seat_party_3cycle_avg", "state_party_wave", "candidate_prev_win"]
    df[features] = df[features].fillna(0)
    target = "vote_share_%"
    train = df[df["year"] < 2019].dropna(subset=features + [target])
    
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(train[features], train[target])
    
    df["baseline_expected_vs"] = model.predict(df[features])
    df["candidate_alpha_score"] = df["vote_share_%"] - df["baseline_expected_vs"]
    
    return df

_DF = _prepare()

_UNIQUE_CANDIDATES = sorted(_DF.drop_duplicates(subset=['candidate_normalized'])['candidate'].dropna().unique().tolist())


def search_candidates(query, year=None):
    choices = _UNIQUE_CANDIDATES
    if year:
        year_cands_normalized = _DF[_DF.year == int(year)]["candidate_normalized"].unique()
        year_cands = sorted(_DF[_DF['candidate_normalized'].isin(year_cands_normalized)]['candidate'].unique().tolist())
        if year_cands:
            choices = year_cands
    matches = process.extract(query, choices, scorer=fuzz.WRatio, limit=10, score_cutoff=75)
    return [match[0] for match in matches]


def _calculate_electoral_strength(candidate_name, party, year, constituency):
    normalized_cand_name = _normalize_name(candidate_name)

    seat_history = _DF[(_DF.constituency_name == constituency) & (_DF.party == party) & (_DF.year < year)].sort_values('year')
    base_seat_strength = seat_history["seat_party_3cycle_avg"].iloc[-1] if not seat_history.empty else 0
    
    if base_seat_strength == 0:
        state_name_series = _DF.loc[_DF.constituency_name == constituency, 'state_name']
        if not state_name_series.empty:
            state_name = state_name_series.iloc[0]
            state_history = _DF[(_DF.state_name == state_name) & (_DF.party == party) & (_DF.year < year)]
            base_seat_strength = state_history['vote_share_%'].mean() if not state_history.empty else 10.0
        else:
            base_seat_strength = 10.0

    state_name = _DF.loc[_DF.constituency_name == constituency, 'state_name'].iloc[0]
    wave_record = _DF[(_DF.year == year) & (_DF.state_name == state_name) & (_DF.party == party)]
    state_wave = wave_record['state_party_wave'].mean() if not wave_record.empty and not np.isnan(wave_record['state_party_wave'].mean()) else 0.0

    cand_history = _DF[(_DF.candidate_normalized == normalized_cand_name) & (_DF.year < year)]
    candidate_alpha = cand_history['candidate_alpha_score'].mean() if not cand_history.empty and not np.isnan(cand_history['candidate_alpha_score'].mean()) else 0.0
    
    wave_effect = max(-10, min(10, state_wave))
    alpha_effect = max(-10, min(10, candidate_alpha))
    strength_score = base_seat_strength + wave_effect + alpha_effect
    return max(5.0, min(70.0, strength_score))


def analyze_performance(year, candidate_name):
    normalized_name = _normalize_name(candidate_name)
    main_cand_record = _DF[(_DF.year == year) & (_DF.candidate_normalized == normalized_name)]
    
    if main_cand_record.empty:
        return {"status": "error", "message": "Candidate did not contest in the selected year or could not be found."}
    
    record = main_cand_record.iloc[0]
    constituency = record.constituency_name
    party = record.party
    actual_vs = record["vote_share_%"]
    
    expected_vs = _calculate_electoral_strength(candidate_name, party, year, constituency)
    overperformance = actual_vs - expected_vs
    
    if abs(overperformance) <= 2.0:
        tag = "AS EXPECTED"
    elif overperformance > 2.0:
        tag = "OVERPERFORMED"
    else:
        tag = "UNDERPERFORMED"

    return {"status": "success", "data": [{
        "candidate": record.candidate, 
        "party": party,
        "constituency": f"{constituency}, {record.state_name} – {year}",
        "actual_vs": round(actual_vs, 2),
        "expected_vs": round(expected_vs, 2),
        "performance_tag": tag,
        "performance_score": round(overperformance, 2)
    }]}


def get_candidate_history(name):
    normalized_name = _normalize_name(name)
    df = _DF[_DF["candidate_normalized"] == normalized_name]
    
    if df.empty:
        return {"status": "error", "message": "No history found for this candidate."}
    
    history_df = df.sort_values("year").copy()
    history_df.rename(columns={"vote_share_%": "vote_share"}, inplace=True)
    display_cols = ["year", "state_name", "constituency_name", "party", "vote_share", "position"]
    history_df.replace({np.nan: None}, inplace=True)
    history = history_df[display_cols].to_dict(orient="records")
    return {"status": "success", "history": history}


def battleground_faceoff(year, state, constituency, cand1_name, cand2_name):
    norm_cand1 = _normalize_name(cand1_name)
    norm_cand2 = _normalize_name(cand2_name)

    if norm_cand1 == "" or norm_cand2 == "":
        return {"status": "error", "message": "One or both candidate names are invalid."}

    party1_series = _DF.loc[_DF.candidate_normalized == norm_cand1].sort_values('year', ascending=False)['party']
    party2_series = _DF.loc[_DF.candidate_normalized == norm_cand2].sort_values('year', ascending=False)['party']

    if party1_series.empty or party2_series.empty:
        return {"status": "error", "message": "Could not determine party for one or both candidates from historical data."}
    
    party1 = party1_series.iloc[0]
    party2 = party2_series.iloc[0]

    score1 = _calculate_electoral_strength(cand1_name, party1, year, constituency)
    score2 = _calculate_electoral_strength(cand2_name, party2, year, constituency)

    total_score = score1 + score2
    if total_score == 0:
        proj1, proj2 = 50.0, 50.0
    else:
        proj1 = round(100 * score1 / total_score, 2)
        proj2 = round(100 * score2 / total_score, 2)

    winner = cand1_name if proj1 > proj2 else cand2_name
    if proj1 == proj2: winner = "Toss-up"

    return {
        "status": "success",
        "simulation": {
            "year": year, "state": state.title(), "constituency": constituency.title(),
            "cand1": {"name": cand1_name, "party": party1, "projection": proj1},
            "cand2": {"name": cand2_name, "party": party2, "projection": proj2},
            "winner": winner
        }
    }


def get_meta_info():
    years = sorted(_DF["year"].unique().tolist(), reverse=True)
    return years, _UNIQUE_CANDIDATES


def get_state_summary(year, state):
    df = _DF[(_DF.year == year) & (_DF.state_name.str.lower() == state.lower())]
    if df.empty:
        return {"status": "error", "message": "No data for given state and year"}
    party_votes = df.groupby("party")["votes_polled"].sum()
    constituency_totals = df[['constituency_name', 'total_votes']].drop_duplicates()
    total_state_votes = constituency_totals['total_votes'].sum()
    if total_state_votes == 0:
        return {"status": "error", "message": "Vote data is not available to calculate summary."}
    summary = (party_votes / total_state_votes * 100).sort_values(ascending=False).reset_index()
    summary.rename(columns={"votes_polled": "vote_share_%"}, inplace=True)
    return {
        "status": "success", "year": year,
        "state": state.title(),
        "party_summary": summary.to_dict(orient="records")
    }
