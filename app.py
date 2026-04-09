from flask import Flask, render_template, request, jsonify
from ml import (
    analyze_performance, get_meta_info, get_state_summary,
    battleground_faceoff, get_candidate_history, search_candidates, _DF
)
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/meta")
def meta():
    years, _ = get_meta_info()
    return jsonify({"years": [int(y) for y in years]})


@app.route("/api/states")
def get_states():
    states = sorted(_DF['state_name'].dropna().unique().tolist())
    return jsonify({"states": states})


@app.route("/api/search_candidates")
def search_candidates_api():
    query = request.args.get("query", "")
    year = request.args.get("year", type=int)
    if not query:
        return jsonify([])
    matches = search_candidates(query, year)
    return jsonify(matches)


@app.route("/api/analyze")
def analyze_api():
    year = request.args.get("year", type=int)
    candidate = request.args.get("candidate", "").strip()
    if not year or not candidate:
        return jsonify({"status": "error", "message": "Year and Candidate Name are required."}), 400
    return jsonify(analyze_performance(year, candidate))


@app.route("/api/history")
def history_api():
    candidate = request.args.get("candidate", "").strip()
    if not candidate:
        return jsonify({"status": "error", "message": "Candidate Name is required."}), 400
    return jsonify(get_candidate_history(candidate))


@app.route("/api/top_performers")
def top_performers():
    year = request.args.get("year", type=int)
    if not year:
        return jsonify({"status": "error", "message": "Year is required."}), 400

    df = _DF[_DF["year"] == year].copy()
    df['overperformance_score'] = df['candidate_alpha_score']
    df = df.sort_values(by="overperformance_score", ascending=False).drop_duplicates(subset=['candidate'])
    display_cols = ['candidate', 'party', 'constituency_name', 'overperformance_score']

    top_df = df.head(15)[display_cols].replace({np.nan: None})
    bottom_df = df.tail(15).sort_values(by="overperformance_score", ascending=True)[display_cols].replace({np.nan: None})
    top = top_df.to_dict(orient="records")
    bottom = bottom_df.to_dict(orient="records")

    return jsonify({"status": "success", "top_overperformers": top, "top_underperformers": bottom})


@app.route("/api/state_insights")
def state_insights():
    year = request.args.get("year", type=int)
    state = request.args.get("state", "").strip()
    if not year or not state:
        return jsonify({"status": "error", "message": "Year and State are required."}), 400
    return jsonify(get_state_summary(year, state))


@app.route("/api/state_seats")
def state_seats():
    year = request.args.get("year", type=int)
    state = request.args.get("state", "").strip()
    if not year or not state:
        return jsonify({"status": "error", "message": "Year and State are required."}), 400

    df = _DF[(_DF["year"] == year) & (_DF["state_name"].str.lower() == state.lower())].copy()
    if df.empty:
        return jsonify({"status": "error", "message": "No seat data found for the given state and year."})

    seats = []
    for const_name, group in df.groupby("constituency_name"):
        group_sorted = group.sort_values("position")
        top_cands = []
        for _, row in group_sorted.head(5).iterrows():
            alpha = row.get("candidate_alpha_score", np.nan)
            expected = row.get("baseline_expected_vs", np.nan)
            actual = row.get("vote_share_%", np.nan)
            top_cands.append({
                "candidate": row["candidate"],
                "party": row["party"],
                "position": int(row["position"]) if not pd.isna(row["position"]) else None,
                "actual_vs": round(float(actual), 2) if not pd.isna(actual) else None,
                "expected_vs": round(float(expected), 2) if not pd.isna(expected) else None,
                "alpha": round(float(alpha), 2) if not pd.isna(alpha) else None,
            })
        winner_row = group_sorted[group_sorted["position"] == 1]
        winner_name = winner_row.iloc[0]["candidate"] if not winner_row.empty else "N/A"
        winner_party = winner_row.iloc[0]["party"] if not winner_row.empty else "N/A"
        seats.append({
            "constituency": const_name,
            "winner": winner_name,
            "winner_party": winner_party,
            "candidates": top_cands,
        })

    seats.sort(key=lambda x: x["constituency"])
    return jsonify({"status": "success", "year": year, "state": state, "seats": seats})


@app.route("/api/performance_overview")
def performance_overview():
    year = request.args.get("year", type=int)
    state = request.args.get("state", "").strip()
    if not year:
        return jsonify({"status": "error", "message": "Year is required."}), 400

    df = _DF[_DF["year"] == year].copy()
    if state and state.lower() != "all":
        df = df[df["state_name"].str.lower() == state.lower()]

    if df.empty:
        return jsonify({"status": "error", "message": "No data found."})

    df["overperformance"] = df["candidate_alpha_score"]
    df = df.sort_values("overperformance", ascending=False).drop_duplicates("candidate")

    cols = ["candidate", "party", "constituency_name", "state_name",
            "vote_share_%", "baseline_expected_vs", "overperformance", "position"]
    df = df[cols].replace({np.nan: None})
    df.rename(columns={"vote_share_%": "actual_vs", "baseline_expected_vs": "expected_vs"}, inplace=True)

    return jsonify({"status": "success", "data": df.to_dict(orient="records")})


@app.route("/api/battleground")
def battleground():
    year = request.args.get("year", type=int)
    state = request.args.get("state", "").strip()
    constituency = request.args.get("constituency", "").strip()
    cand1 = request.args.get("cand1", "").strip()
    cand2 = request.args.get("cand2", "").strip()
    if not all([year, state, constituency, cand1, cand2]):
        return jsonify({"status": "error", "message": "All fields are required."}), 400
    return jsonify(battleground_faceoff(year, state, constituency, cand1, cand2))


if __name__ == "__main__":
    app.run(debug=True)
