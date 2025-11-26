import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import traceback
from datetime import datetime

try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    ollama = None
    OLLAMA_AVAILABLE = False

# ---------- CONFIG ----------
st.set_page_config(page_title="EPL Assistant Coach — MVP", layout="wide")
st.title("Premier League Assistant Coach — Enhanced MVP")

MODEL_PKL = "model_package.pkl"
CSV_PATHS = ["epl_matches.csv", "/mnt/data/epl_matches.csv"]
OLLAMA_MODEL = "phi3:mini"  # used only if ollama available

# Global constants
ROLLING_WINDOW = 3

# ---------- UTIL: Loading resources ----------
@st.cache_resource
def load_resources():
    # Load CSV (try multiple locations)
    raw = None
    last_err = None
    for p in CSV_PATHS:
        try:
            raw = pd.read_csv(p)
            break
        except Exception as e:
            last_err = e
            raw = None

    if raw is None:
        # Return partial resources and let the UI show a friendly error
        return {'raw': None, 'model': None, 'team_encoder': None, 'ref_encoder': None, 'res_encoder': None,
                'ref_db': None, 'team_db': None, 'raw_db': None, 'predictors': None, 'load_error': str(last_err)}

    # Ensure Date typed
    if 'Date' in raw.columns:
        try:
            raw['Date'] = pd.to_datetime(raw['Date'], dayfirst=True, errors='coerce')
        except Exception:
            raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')

    # Load ML package safely
    pkg = None
    model = None
    encoders = {}
    ref_db = None
    team_db = None
    raw_db = None
    predictors = None
    try:
        with open(MODEL_PKL, 'rb') as f:
            pkg = pickle.load(f)
    except Exception as e:
        # Could not load model package; continue and return what we have
        return {'raw': raw, 'model': None, 'team_encoder': None, 'ref_encoder': None, 'res_encoder': None,
                'ref_db': None, 'team_db': None, 'raw_db': None, 'predictors': None, 'load_error': str(e)}

    # Normalize keys across different saves
    try:
        model = pkg.get('model') or pkg.get('clf') or pkg
    except Exception:
        model = None
    try:
        encoders = pkg.get('encoders') or pkg.get('enc') or {}
    except Exception:
        encoders = {}
    team_encoder = encoders.get('team') if isinstance(encoders, dict) else getattr(pkg, 'team_encoder', None)
    ref_encoder = encoders.get('ref') if isinstance(encoders, dict) else getattr(pkg, 'ref_encoder', None)
    res_encoder = encoders.get('res') if isinstance(encoders, dict) else getattr(pkg, 'res_encoder', None)

    # older naming support and DBs
    if team_encoder is None and 'team_encoder' in pkg:
        team_encoder = pkg['team_encoder']
    if res_encoder is None and 'result_encoder' in pkg:
        res_encoder = pkg['result_encoder']

    ref_db = pkg.get('ref_db') if 'ref_db' in pkg else pkg.get('ref_stats') if isinstance(pkg, dict) else None
    team_db = pkg.get('team_db') if 'team_db' in pkg else pkg.get('advanced_stats') if isinstance(pkg, dict) else None
    raw_db = pkg.get('raw_db') if 'raw_db' in pkg else raw
    predictors = pkg.get('predictors') if 'predictors' in pkg else None

    return {
        'raw': raw,
        'model': model,
        'team_encoder': team_encoder,
        'ref_encoder': ref_encoder,
        'res_encoder': res_encoder,
        'ref_db': ref_db,
        'team_db': team_db,
        'raw_db': raw_db,
        'predictors': predictors,
        'load_error': None
    }

resources = load_resources()
raw_df = resources.get('raw')
model = resources.get('model')
team_encoder = resources.get('team_encoder')
ref_encoder = resources.get('ref_encoder')
res_encoder = resources.get('res_encoder')
ref_db = resources.get('ref_db')
team_db = resources.get('team_db')
predictors = resources.get('predictors')
load_error = resources.get('load_error')

# Prediction availability flag
PREDICTION_ENABLED = True
if raw_df is None:
    PREDICTION_ENABLED = False
    st.warning(f"Raw CSV not found or failed to load. Check that one of {CSV_PATHS} exists. Error: {load_error}")

if model is None:
    PREDICTION_ENABLED = False
    st.warning(f"Model package not loaded ({MODEL_PKL}). Predictions disabled. Error: {load_error}")

if not predictors:
    # disable predictions if we don't have predictors list
    PREDICTION_ENABLED = False
    st.warning("Model 'predictors' metadata not found in model_package.pkl. Predictions disabled until predictors are available.")

# extra check: ensure model feature count matches predictors
if model is not None and predictors:
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and expected != len(predictors):
        st.warning(f"Model expects {expected} features but 'predictors' length is {len(predictors)}. Predictions may fail.")
        # keep enabled but guard later on

# ---------- HELPERS / PIPELINE ----------

def last_team_stats(team_name, n=1):
    """Return the most recent aggregated rolling stats for a team from team_db.
    If team_db missing, compute simple aggregates from raw_df for recent ROLLING_WINDOW matches.
    """
    if team_db is None:
        if raw_df is None:
            return {}
        # fallback simple aggregates
        df = raw_df[(raw_df['HomeTeam'] == team_name) | (raw_df['AwayTeam'] == team_name)].sort_values('Date')
        if df.empty:
            return {}
        last = df.tail(ROLLING_WINDOW)

        # compute shots/shot targets/goals by combining home and away contexts if present
        shots_home = last.loc[last['HomeTeam'] == team_name, 'HS'] if 'HS' in last.columns else pd.Series(dtype=float)
        shots_away = last.loc[last['AwayTeam'] == team_name, 'AS'] if 'AS' in last.columns else pd.Series(dtype=float)
        shots_series = pd.concat([shots_home, shots_away], ignore_index=True)
        shots_roll = shots_series.mean() if not shots_series.empty else 0.0

        # shots on target
        st_home = last.loc[last['HomeTeam'] == team_name, 'HST'] if 'HST' in last.columns else pd.Series(dtype=float)
        st_away = last.loc[last['AwayTeam'] == team_name, 'AST'] if 'AST' in last.columns else pd.Series(dtype=float)
        st_series = pd.concat([st_home, st_away], ignore_index=True)
        st_roll = st_series.mean() if not st_series.empty else 0.0

        # goals
        gf_home = last.loc[last['HomeTeam'] == team_name, 'FTHG'] if 'FTHG' in last.columns else pd.Series(dtype=float)
        gf_away = last.loc[last['AwayTeam'] == team_name, 'FTAG'] if 'FTAG' in last.columns else pd.Series(dtype=float)
        gf_series = pd.concat([gf_home, gf_away], ignore_index=True)
        goals_roll = gf_series.mean() if not gf_series.empty else 0.0

        return {
            'Shots_Roll': float(shots_roll),
            'ST_Roll': float(st_roll),
            'Goals_Roll': float(goals_roll)
        }

    # If team_db is a DataFrame
    if isinstance(team_db, pd.DataFrame):
        # try to find the Team column name
        team_col = 'Team' if 'Team' in team_db.columns else team_db.columns[0]
        df = team_db[team_db.get(team_col, team_db[team_col]) == team_name] if team_col in team_db.columns else team_db
        df = team_db[team_db[team_col] == team_name] if team_col in team_db.columns else pd.DataFrame()
        if df.empty:
            return {}
        row = df.sort_values('Date').tail(n).iloc[-1]
        return row.to_dict()
    # else unsupported type
    return {}

def get_ref_strictness(ref_name):
    if ref_db is None:
        return 0.0
    try:
        if isinstance(ref_db, pd.DataFrame):
            if 'Strictness_Score' in ref_db.columns:
                if 'Referee' in ref_db.columns:
                    row = ref_db[ref_db['Referee'] == ref_name]
                    if not row.empty:
                        return float(row['Strictness_Score'].iloc[0])
                else:
                    # if ref_db is indexed by Referee
                    try:
                        return float(ref_db.loc[ref_name, 'Strictness_Score'])
                    except Exception:
                        return 0.0
    except Exception:
        return 0.0
    return 0.0

def _safe_avg_from_user(col_name, user_val):
    """If user supplies 0 or None, use historical mean if available, else None"""
    if raw_df is None:
        # default fallback odds
        return None
    if user_val is None:
        return float(raw_df[col_name].mean()) if col_name in raw_df.columns else None
    try:
        # treat explicitly provided 0 or 0.0 as sentinel to use mean
        if isinstance(user_val, (int, float)) and float(user_val) == 0.0:
            return float(raw_df[col_name].mean()) if col_name in raw_df.columns else None
        return float(user_val)
    except Exception:
        return float(raw_df[col_name].mean()) if col_name in raw_df.columns else None

def build_feature_vector(home, away, ref_name=None, avgH=None, avgD=None, avgA=None):
    """Construct predictor vector matching 'predictors' from the pickle.
    Uses available team_db/ref_db/raw_df to populate rolling stats; fills missing with sensible defaults.
    """
    if not PREDICTION_ENABLED:
        raise ValueError("Predictions are disabled because model/predictors/raw data did not load.")

    if not predictors:
        raise ValueError("No predictors available in model package.")

    # codes
    if team_encoder is not None:
        try:
            h_code = int(team_encoder.transform([home])[0])
        except Exception:
            h_code = 0
        try:
            a_code = int(team_encoder.transform([away])[0])
        except Exception:
            a_code = 1
    else:
        # fallback: hash-based stable codes
        h_code = abs(hash(home)) % 1000
        a_code = abs(hash(away)) % 1000

    if ref_encoder is not None and ref_name:
        try:
            r_code = int(ref_encoder.transform([ref_name])[0])
        except Exception:
            r_code = 0
    else:
        r_code = 0

    strict = get_ref_strictness(ref_name) if ref_name else 0.0

    # Odds: treat 0 as sentinel => use historical means
    avgH = _safe_avg_from_user('AvgH', avgH) or 2.0
    avgD = _safe_avg_from_user('AvgD', avgD) or 3.5
    avgA = _safe_avg_from_user('AvgA', avgA) or 4.0

    odds_var = 0.0
    try:
        if 'MaxH' in raw_df.columns and 'AvgH' in raw_df.columns:
            odds_var = float(raw_df['MaxH'].mean() - raw_df['AvgH'].mean())
    except Exception:
        odds_var = 0.0

    # Team rolling stats (try team_db)
    h_stats = last_team_stats(home)
    a_stats = last_team_stats(away)

    # Use predictors list order to assemble vector
    vec = []
    for p in predictors:
        if p == 'Home_Code':
            vec.append(h_code)
        elif p == 'Away_Code':
            vec.append(a_code)
        elif p == 'Ref_Code':
            vec.append(r_code)
        elif p == 'Strictness_Score':
            vec.append(strict)
        elif p == 'AvgH':
            vec.append(avgH)
        elif p == 'AvgD':
            vec.append(avgD)
        elif p == 'AvgA':
            vec.append(avgA)
        elif p == 'Odds_Variance':
            vec.append(odds_var)
        else:
            # team-specific features: prefix H_ and A_
            if p.startswith('H_'):
                key = p.replace('H_', '')
                vec.append(float(h_stats.get(key, 0) or 0))
            elif p.startswith('A_'):
                key = p.replace('A_', '')
                vec.append(float(a_stats.get(key, 0) or 0))
            else:
                # fallback numeric default
                vec.append(0.0)

    X = np.array(vec).reshape(1, -1)
    # final safety: verify shape matches model expectation
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        raise ValueError(f"Feature vector length {X.shape[1]} does not match model expectation {expected}.")
    return X

# ---------- ANALYTICS FUNCTIONS ----------

def predict_match_full(home, away, ref=None, avgH=None, avgD=None, avgA=None):
    if not PREDICTION_ENABLED:
        return {'error': 'Prediction subsystem not available.'}
    try:
        X = build_feature_vector(home, away, ref, avgH, avgD, avgA)
        pred = model.predict(X)[0]
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
        label = None
        try:
            label = res_encoder.inverse_transform([pred])[0] if res_encoder is not None else str(pred)
        except Exception:
            label = str(pred)
        # Map label to human message
        if label == 'H':
            outcome = f"{home} likely wins at home."
        elif label == 'A':
            outcome = f"{away} likely wins away."
        else:
            outcome = "Draw is most likely."
        return {
            'prediction': label,
            'message': outcome,
            'proba': proba
        }
    except Exception as e:
        return {'error': str(e)}

def team_summary(team):
    if raw_df is None:
        return {'error': 'Raw data not loaded.'}
    df = raw_df[(raw_df['HomeTeam'] == team) | (raw_df['AwayTeam'] == team)]
    if df.empty:
        return {'error': 'Team not found in data.'}
    played = len(df)
    wins = len(df[((df['HomeTeam'] == team) & (df['FTR'] == 'H')) | ((df['AwayTeam'] == team) & (df['FTR'] == 'A'))])
    draws = len(df[(df['FTR'] == 'D') & ((df['HomeTeam'] == team) | (df['AwayTeam'] == team))])
    goals_for = df.apply(lambda r: r['FTHG'] if r['HomeTeam'] == team else r['FTAG'], axis=1).sum()
    goals_against = df.apply(lambda r: r['FTAG'] if r['HomeTeam'] == team else r['FTHG'], axis=1).sum()
    return {
        'played': int(played),
        'wins': int(wins),
        'draws': int(draws),
        'goals_for': int(goals_for),
        'goals_against': int(goals_against),
        'win_rate': (wins / played) if played else 0.0
    }

def top_scoring_teams(n=10):
    if raw_df is None:
        return pd.Series([])
    if 'FTHG' not in raw_df.columns or 'FTAG' not in raw_df.columns:
        return pd.Series([])
    home_goals = raw_df.groupby('HomeTeam')['FTHG'].sum()
    away_goals = raw_df.groupby('AwayTeam')['FTAG'].sum()
    total = home_goals.add(away_goals, fill_value=0).sort_values(ascending=False)
    return total.head(n)

def cards_summary(team=None):
    if raw_df is None:
        return {'yellow': 0, 'red': 0}
    if not {'HY', 'AY', 'HR', 'AR'}.intersection(raw_df.columns):
        return {'yellow': 0, 'red': 0}
    df = raw_df
    if team:
        df = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
    yellow = 0
    red = 0
    try:
        yellow = int(df.get('HY', 0).sum() + df.get('AY', 0).sum())
        red = int(df.get('HR', 0).sum() + df.get('AR', 0).sum())
    except Exception:
        yellow, red = 0, 0
    return {'yellow': yellow, 'red': red}

# ---------- PLOTTING ----------

def plot_top_scoring(n=10):
    s = top_scoring_teams(n)
    if s.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha='center')
        return fig
    fig, ax = plt.subplots(figsize=(9, 5))
    s.plot.bar(ax=ax)
    ax.set_title('Top Scoring Teams (Total Goals)')
    ax.set_ylabel('Goals')
    plt.tight_layout()
    return fig

def plot_cards_distribution():
    if raw_df is None or 'HY' not in raw_df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No card data", ha='center')
        return fig
    team_y = raw_df.groupby('HomeTeam')['HY'].sum().add(raw_df.groupby('AwayTeam')['AY'].sum(), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    team_y.sort_values(ascending=False).head(15).plot.bar(ax=ax)
    ax.set_title('Top Teams by Yellow Cards')
    plt.tight_layout()
    return fig

def plot_team_form(team, limit=20):
    """Plot recent form for team: win=3, draw=1, loss=0"""
    if raw_df is None:
        return None, None
    df = raw_df[(raw_df['HomeTeam'] == team) | (raw_df['AwayTeam'] == team)].sort_values('Date', ascending=False).head(limit)
    if df.empty:
        return None, None
    df = df.sort_values('Date')
    def result_points(row):
        if row['FTR'] == 'D':
            return 1
        if (row['HomeTeam'] == team and row['FTR'] == 'H') or (row['AwayTeam'] == team and row['FTR'] == 'A'):
            return 3
        return 0
    points = df.apply(result_points, axis=1).values
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df['Date'], points, marker='o')
    ax.set_title(f'{team} — Recent Form (3=win,1=draw,0=loss)')
    ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([0, 1, 2, 3])
    plt.tight_layout()
    return fig, df

# ---------- STREAMLIT UI ----------

st.sidebar.header('Tools')
mode = st.sidebar.radio('Choose a tool', ['Chat (LLM)', 'Predict match', 'Analyze team', 'Visualize', 'Bulk predict'])

# Chat mode: generic conversation with assistant; user can request function calls via explicit phrases
if mode == 'Chat (LLM)':
    st.subheader('Chat with the Assistant')
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    for msg in st.session_state['chat_history']:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    prompt = st.chat_input('Ask the assistant (you can ask for predictions, charts, or analysis)')
    if prompt:
        st.session_state['chat_history'].append({'role': 'user', 'content': prompt})

        low = prompt.lower()
        out_text = None
        fig = None

        try:
            # Predict intent detection
            if any(k in low for k in ['predict', 'who will win', 'vs', ' v ']):
                # attempt to extract team names with ollama if available
                extracted = None
                if OLLAMA_AVAILABLE:
                    try:
                        extract_prompt = f"Extract home and away team names from: {prompt}. Return exact two names separated by COMMA, nothing else."
                        ext = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': extract_prompt}])
                        extracted = ext.get('message', {}).get('content', '')
                    except Exception:
                        extracted = None

                if extracted and ',' in extracted:
                    h, a = [t.strip() for t in extracted.split(',')[:2]]
                else:
                    # fallback simple parsing: look for "team1 vs team2" variants
                    if ' vs ' in prompt.lower():
                        parts = prompt.lower().split(' vs ')
                        h, a = parts[0].strip(), parts[-1].strip()
                    elif ' v ' in prompt.lower():
                        parts = prompt.lower().split(' v ')
                        h, a = parts[0].strip(), parts[-1].strip()
                    else:
                        # as last resort, split on ' vs. ' or comma
                        parts = prompt.replace(' vs ', ',').replace(' v ', ',').split(',')
                        if len(parts) >= 2:
                            h, a = parts[0].strip(), parts[1].strip()
                        else:
                            h, a = None, None

                if not h or not a:
                    out_text = "Couldn't reliably extract two team names. Try 'Arsenal vs Chelsea' or use the 'Predict match' tool."
                else:
                    # user might have typed odds - respect sentinel 0 meaning use mean
                    res = predict_match_full(h.title(), a.title())
                    if 'error' in res:
                        out_text = 'Prediction failed: ' + res['error']
                    else:
                        probs = res.get('proba')
                        if probs is not None:
                            out_text = f"{res['message']} Probabilities (H/D/A): {np.round(probs,3).tolist()}"
                        else:
                            out_text = res['message']

            elif any(k in low for k in ['plot', 'chart', 'visualize', 'graph']):
                if 'top' in low and 'score' in low:
                    fig = plot_top_scoring(10)
                    out_text = "Top scoring teams (total goals)."
                elif 'cards' in low:
                    fig = plot_cards_distribution()
                    out_text = "Card distribution (top teams)."
                else:
                    out_text = "Tell me what to plot: e.g. 'plot top scoring teams' or 'plot cards for Manchester United'"

            elif any(k in low for k in ['analyze', 'stats', 'record']):
                # extract team name
                team = None
                if OLLAMA_AVAILABLE:
                    try:
                        extract_prompt = f"Extract the team name from: {prompt}. Return name only."
                        resp = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': extract_prompt}])
                        team = resp.get('message', {}).get('content', '').strip()
                    except Exception:
                        team = None

                if not team:
                    # fallback: last word heuristic
                    tokens = prompt.split()
                    team = tokens[-1].strip().title()

                summary = team_summary(team)
                if 'error' in summary:
                    out_text = summary['error']
                else:
                    out_text = (f"{team}: Played {summary['played']}, Wins {summary['wins']}, Draws {summary['draws']}, "
                                f"Goals For {summary['goals_for']}, Goals Against {summary['goals_against']}")

            else:
                # default: forward to LLM for conversational reply
                if OLLAMA_AVAILABLE:
                    sys_prompt = "You are a Premier League Assistant Coach. Be concise, helpful and when the user asks for an analysis, call the appropriate internal tool."
                    try:
                        resp = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': prompt}])
                        out_text = resp.get('message', {}).get('content', '')
                    except Exception as e:
                        out_text = f"LLM error: {e}"
                else:
                    out_text = "LLM is not available (ollama not installed or running). Use the built-in tools (Predict match, Analyze team, Visualize)."

        except Exception as e:
            out_text = f"Error in assistant pipeline: {e}\n\n{traceback.format_exc()}"

        # render
        with st.chat_message('assistant'):
            st.markdown(out_text or "No output produced.")
            if fig:
                st.pyplot(fig)

        st.session_state['chat_history'].append({'role': 'assistant', 'content': out_text or ''})

# Predict match UI
elif mode == 'Predict match':
    st.subheader('Predict a single match using the ML model')
    if raw_df is None:
        st.error("Raw data not loaded. Upload or place the CSV at one of the configured paths.")
    else:
        teams = sorted(list(set(raw_df['HomeTeam'].unique()).union(set(raw_df['AwayTeam'].unique()))))
        col1, col2 = st.columns([1, 1])
        with col1:
            home = st.selectbox('Home team', teams)
        with col2:
            away = st.selectbox('Away team', teams, index=1)

        ref_name = st.text_input('Referee (optional)')
        avgH_input = st.number_input('AvgH (bookie) — leave 0 to use historical mean', value=0.0, format="%.2f")
        avgD_input = st.number_input('AvgD (bookie) — leave 0 to use historical mean', value=0.0, format="%.2f")
        avgA_input = st.number_input('AvgA (bookie) — leave 0 to use historical mean', value=0.0, format="%.2f")

        # treat 0.0 as sentinel: pass None to builder so it uses historical mean
        avgH = None if float(avgH_input) == 0.0 else float(avgH_input)
        avgD = None if float(avgD_input) == 0.0 else float(avgD_input)
        avgA = None if float(avgA_input) == 0.0 else float(avgA_input)

        if st.button('Run prediction'):
            if not PREDICTION_ENABLED:
                st.error("Prediction features are not available. Check logs/warnings in the app.")
            else:
                out = predict_match_full(home, away, ref_name or None, avgH, avgD, avgA)
                if 'error' in out:
                    st.error(out['error'])
                else:
                    st.success(out['message'])
                    if out.get('proba') is not None:
                        st.write('Probabilities (H, D, A):', np.round(out['proba'], 3).tolist())

# Analyze team UI
elif mode == 'Analyze team':
    st.subheader('Deep-dive team analysis (uses full CSV)')
    if raw_df is None:
        st.error("Raw data not loaded.")
    else:
        teams = sorted(list(set(raw_df['HomeTeam'].unique()).union(set(raw_df['AwayTeam'].unique()))))
        team = st.selectbox('Choose a team', teams)
        if st.button('Run analysis'):
            summary = team_summary(team)
            if 'error' in summary:
                st.error(summary['error'])
            else:
                st.metric('Games Played', summary['played'])
                st.metric('Wins', summary['wins'])
                st.metric('Win Rate', f"{summary['win_rate']*100:.1f}%")
                st.write('Goals For:', summary['goals_for'], 'Goals Against:', summary['goals_against'])
                fig, _ = plot_team_form(team, limit=20)
                if fig:
                    st.pyplot(fig)

# Visualization UI
elif mode == 'Visualize':
    st.subheader('Pre-built visualizations')
    if raw_df is None:
        st.error("Raw data not loaded.")
    else:
        viz = st.selectbox('Choose chart', ['Top scoring teams', 'Cards distribution', 'Head-to-head (team vs team)'])
        if viz == 'Top scoring teams':
            fig = plot_top_scoring(15)
            st.pyplot(fig)
        elif viz == 'Cards distribution':
            fig = plot_cards_distribution()
            st.pyplot(fig)
        elif viz == 'Head-to-head (team vs team)':
            teams = sorted(list(set(raw_df['HomeTeam'].unique()).union(set(raw_df['AwayTeam'].unique()))))
            t1 = st.selectbox('Team 1', teams)
            t2 = st.selectbox('Team 2', teams, index=1)
            if st.button('Show H2H'):
                h2h = raw_df[((raw_df['HomeTeam'] == t1) & (raw_df['AwayTeam'] == t2)) | ((raw_df['HomeTeam'] == t2) & (raw_df['AwayTeam'] == t1))]
                cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                st.write(h2h[cols].sort_values('Date', ascending=False).head(50))

# Bulk predict: upload CSV of upcoming matches
elif mode == 'Bulk predict':
    st.subheader('Batch predict from a CSV of upcoming fixtures')
    st.markdown('CSV must contain columns: HomeTeam, AwayTeam (optional: Referee, AvgH, AvgD, AvgA)')
    uploaded = st.file_uploader('Upload fixtures CSV', type=['csv'])
    if uploaded is not None:
        try:
            fixtures = pd.read_csv(uploaded)
            req = ['HomeTeam', 'AwayTeam']
            if not all(c in fixtures.columns for c in req):
                st.error('CSV missing required columns HomeTeam and AwayTeam')
            else:
                results = []
                for _, r in fixtures.iterrows():
                    out = predict_match_full(r['HomeTeam'], r['AwayTeam'], r.get('Referee'), r.get('AvgH'), r.get('AvgD'), r.get('AvgA'))
                    results.append({
                        'HomeTeam': r['HomeTeam'],
                        'AwayTeam': r['AwayTeam'],
                        'Prediction': out.get('prediction'),
                        'Message': out.get('message'),
                        'Proba': out.get('proba')
                    })
                res_df = pd.DataFrame(results)
                st.dataframe(res_df)
                csv_bytes = res_df.to_csv(index=False).encode('utf-8')
                st.download_button('Download results CSV', csv_bytes, file_name='predictions.csv')
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

# Utility: expose some debug info
with st.expander('Debug / Data sample'):
    st.write('Model predictors:', predictors)
    st.write('Model loaded:', model is not None)
    st.write('Prediction enabled:', PREDICTION_ENABLED)
    if raw_df is not None:
        st.write('Sample raw data:')
        st.dataframe(raw_df.head(5))
    else:
        st.write('Raw data not loaded.')
