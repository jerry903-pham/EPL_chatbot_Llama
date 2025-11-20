import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import ollama
from datetime import datetime

# ---------- CONFIG ----------
st.set_page_config(page_title="EPL Assistant Coach — MVP", layout="wide")
st.title("Premier League Assistant Coach — Enhanced MVP")

MODEL_PKL = "model_package.pkl"
CSV_PATHS = ["epl_matches.csv", "/mnt/data/epl_matches.csv"]
OLLAMA_MODEL = "phi3:mini"  # change if you use a different local model

# ---------- UTIL: Loading resources ----------
@st.cache_resource
def load_resources():
    # Load CSV (try multiple locations)
    raw = None
    for p in CSV_PATHS:
        try:
            raw = pd.read_csv(p)
            break
        except Exception:
            raw = None
    if raw is None:
        raise FileNotFoundError(f"Could not find CSV at {CSV_PATHS}")

    # Ensure Date typed
    if 'Date' in raw.columns:
        try:
            raw['Date'] = pd.to_datetime(raw['Date'], dayfirst=True, errors='coerce')
        except Exception:
            raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')

    # Load ML package
    with open(MODEL_PKL, 'rb') as f:
        pkg = pickle.load(f)

    # Normalize keys across different saves
    model = pkg.get('model') or pkg.get('clf') or pkg
    encoders = pkg.get('encoders') or pkg.get('enc') or {}
    team_encoder = encoders.get('team') if isinstance(encoders, dict) else None
    ref_encoder = encoders.get('ref') if isinstance(encoders, dict) else None
    res_encoder = encoders.get('res') if isinstance(encoders, dict) else None

    # older naming: team_encoder, result_encoder
    if team_encoder is None and 'team_encoder' in pkg:
        team_encoder = pkg['team_encoder']
    if res_encoder is None and 'result_encoder' in pkg:
        res_encoder = pkg['result_encoder']

    ref_db = pkg.get('ref_db') if 'ref_db' in pkg else pkg.get('ref_stats')
    team_db = pkg.get('team_db') if 'team_db' in pkg else pkg.get('advanced_stats')
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
        'predictors': predictors
    }

resources = load_resources()
raw_df = resources['raw']
model = resources['model']
team_encoder = resources['team_encoder']
ref_encoder = resources['ref_encoder']
res_encoder = resources['res_encoder']
ref_db = resources['ref_db']
team_db = resources['team_db']
predictors = resources['predictors']

# Basic checks
if team_encoder is None:
    st.warning("Team encoder not found in model package. Some features will be limited.")

# ---------- HELPERS / PIPELINE ----------

def last_team_stats(team_name, n=1):
    """Return the most recent aggregated rolling stats for a team from team_db.
    team_db is expected to have columns like 'Team','Date','Shots_Roll', etc.
    If team_db is missing, fall back to computing simple aggregates from raw_df.
    """
    if team_db is None:
        # fallback simple aggregates
        df = raw_df[(raw_df['HomeTeam'] == team_name) | (raw_df['AwayTeam'] == team_name)].sort_values('Date')
        if df.empty:
            return {}
        last = df.tail(ROLLING_WINDOW:=3)
        return {
            'Shots_Roll': last['HS'].append(last['AS']).mean() if 'HS' in last.columns else 0,
        }

    # team_db may be a dataframe or series
    if isinstance(team_db, pd.DataFrame):
        df = team_db[team_db.get('Team', team_db.columns[0]) == team_name].sort_values('Date')
        if df.empty:
            return {}
        row = df.tail(n).iloc[-1]
        return row.to_dict()
    else:
        return {}


def get_ref_strictness(ref_name):
    if ref_db is None:
        return 0
    if isinstance(ref_db, pd.DataFrame):
        if 'Strictness_Score' in ref_db.columns:
            # ref_db may be indexed by Referee
            if 'Referee' in ref_db.columns:
                row = ref_db[ref_db['Referee'] == ref_name]
                if not row.empty:
                    return float(row['Strictness_Score'].iloc[0])
            else:
                # if ref_db is index
                try:
                    return float(ref_db.loc[ref_name, 'Strictness_Score'])
                except Exception:
                    return 0
    # default
    return 0


def build_feature_vector(home, away, ref_name=None, avgH=None, avgD=None, avgA=None):
    """Construct predictor vector matching 'predictors' from the pickle.
    Uses available team_db/ref_db/raw_df to populate rolling stats; fills missing with sensible defaults.
    """
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

    strict = get_ref_strictness(ref_name) if ref_name else 0

    # Odds
    avgH = float(avgH) if avgH is not None else float(raw_df['AvgH'].mean()) if 'AvgH' in raw_df.columns else 2.0
    avgD = float(avgD) if avgD is not None else float(raw_df['AvgD'].mean()) if 'AvgD' in raw_df.columns else 3.5
    avgA = float(avgA) if avgA is not None else float(raw_df['AvgA'].mean()) if 'AvgA' in raw_df.columns else 4.0
    odds_var = 0.0
    if 'MaxH' in raw_df.columns:
        try:
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
                # fallback 0
                vec.append(0.0)

    return np.array(vec).reshape(1, -1)


# ---------- ANALYTICS FUNCTIONS ----------

def predict_match_full(home, away, ref=None, avgH=None, avgD=None, avgA=None):
    try:
        X = build_feature_vector(home, away, ref, avgH, avgD, avgA)
        pred = model.predict(X)[0]
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
        label = res_encoder.inverse_transform([pred])[0] if res_encoder is not None else str(pred)
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
    df = raw_df[(raw_df['HomeTeam'] == team) | (raw_df['AwayTeam'] == team)]
    if df.empty:
        return {'error': 'Team not found in data.'}
    played = len(df)
    wins = len(df[((df['HomeTeam'] == team) & (df['FTR'] == 'H')) | ((df['AwayTeam'] == team) & (df['FTR'] == 'A'))])
    draws = len(df[df['FTR'] == 'D'])
    goals_for = df.apply(lambda r: r['FTHG'] if r['HomeTeam'] == team else r['FTAG'], axis=1).sum()
    goals_against = df.apply(lambda r: r['FTAG'] if r['HomeTeam'] == team else r['FTHG'], axis=1).sum()
    return {
        'played': played,
        'wins': wins,
        'draws': draws,
        'goals_for': int(goals_for),
        'goals_against': int(goals_against),
        'win_rate': wins / played if played else 0
    }


def top_scoring_teams(n=10):
    if 'FTHG' not in raw_df.columns or 'FTAG' not in raw_df.columns:
        return []
    # Sum goals by team across home and away
    home_goals = raw_df.groupby('HomeTeam')['FTHG'].sum()
    away_goals = raw_df.groupby('AwayTeam')['FTAG'].sum()
    total = home_goals.add(away_goals, fill_value=0).sort_values(ascending=False)
    return total.head(n)


def cards_summary(team=None):
    # returns aggregated card counts
    if 'HY' not in raw_df.columns:
        return {}
    df = raw_df
    if team:
        df = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
    yellow = df['HY'].sum() + df['AY'].sum()
    red = df['HR'].sum() + df['AR'].sum()
    return {'yellow': int(yellow), 'red': int(red)}


# ---------- PLOTTING ----------

def plot_top_scoring(n=10):
    s = top_scoring_teams(n)
    fig, ax = plt.subplots(figsize=(9, 5))
    s.plot.bar(ax=ax)
    ax.set_title('Top Scoring Teams (Total Goals)')
    ax.set_ylabel('Goals')
    plt.tight_layout()
    return fig


def plot_cards_distribution():
    df = raw_df
    if 'HY' not in df.columns:
        return None
    team_y = df.groupby('HomeTeam')['HY'].sum().add(df.groupby('AwayTeam')['AY'].sum(), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    team_y.sort_values(ascending=False).head(15).plot.bar(ax=ax)
    ax.set_title('Top Teams by Yellow Cards')
    plt.tight_layout()
    return fig


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

        # simple router via keywords to call functions
        low = prompt.lower()
        out_text = None
        fig = None

        try:
            if any(k in low for k in ['predict', 'who will win', 'vs', 'v']):
                # attempt to extract team names with ollama
                extract_prompt = f"Extract home and away team names from: {prompt}. Return exact two names separated by COMMA, nothing else."
                ext = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': extract_prompt}])['message']['content']
                if ',' in ext:
                    h, a = [t.strip() for t in ext.split(',')[:2]]
                else:
                    # fallback simple split
                    parts = prompt.replace(' vs ', ',').replace(' v ', ',').split(',')
                    h, a = parts[0].strip(), parts[-1].strip()
                res = predict_match_full(h, a)
                if 'error' in res:
                    out_text = 'Prediction failed: ' + res['error']
                else:
                    probs = res.get('proba')
                    if probs is not None:
                        out_text = f"{res['message']} Probabilities (H/D/A): {probs.round(3).tolist()}"
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
                    out_text = "Tell me what to plot: e.g. 'plot top scoring teams' or 'plot cards for manchester united'"

            elif any(k in low for k in ['analyze', 'stats', 'record']):
                extract_prompt = f"Extract the team name from: {prompt}. Return name only."
                team = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': extract_prompt}])['message']['content'].strip()
                summary = team_summary(team)
                if 'error' in summary:
                    out_text = summary['error']
                else:
                    out_text = (f"{team}: Played {summary['played']}, Wins {summary['wins']}, Draws {summary['draws']}, "
                                f"Goals For {summary['goals_for']}, Goals Against {summary['goals_against']}")

            else:
                # default: forward to LLM for conversational reply
                sys_prompt = "You are a Premier League Assistant Coach. Be concise, helpful and when the user asks for an analysis, call the appropriate internal tool."
                resp = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': prompt}])
                out_text = resp['message']['content']
        except Exception as e:
            out_text = f"Error in assistant pipeline: {e}"

        # render
        with st.chat_message('assistant'):
            st.markdown(out_text or "No output produced.")
            if fig:
                st.pyplot(fig)

        st.session_state['chat_history'].append({'role': 'assistant', 'content': out_text})


# Predict match UI
elif mode == 'Predict match':
    st.subheader('Predict a single match using the ML model')
    teams = sorted(list(set(raw_df['HomeTeam'].unique()).union(set(raw_df['AwayTeam'].unique()))))
    col1, col2 = st.columns([1, 1])
    with col1:
        home = st.selectbox('Home team', teams)
    with col2:
        away = st.selectbox('Away team', teams, index=1)

    ref_name = st.text_input('Referee (optional)')
    avgH = st.number_input('AvgH (bookie) — leave 0 to use historical mean', value=0.0)
    avgD = st.number_input('AvgD (bookie) — leave 0 to use historical mean', value=0.0)
    avgA = st.number_input('AvgA (bookie) — leave 0 to use historical mean', value=0.0)

    if st.button('Run prediction'):
        out = predict_match_full(home, away, ref_name or None, avgH or None, avgD or None, avgA or None)
        if 'error' in out:
            st.error(out['error'])
        else:
            st.success(out['message'])
            if out.get('proba') is not None:
                st.write('Probabilities (H, D, A):', out['proba'].round(3).tolist())


# Analyze team UI
elif mode == 'Analyze team':
    st.subheader('Deep-dive team analysis (uses full CSV)')
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
            # quick plots
            fig, _ = plot_team_form(team, limit=20) if 'plot_team_form' in globals() else (None, None)
            if fig:
                st.pyplot(fig)


# Visualization UI
elif mode == 'Visualize':
    st.subheader('Pre-built visualizations')
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
            st.write(h2h[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].sort_values('Date', ascending=False).head(50))


# Bulk predict: upload CSV of upcoming matches
elif mode == 'Bulk predict':
    st.subheader('Batch predict from a CSV of upcoming fixtures')
    st.markdown('CSV must contain columns: HomeTeam, AwayTeam (optional: Referee, AvgH, AvgD, AvgA)')
    uploaded = st.file_uploader('Upload fixtures CSV', type=['csv'])
    if uploaded is not None:
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
            st.download_button('Download results CSV', res_df.to_csv(index=False), file_name='predictions.csv')


# Utility: expose some debug info
with st.expander('Debug / Data sample'):
    st.write('Model predictors:', predictors)
    st.write('Sample raw data:')
    st.dataframe(raw_df.head(5))

# End of file
