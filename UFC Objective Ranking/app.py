"""
DISCLAIMER :
    -I did not write any of the code for this web app, all the code in this
    app was made using AI and I only directed it.
    -This was done to speed up the front end part and also due to me not having expertise
    with this library

UFC Rankings Dashboard
======================
A Streamlit web app for exploring UFC fighter rankings and stats.

Instructions:
  pip install streamlit plotly pandas
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(
    page_title="UFC Objective Rankings",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@700&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0a0a0a;
    color: #f0f0f0;
}

.stApp {
    background-color: #0a0a0a;
}

h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
}

.big-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 72px;
    line-height: 1;
    letter-spacing: 4px;
    background: linear-gradient(135deg, #ffffff 0%, #e63946 60%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0px;
}

.subtitle {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 16px;
    letter-spacing: 6px;
    color: #888;
    text-transform: uppercase;
    margin-bottom: 40px;
}

.stat-card {
    background: linear-gradient(135deg, #1a1a1a 0%, #111 100%);
    border: 1px solid #2a2a2a;
    border-left: 3px solid #e63946;
    border-radius: 4px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

.stat-card-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 48px;
    color: #e63946;
    line-height: 1;
}

.stat-card-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 13px;
    letter-spacing: 3px;
    color: #888;
    text-transform: uppercase;
    margin-top: 4px;
}

.rank-badge {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 28px;
    color: #e63946;
}

.fighter-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 1px;
}

.divider {
    border: none;
    border-top: 1px solid #2a2a2a;
    margin: 32px 0;
}

section[data-testid="stSidebar"] {
    background-color: #111;
    border-right: 1px solid #2a2a2a;
    min-width: 300px !important;
    width: 300px !important;
}

section[data-testid="stSidebar"] .stRadio label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 20px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #ccc !important;
    padding: 8px 0 !important;
}

section[data-testid="stSidebar"] p {
    font-size: 15px !important;
    color: #aaa !important;
}

.stSelectbox > div > div {
    background-color: #1a1a1a;
    border-color: #333;
    color: #f0f0f0;
}

.stDataFrame {
    border: 1px solid #2a2a2a;
}

.stDataFrame thead tr th {
    background-color: #1a1a1a !important;
    color: #e63946 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #333 !important;
}

.stDataFrame tbody tr {
    background-color: #111 !important;
    color: #f0f0f0 !important;
    border-bottom: 1px solid #1e1e1e !important;
}

.stDataFrame tbody tr:hover {
    background-color: #1a1a1a !important;
}

.stDataFrame tbody tr td {
    color: #f0f0f0 !important;
    font-family: 'Barlow', sans-serif !important;
}

div[data-testid="metric-container"] {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #e63946;
    padding: 16px;
    border-radius: 4px;
}

div[data-testid="metric-container"] label {
    font-family: 'Barlow Condensed', sans-serif;
    letter-spacing: 2px;
    font-size: 12px;
    color: #888 !important;
    text-transform: uppercase;
}

div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 36px;
    color: #f0f0f0;
}
</style>
""", unsafe_allow_html=True)



# ========================================================
# HELPER: Render a styled dark HTML table
# We use this instead of st.dataframe so we can fully
# control the colours and font sizes to match the theme.
# ========================================================

def dark_table(df: pd.DataFrame):
    """Render a dataframe as a styled dark HTML table."""
    headers = "".join(f'<th>{col}</th>' for col in df.columns)
    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for val in row:
            if str(val) == "WIN":
                cells += f'<td style="color:#4caf50;font-weight:700;">{val}</td>'
            elif str(val) == "LOSS":
                cells += f'<td style="color:#e63946;font-weight:700;">{val}</td>'
            else:
                cells += f"<td>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"

    html = f"""
    <div style="overflow-x:auto; margin-bottom: 24px;">
    <table style="
        width:100%;
        border-collapse:collapse;
        font-family:'Barlow',sans-serif;
        font-size:15px;
        background:#111;
        color:#f0f0f0;
    ">
        <thead>
            <tr style="border-bottom:2px solid #e63946;">
                {headers}
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    </div>
    <style>
        table th {{
            font-family:'Barlow Condensed',sans-serif;
            font-size:13px;
            letter-spacing:3px;
            text-transform:uppercase;
            color:#e63946;
            padding:12px 16px;
            text-align:left;
            background:#0a0a0a;
        }}
        table td {{
            padding:12px 16px;
            border-bottom:1px solid #1e1e1e;
            font-size:15px;
        }}
        table tbody tr:hover td {{
            background:#1a1a1a;
        }}
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)

# ========================================================
# LOAD DATA
# ========================================================

@st.cache_data
def load_data():
    fights    = pd.read_csv("fights_clean.csv", parse_dates=["date"])
    rankings  = pd.read_csv("rankings.csv")
    return fights, rankings

@st.cache_data
def build_fighter_table(_fights):
    rows = []
    for _, fight in _fights.iterrows():
        f1_won = fight["result"] == "win"
        base = {
            "date":          fight["date"],
            "weight_class":  fight["weight_class"],
            "method":        fight["method"],
            "method_detail": fight["method_detail"],
            "finish":        fight["finish"],
            "round":         fight["round"],
        }
        rows.append({**base,
            "fighter": fight["fighter_1"], "opponent": fight["fighter_2"],
            "won": 1 if f1_won else 0,
            "td": fight["td_f1"], "sig_str": fight["sig_str_f1_landed"],
            "kd": fight["kd_f1"], "sub_att": fight["sub_att_f1"],
        })
        rows.append({**base,
            "fighter": fight["fighter_2"], "opponent": fight["fighter_1"],
            "won": 0 if f1_won else 1,
            "td": fight["td_f2"], "sig_str": fight["sig_str_f2_landed"],
            "kd": fight["kd_f2"], "sub_att": fight["sub_att_f2"],
        })
    return pd.DataFrame(rows)

def time_to_seconds(t):
    try:
        parts = str(t).strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return np.nan

fights, rankings = load_data()
ff = build_fighter_table(fights)

fights["time_seconds"] = fights["time"].apply(time_to_seconds)
fights["total_seconds"] = fights.apply(
    lambda r: (int(r["round"]) - 1) * 300 + r["time_seconds"]
    if pd.notna(r["round"]) and pd.notna(r["time_seconds"]) else np.nan, axis=1
)


# ========================================================
# SIDEBAR NAVIGATION
# ========================================================

with st.sidebar:
    st.markdown('<div style="font-family: Bebas Neue; font-size: 28px; letter-spacing: 3px; color: #e63946; margin-bottom: 24px;">UFC RANKINGS</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["Rankings", "Fighter Profile", "Interesting Stats", "Trends"],
        label_visibility="collapsed"
    )
    st.markdown('<hr style="border-color: #2a2a2a; margin: 24px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 11px; color: #555; letter-spacing: 1px;">DATA SOURCE: UFCSTATS.COM<br>RANKING METHOD: ELO SYSTEM<br>FIGHTS TRACKED: ' + f"{len(fights):,}" + '</div>', unsafe_allow_html=True)


# ========================================================
# PAGE 1: RANKINGS
# ========================================================

if page == "Rankings":
    st.markdown('<div class="big-title">FIGHTER RANKINGS</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Objective Elo-Based Rankings — Updated Through Latest Event</div>', unsafe_allow_html=True)

    weight_classes = sorted(rankings["weight_class"].unique())
    selected_wc = st.selectbox("Select Weight Class", weight_classes)

    wc_rankings = rankings[rankings["weight_class"] == selected_wc].sort_values("rank").head(15)

    col1, col2, col3 = st.columns(3)
    top = wc_rankings.iloc[0]
    col1.metric("Champion (by Elo)", top["fighter"])
    col2.metric("Top Elo Rating", f"{top['elo']:.0f}")
    col3.metric("Fighters Ranked", len(wc_rankings))

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    display_df = wc_rankings[["rank", "fighter", "record", "win_rate", "finish_rate", "elo", "last_fight"]].copy()
    display_df.columns = ["Rank", "Fighter", "Record", "Win %", "Finish %", "Elo Rating", "Last Fight"]
    display_df["Win %"]     = display_df["Win %"].apply(lambda x: f"{x}%")
    display_df["Finish %"]  = display_df["Finish %"].apply(lambda x: f"{x}%")
    display_df["Elo Rating"] = display_df["Elo Rating"].apply(lambda x: f"{x:.0f}")

    dark_table(display_df)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    fig = px.bar(
        wc_rankings,
        x="fighter", y="elo",
        color="elo",
        color_continuous_scale=["#1a1a1a", "#e63946"],
        labels={"fighter": "", "elo": "Elo Rating"},
        title=f"{selected_wc} — Elo Ratings"
    )
    fig.update_layout(
        plot_bgcolor="#111",
        paper_bgcolor="#111",
        font_color="#f0f0f0",
        font_family="Barlow",
        title_font_family="Bebas Neue",
        title_font_size=24,
        coloraxis_showscale=False,
        xaxis_tickangle=-30,
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)


# ========================================================
# PAGE 2: FIGHTER PROFILE
# ========================================================

elif page == "Fighter Profile":
    st.markdown('<div class="big-title">FIGHTER PROFILE</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Search Any UFC Fighter</div>', unsafe_allow_html=True)

    all_fighters = sorted(ff["fighter"].unique())
    selected_fighter = st.selectbox("Search Fighter", all_fighters)

    fighter_data = ff[ff["fighter"] == selected_fighter].sort_values("date")

    if fighter_data.empty:
        st.warning("No data found for this fighter.")
    else:
        total_fights  = len(fighter_data)
        total_wins    = int(fighter_data["won"].sum())
        total_losses  = total_fights - total_wins
        finish_wins   = int(fighter_data[fighter_data["won"] == 1]["finish"].sum())
        finish_rate   = round(finish_wins / total_wins * 100, 1) if total_wins > 0 else 0
        total_kd      = int(fighter_data["kd"].sum())
        total_td      = int(fighter_data["td"].sum())
        total_str     = int(fighter_data["sig_str"].sum())

        rank_row = rankings[rankings["fighter"] == selected_fighter]

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("UFC Record", f"{total_wins}-{total_losses}-0")
        c2.metric("Finish Rate", f"{finish_rate}%")
        c3.metric("Knockdowns", total_kd)
        c4.metric("Takedowns", total_td)
        c5.metric("Sig. Strikes", f"{total_str:,}")

        if not rank_row.empty:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div style="font-family: Bebas Neue; font-size: 24px; letter-spacing: 2px; color: #888; margin-bottom: 16px;">ELO RANKINGS BY WEIGHT CLASS</div>', unsafe_allow_html=True)
            rank_display = rank_row[["weight_class", "elo", "rank", "record"]].copy()
            rank_display.columns = ["Weight Class", "Elo", "Rank", "Record"]
            rank_display["Elo"] = rank_display["Elo"].apply(lambda x: f"{x:.0f}")
            dark_table(rank_display)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div style="font-family: Bebas Neue; font-size: 24px; letter-spacing: 2px; color: #888; margin-bottom: 16px;">FIGHT LOG</div>', unsafe_allow_html=True)

        log = fighter_data[["date", "opponent", "won", "weight_class", "method", "method_detail", "round"]].copy()
        log["result"] = log["won"].apply(lambda x: "WIN" if x == 1 else "LOSS")
        log["finish"] = log["method"].apply(lambda m: "YES" if str(m).upper() in ["KO/TKO", "SUB"] else "NO")
        log = log.drop(columns=["won"]).rename(columns={
            "date": "Date", "opponent": "Opponent", "result": "Result",
            "weight_class": "Division", "method": "Method",
            "method_detail": "Detail", "round": "Round", "finish": "Finish"
        })
        log["Date"] = log["Date"].dt.date
        log = log.sort_values("Date", ascending=False)
        dark_table(log)


# ========================================================
# PAGE 3: INTERESTING STATS
# ========================================================

elif page == "Interesting Stats":
    st.markdown('<div class="big-title">STATS AND RECORDS</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">UFC Historical Leaderboards</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Most Finishes", "Finish Rate", "Takedowns", "KO Methods", "Sub Methods"
    ])

    win_rows = ff[ff["won"] == 1].copy()

    with tab1:
        finish_wins_df = win_rows[win_rows["finish"] == True]
        most_finishes = (
            finish_wins_df.groupby("fighter").size().reset_index(name="finishes")
            .sort_values("finishes", ascending=False).head(20)
        )
        fig = px.bar(most_finishes, x="finishes", y="fighter", orientation="h",
            color="finishes", color_continuous_scale=["#2a2a2a", "#e63946"],
            title="Most Career Finishes", labels={"finishes": "Total Finishes", "fighter": ""})
        fig.update_layout(plot_bgcolor="#111", paper_bgcolor="#111", font_color="#f0f0f0",
            font_family="Barlow", title_font_family="Bebas Neue", title_font_size=24,
            yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        finish_rate_df = (
            win_rows.groupby("fighter")
            .agg(total_wins=("won", "count"), finishes=("finish", "sum")).reset_index()
        )
        finish_rate_df = finish_rate_df[finish_rate_df["total_wins"] >= 10]
        finish_rate_df["finish_rate"] = (finish_rate_df["finishes"] / finish_rate_df["total_wins"] * 100).round(1)
        finish_rate_df = finish_rate_df.sort_values("finish_rate", ascending=False).head(20)
        fig = px.bar(finish_rate_df, x="finish_rate", y="fighter", orientation="h",
            color="finish_rate", color_continuous_scale=["#2a2a2a", "#e63946"],
            title="Highest Finish Rate (Min 10 UFC Wins)", labels={"finish_rate": "Finish Rate %", "fighter": ""})
        fig.update_layout(plot_bgcolor="#111", paper_bgcolor="#111", font_color="#f0f0f0",
            font_family="Barlow", title_font_family="Bebas Neue", title_font_size=24,
            yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        most_td = (
            ff.groupby("fighter")["td"].sum().reset_index(name="total_td")
            .sort_values("total_td", ascending=False).head(20)
        )
        most_td["total_td"] = most_td["total_td"].astype(int)
        fig = px.bar(most_td, x="total_td", y="fighter", orientation="h",
            color="total_td", color_continuous_scale=["#2a2a2a", "#e63946"],
            title="Most Career Takedowns Landed", labels={"total_td": "Takedowns", "fighter": ""})
        fig.update_layout(plot_bgcolor="#111", paper_bgcolor="#111", font_color="#f0f0f0",
            font_family="Barlow", title_font_family="Bebas Neue", title_font_size=24,
            yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        ko_fights = fights[fights["method"].str.upper().str.contains("KO", na=False)]
        ko_methods = (ko_fights["method_detail"].dropna().str.strip().value_counts().head(10).reset_index())
        ko_methods.columns = ["method", "count"]
        fig = px.bar(ko_methods, x="count", y="method", orientation="h",
            color="count", color_continuous_scale=["#2a2a2a", "#e63946"],
            title="Most Common KO Methods", labels={"count": "Occurrences", "method": ""})
        fig.update_layout(plot_bgcolor="#111", paper_bgcolor="#111", font_color="#f0f0f0",
            font_family="Barlow", title_font_family="Bebas Neue", title_font_size=24,
            yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        sub_fights = fights[fights["method"].str.upper().str.contains("SUB", na=False)]
        sub_methods = (sub_fights["method_detail"].dropna().str.strip().value_counts().head(10).reset_index())
        sub_methods.columns = ["method", "count"]
        fig = px.bar(sub_methods, x="count", y="method", orientation="h",
            color="count", color_continuous_scale=["#2a2a2a", "#e63946"],
            title="Most Common Submission Methods", labels={"count": "Occurrences", "method": ""})
        fig.update_layout(plot_bgcolor="#111", paper_bgcolor="#111", font_color="#f0f0f0",
            font_family="Barlow", title_font_family="Bebas Neue", title_font_size=24,
            yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ========================================================
# PAGE 4: TRENDS
# ========================================================

elif page == "Trends":
    st.markdown('<div class="big-title">TRENDS OVER TIME</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">How the UFC Has Changed Since 1994</div>', unsafe_allow_html=True)

    fights["year"] = fights["date"].dt.year

    yearly = (
        fights.groupby("year")
        .agg(total_fights=("finish", "count"), finishes=("finish", "sum"))
        .reset_index()
    )
    yearly["finish_rate"] = (yearly["finishes"] / yearly["total_fights"] * 100).round(1)
    yearly = yearly[yearly["year"] >= 2000]

    st.markdown('<div style="font-family: Bebas Neue; font-size: 24px; letter-spacing: 2px; color: #888; margin-bottom: 16px;">FINISH RATE BY YEAR</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["finish_rate"],
        mode="lines+markers",
        line=dict(color="#e63946", width=3),
        marker=dict(color="#e63946", size=8),
        fill="tozeroy",
        fillcolor="rgba(230, 57, 70, 0.1)",
        name="Finish Rate %"
    ))
    fig.update_layout(
        plot_bgcolor="#111", paper_bgcolor="#111", font_color="#f0f0f0",
        font_family="Barlow", xaxis_title="Year", yaxis_title="Finish Rate %",
        hovermode="x unified", showlegend=False,
        xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div style="font-family: Bebas Neue; font-size: 24px; letter-spacing: 2px; color: #888; margin-bottom: 16px;">FIGHTS PER YEAR</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=yearly["year"], y=yearly["total_fights"],
        marker_color="#e63946", marker_line_width=0, name="Fights"
    ))
    fig2.update_layout(
        plot_bgcolor="#111", paper_bgcolor="#111", font_color="#f0f0f0",
        font_family="Barlow", xaxis_title="Year", yaxis_title="Total Fights",
        showlegend=False, xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222")
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div style="font-family: Bebas Neue; font-size: 24px; letter-spacing: 2px; color: #888; margin-bottom: 16px;">FINISH RATE BY DIVISION</div>', unsafe_allow_html=True)
    div_finish = (
        fights.groupby("weight_class")
        .agg(total_fights=("finish", "count"), finishes=("finish", "sum")).reset_index()
    )
    div_finish["finish_rate"] = (div_finish["finishes"] / div_finish["total_fights"] * 100).round(1)
    div_finish = div_finish.sort_values("finish_rate", ascending=False)

    fig3 = px.bar(div_finish, x="weight_class", y="finish_rate",
        color="finish_rate", color_continuous_scale=["#2a2a2a", "#e63946"],
        labels={"weight_class": "", "finish_rate": "Finish Rate %"},
        title="")
    fig3.update_layout(
        plot_bgcolor="#111", paper_bgcolor="#111", font_color="#f0f0f0",
        font_family="Barlow", coloraxis_showscale=False, xaxis_tickangle=-30
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div style="font-family: Bebas Neue; font-size: 24px; letter-spacing: 2px; color: #888; margin-bottom: 16px;">AVERAGE FIGHT TIME BY DIVISION</div>', unsafe_allow_html=True)
    div_time = (
        fights.groupby("weight_class")["total_seconds"].mean()
        .reset_index(name="avg_seconds").dropna()
    )
    div_time["avg_minutes"] = (div_time["avg_seconds"] / 60).round(2)
    div_time = div_time.sort_values("avg_minutes", ascending=False)

    fig4 = px.bar(div_time, x="weight_class", y="avg_minutes",
        color="avg_minutes", color_continuous_scale=["#2a2a2a", "#e63946"],
        labels={"weight_class": "", "avg_minutes": "Avg Fight Time (mins)"},
        title="")
    fig4.update_layout(
        plot_bgcolor="#111", paper_bgcolor="#111", font_color="#f0f0f0",
        font_family="Barlow", coloraxis_showscale=False, xaxis_tickangle=-30
    )
    st.plotly_chart(fig4, use_container_width=True)