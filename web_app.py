import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import poisson
import os

st.set_page_config(page_title="AI 足球预测器 - 完整版", page_icon="⚽")
st.title("⚽ AI 足球赛前预测器 V7.0 (完整版)")
st.write("支持中英文球队搜索，数据本地缓存，一键预测胜平负和精确比分！")

# ================= 中英文球队对照表（精选五大联赛常用队名）=================
TEAM_MAP = {
    # 英超
    "曼城": "Man City", "阿森纳": "Arsenal", "利物浦": "Liverpool", "曼联": "Man United",
    "切尔西": "Chelsea", "热刺": "Spurs", "纽卡斯尔": "Newcastle", "阿斯顿维拉": "Aston Villa",
    "布莱顿": "Brighton", "西汉姆联": "West Ham", "埃弗顿": "Everton", "富勒姆": "Fulham",
    "水晶宫": "Crystal Palace", "布伦特福德": "Brentford", "狼队": "Wolves", "伯恩茅斯": "Bournemouth",
    "诺丁汉森林": "Nott'm Forest", "卢顿": "Luton", "伯恩利": "Burnley", "谢菲联": "Sheffield United",
    # 西甲
    "皇家马德里": "Real Madrid", "巴塞罗那": "Barcelona", "马德里竞技": "Ath Madrid",
    "塞维利亚": "Sevilla", "皇家社会": "Sociedad", "毕尔巴鄂": "Ath Bilbao", "比利亚雷亚尔": "Villarreal",
    # 德甲
    "拜仁慕尼黑": "Bayern Munich", "多特蒙德": "Dortmund", "莱比锡红牛": "RB Leipzig",
    "勒沃库森": "Leverkusen", "法兰克福": "Ein Frankfurt", "斯图加特": "Stuttgart", "云达不莱梅": "Werder Bremen",
    # 意甲
    "国际米兰": "Inter", "AC米兰": "AC Milan", "尤文图斯": "Juventus", "那不勒斯": "Napoli",
    "罗马": "Roma", "拉齐奥": "Lazio", "亚特兰大": "Atalanta", "佛罗伦萨": "Fiorentina",
    # 法甲
    "巴黎圣日耳曼": "Paris SG", "马赛": "Marseille", "里昂": "Lyon", "摩纳哥": "Monaco",
    "里尔": "Lille", "朗斯": "Lens", "雷恩": "Rennes", "尼斯": "Nice"
}

@st.cache_data
def load_and_cache_data():
    """下载数据并缓存到本地，防止每次刷新都重新下载"""
    cache_file = "epl_data_cache.csv"
    if os.path.exists(cache_file):
        print("从本地缓存加载数据...")
        return pd.read_csv(cache_file)

    print("首次运行，正在从网络下载数据...")
    leagues = {'E0': '英超', 'SP1': '西甲', 'D1': '德甲', 'I1': '意甲', 'F1': '法甲'}
    seasons = ['2526', '2425', '2424', '2324', '2223']
    dfs = []
    for season in seasons:
        for code, name in leagues.items():
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            try:
                d = pd.read_csv(url)
                if len(d) > 50:
                    dfs.append(d)
            except: pass
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(cache_file, index=False)
    return df

@st.cache_resource
def load_model():
    df = load_and_cache_data()
    cols_needed = ['B365H', 'B365D', 'B365A', 'B365>2.5', 'B365<2.5', 'AHh', 'B365AHH', 'B365AHA', 'FTHG', 'FTAG']
    df = df.dropna(subset=cols_needed)
    
    # 去水处理
    df['Overround'] = 1/df['B365H'] + 1/df['B365D'] + 1/df['B365A']
    df['Prob_H'] = (1 / df['B365H']) / df['Overround']
    df['Prob_D'] = (1 / df['B365D']) / df['Overround']
    df['Prob_A'] = (1 / df['B365A']) / df['Overround']
    df['AH_Overround'] = 1/df['B365AHH'] + 1/df['B365AHA']
    df['Prob_AH_Home'] = (1 / df['B365AHH']) / df['AH_Overround']
    df['Prob_Over25'] = 1 / df['B365>2.5']
    df['Prob_Under25'] = 1 / df['B365<2.5']
    
    all_teams = sorted(list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())))
    team_to_code = {team: i for i, team in enumerate(all_teams)}
    df['HomeTeam_Code'] = df['HomeTeam'].map(team_to_code)
    df['AwayTeam_Code'] = df['AwayTeam'].map(team_to_code)
    
    features = ['Prob_H', 'Prob_D', 'Prob_A', 'AHh', 'Prob_AH_Home', 'Prob_Over25', 'Prob_Under25', 'HomeTeam_Code', 'AwayTeam_Code']
    X = df[features]
    model_home = xgb.XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=5, random_state=42).fit(X, df['FTHG'])
    model_away = xgb.XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=5, random_state=42).fit(X, df['FTAG'])
    
    return model_home, model_away, all_teams, team_to_code

with st.spinner('正在加载模型和数据（第一次会下载，之后秒开）...'):
    model_home, model_away, all_teams, team_to_code = load_model()

# ================= 展示所有球队 =================
with st.expander("📋 点击查看所有可预测球队（英文）"):
    st.write(", ".join(all_teams))

# ================= 输入界面 =================
col1, col2 = st.columns(2)
with col1:
    home_input = st.text_input("主队 (可输入中文如'曼城'，或英文'Man City')", value="曼城")
with col2:
    away_input = st.text_input("客队 (可输入中文如'阿森纳'，或英文'Arsenal')", value="阿森纳")

st.markdown("---")
st.subheader("📊 输入博彩公司赔率")

col3, col4, col5 = st.columns(3)
with col3:
    b365h = st.number_input("主胜赔率 (B365H)", value=1.80, step=0.01)
    b365d = st.number_input("平局赔率 (B365D)", value=3.60, step=0.01)
with col4:
    b365a = st.number_input("客胜赔率 (B365A)", value=4.50, step=0.01)
    ahh = st.number_input("让球盘口 (AHh)", value=-0.5, step=0.25)
with col5:
    over25 = st.number_input("大球赔率 >2.5", value=2.10, step=0.01)
    under25 = st.number_input("小球赔率 <2.5", value=1.80, step=0.01)
    b365_ahh = st.number_input("主队让球赔率 (B365AHH)", value=1.90, step=0.01)
    b365_aha = st.number_input("客队让球赔率 (B365AHA)", value=1.90, step=0.01)

if st.button("开始 AI 预测", type="primary", use_container_width=True):
    # 处理输入，先尝试中英文匹配
    home_team = TEAM_MAP.get(home_input, home_input)
    away_team = TEAM_MAP.get(away_input, away_input)
    
    if home_team == away_team:
        st.error("主队和客队不能是同一支球队！")
    elif home_team not in team_to_code or away_team not in team_to_code:
        st.error(f"无法识别球队：{home_team} 或 {away_team}。请点击上方展开列表查看准确的英文名。")
    else:
        overround = 1/b365h + 1/b365d + 1/b365a
        prob_h = (1/b365h) / overround
        prob_d = (1/b365d) / overround
        prob_a = (1/b365a) / overround
        ah_overround = 1/b365_ahh + 1/b365_aha
        prob_ah_home = (1/b365_ahh) / ah_overround
        
        data = {
            'Prob_H': [prob_h], 'Prob_D': [prob_d], 'Prob_A': [prob_a],
            'AHh': [ahh], 'Prob_AH_Home': [prob_ah_home],
            'Prob_Over25': [1/over25], 'Prob_Under25': [1/under25],
            'HomeTeam_Code': [team_to_code[home_team]], 'AwayTeam_Code': [team_to_code[away_team]]
        }
        new_match = pd.DataFrame(data)
        
        lambda_home = max(0.1, model_home.predict(new_match)[0])
        lambda_away = max(0.1, model_away.predict(new_match)[0])
        
        st.success(f"⚽ AI 预测：{home_input} {lambda_home:.2f} 球，{away_input} {lambda_away:.2f} 球")
        
        max_goals = 6
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob_matrix[i][j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                
        prob_win = np.sum(np.tril(prob_matrix, -1))
        prob_draw = np.sum(np.diag(prob_matrix))
        prob_loss = np.sum(np.triu(prob_matrix, 1))
        
        st.write("### 📈 赛果概率分布（基于泊松分布计算）")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("主胜概率", f"{prob_win*100:.1f}%")
        col_b.metric("平局概率", f"{prob_draw*100:.1f}%")
        col_c.metric("客胜概率", f"{prob_loss*100:.1f}%")
        
        flat_probs = [(i, j, prob_matrix[i][j]) for i in range(max_goals+1) for j in range(max_goals+1)]
        top_scores = sorted(flat_probs, key=lambda x: x[2], reverse=True)[:5]
        st.write("### 🎯 最可能的5个具体比分")
        for score in top_scores:
            st.write(f"比分 **{score[0]} : {score[1]}** — 概率 {score[2]*100:.1f}%")