import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업의 기준 기간 끝
MIN_DAYS = 300        # 연평균을 계산하기 위한 최소 관측일 수
RECENT_YEARS = 20     # 최근 구간 비교에 쓸 연수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")
st.title("🌡️ 기온 예측기")
st.caption("서울 연평균기온의 변화 추세를 직선(선형 회귀)으로 맞추고, 원하는 해의 기온을 예상합니다.")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly_mean():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year

    # 기준 기간(2025년까지)만 사용
    df = df[df["연도"] <= LAST_YEAR]

    # 연도별 관측일 수와 평균, 관측일 300일 미만인 해는 제외
    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(관측일수="count", 연평균기온="mean")
        .reset_index()
    )
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS].reset_index(drop=True)
    return yearly


try:
    yearly = load_yearly_mean()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.warning("회귀 직선을 만들기에 데이터가 부족합니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

# 회귀 직선과 상관계수
slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

# 연도 슬라이더와 예상 기온
YEAR_MIN, YEAR_MAX = 1900, 2100

if "year_slider" not in st.session_state:
    st.session_state["year_slider"] = 2050
if "year_input" not in st.session_state:
    st.session_state["year_input"] = st.session_state["year_slider"]


def sync_from_slider():
    st.session_state["year_input"] = st.session_state["year_slider"]


def sync_from_input():
    st.session_state["year_slider"] = st.session_state["year_input"]


col_s, col_n = st.columns([3, 1])
col_s.slider("예상할 연도를 고르세요", min_value=YEAR_MIN, max_value=YEAR_MAX,
             step=1, key="year_slider", on_change=sync_from_slider)
col_n.number_input("직접 입력", min_value=YEAR_MIN, max_value=YEAR_MAX,
                   step=1, key="year_input", on_change=sync_from_input)

year = int(st.session_state["year_slider"])
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
        <div style="font-size:1.2rem; color:gray;">{year}년의 예상 연평균기온</div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1.1;">{pred:.2f} ℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 산점도 + 회귀 직선 + 예상 지점
x_min = min(x.min(), year)
x_max = max(x.max(), year)
line_x = np.array([x_min, x_max])
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=x, y=y, mode="markers", name="연평균기온",
               marker=dict(size=7, opacity=0.75))
)
fig.add_trace(
    go.Scatter(x=line_x, y=line_y, mode="lines", name="회귀 직선",
               line=dict(width=3, color="crimson"))
)
fig.add_trace(
    go.Scatter(x=[year], y=[pred], mode="markers", name=f"{year}년 예상",
               marker=dict(size=14, color="gold", line=dict(width=2, color="black")))
)
fig.update_layout(
    title="서울 연도별 평균기온과 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=80, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# 상관계수
st.metric("상관계수 (연도 vs 연평균기온)", f"{r:.3f}")

# 기울기를 "100년당 변화"로 환산해 전체 기간과 최근 20년 비교
st.subheader("100년에 몇 도 오르는가?")


def big_number(title, value, note):
    return f"""
    <div style="text-align:center; padding:0.8rem 0.4rem; border:1px solid #8884; border-radius:12px;">
        <div style="font-size:1.05rem; color:gray;">{title}</div>
        <div style="font-size:3rem; font-weight:800; line-height:1.15;">{value}</div>
        <div style="font-size:0.9rem; color:gray;">{note}</div>
    </div>
    """


recent = yearly[yearly["연도"] > yearly["연도"].max() - RECENT_YEARS]
col_all, col_recent = st.columns(2)

col_all.markdown(
    big_number(
        "전체 기간",
        f"{slope * 100:+.2f} ℃ / 100년",
        f"{int(yearly['연도'].min())}~{int(yearly['연도'].max())}년 · {len(yearly)}개 연도",
    ),
    unsafe_allow_html=True,
)

if len(recent) >= 2:
    rx = recent["연도"].to_numpy(dtype=float)
    ry = recent["연평균기온"].to_numpy(dtype=float)
    r_slope, r_intercept = np.polyfit(rx, ry, 1)
    col_recent.markdown(
        big_number(
            f"최근 {RECENT_YEARS}년",
            f"{r_slope * 100:+.2f} ℃ / 100년",
            f"{int(recent['연도'].min())}~{int(recent['연도'].max())}년 · {len(recent)}개 연도",
        ),
        unsafe_allow_html=True,
    )
    ratio = r_slope / slope if slope != 0 else float("nan")
    if np.isfinite(ratio):
        st.write(f"최근 {RECENT_YEARS}년의 상승 속도는 전체 기간의 약 **{ratio:.1f}배**입니다.")
    st.caption(
        f"최근 {RECENT_YEARS}년은 표본이 적어 기울기가 해마다 크게 흔들릴 수 있습니다. "
        "또한 '100년당'은 현재 추세가 100년간 이어질 경우로 환산한 값입니다."
    )
else:
    col_recent.warning("최근 구간의 데이터가 부족합니다.")

st.info(
    f"회귀 직선을 만든 데이터: **{len(yearly)}개 연도**, "
    f"**{int(yearly['연도'].min())}년 ~ {int(yearly['연도'].max())}년**  \n"
    f"(기준 기간 {LAST_YEAR}년까지, 관측일 {MIN_DAYS}일 미만인 해는 제외)"
)
st.caption(f"회귀식: 연평균기온 = {slope:.5f} × 연도 + ({intercept:.3f})")
