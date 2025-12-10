import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import os
from matplotlib import font_manager

# ---- 한글 폰트 (배포 서버용) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "fonts", "NanumGothic.ttf")

# 폰트 등록
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
# -------------------------------
# 기본 설정
# -------------------------------
st.set_page_config(
    page_title="온실 파프리카 곰팡이병 정밀 방제 시스템",
    layout="centered",
)

st.title("🫑 온실 파프리카 곰팡이병 정밀 방제 시스템")

st.markdown(
"""
모델이 예측한 **병해 유무(has_disease)** 와  
**3단계 중증도(severity_high_3class: 0/1/2\\)** 를 이용해  
구역별 병해율과 중증도(A/B/C 단계)를 계산하여 ***방제 계획을 제시하는 시스템입니다.***
"""
)
 

# -------------------------------
# 0. 사이드바 설정
# -------------------------------
st.sidebar.header("⚙️ 설정")

top_k = st.sidebar.slider("방제 우선순위 Top-K 구역", 1, 12, 3)

zone_assign_method = st.sidebar.radio(
    "구역 배정 방식",
    ["랜덤 균등 분배", "해시 기반 고정 배정"],
    help="실제 좌표가 없어서 가상의 규칙으로 12구역(A1~C4)에 이미지를 배정합니다.",
)

seed = st.sidebar.number_input(
    "랜덤 시드 (랜덤 균등 분배일 때만 사용)", value=42, step=1
)

# 이 값 = 경로에 포함할 구역 수 (최소 2, 최대 12)
path_len = st.sidebar.slider("예상 감염 경로에 포함할 구역 수", 2, 12, 6)

st.sidebar.markdown("---")
st.sidebar.markdown("**입력 파일 포맷 예시**")
st.sidebar.code(
"""
- 파일 형태 : csv
- 필수 컬럼 : 
stem(이미지 고유 아이디),
image_path(원본 이미지 경로), 
has_disease(병해 유무 예측), 
severity_3class(흰가루병 중증도)
""",
    language="text",
)

# -------------------------------
# 1. 파일 업로드
# -------------------------------
uploaded_file = st.file_uploader(
    " 입력 파일 업로드", type=["csv"]
)

if uploaded_file is None:
    st.info("👆 위에 중증도 결과 CSV 파일을 업로드해 주세요.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ---- 중증도 컬럼 이름 유연하게 찾기 ----
severity_candidates = [
    "severity_high_3class",
    "severity_3class",
    "severity_class3",
    "severity_high3class",
]
sev_col = None
for c in severity_candidates:
    if c in df.columns:
        sev_col = c
        break

if sev_col is None:
    st.error(f"3클래스 중증도 컬럼을 찾을 수 없습니다. 현재 컬럼들: {list(df.columns)}")
    st.stop()

df = df.rename(columns={sev_col: "severity_high_3class"})

required_cols = {"stem", "image_path", "has_disease", "severity_high_3class"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"다음 컬럼이 없습니다: {missing}")
    st.stop()

df["stem"] = df["stem"].astype(str)
df["has_disease"] = df["has_disease"].astype(int)
df["severity_high_3class"] = df["severity_high_3class"].astype(int)

st.success("✅ 파일 업로드 완료!")

st.markdown("### 1️⃣ 원시 예측 결과 요약")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("총 이미지 수", len(df))
with col2:
    st.metric("병해 비율", f"{df['has_disease'].mean():.3f}")
with col3:
    st.metric("평균 중증도(0~2)", f"{df['severity_high_3class'].mean():.3f}")
with col4:
    sev2_rate_global = (df["severity_high_3class"] == 2).mean()
    st.metric("상(2) 비율", f"{sev2_rate_global:.3f}")

st.dataframe(df.head(), use_container_width=True)

# -------------------------------
# 2. 12구역(3×4) 가상 온실 zone 배정
# -------------------------------
st.markdown("### 2️⃣ 3×4 가상 온실 구역 배정")

# 3(가로: A,B,C) × 4(세로: 1~4) = 12구역
cols = ["A", "B", "C"]   # 좌, 가운데, 우
rows = [1, 2, 3, 4]      # 앞 → 뒤
zones = [f"{c}{r}" for r in rows for c in cols]  # A1,B1,C1,A2,B2,...

n_zones = len(zones)

stems = df["stem"].unique()
df_zone = pd.DataFrame({"stem": stems})

if zone_assign_method == "랜덤 균등 분배":
    np.random.seed(int(seed))
    df_zone = df_zone.sample(frac=1).reset_index(drop=True)

    n = len(df_zone)
    per_zone = n // n_zones
    zone_list = []
    for i, z in enumerate(zones):
        start = i * per_zone
        end = (i + 1) * per_zone if i < n_zones - 1 else n
        zone_list.extend([z] * (end - start))
    df_zone["zone"] = zone_list
else:
    def assign_zone_hash(stem: str) -> str:
        return zones[hash(stem) % n_zones]
    df_zone["zone"] = df_zone["stem"].apply(assign_zone_hash)

st.write("구역 배정 예시:")
st.dataframe(df_zone.head(), use_container_width=True)

df_merged = df.merge(df_zone, on="stem")

# -------------------------------
# 3. 구역별 병해·중증도 통계
# -------------------------------
st.markdown("### 3️⃣ 구역별 병해·중증도 통계")

def rate_of(x, cls):
    return (x == cls).mean()

zone_stats = (
    df_merged.groupby("zone")
    .agg(
        n_images=("stem", "count"),
        disease_rate=("has_disease", "mean"),
        sev0_rate=("severity_high_3class", lambda x: rate_of(x, 0)),
        sev1_rate=("severity_high_3class", lambda x: rate_of(x, 1)),
        sev2_rate=("severity_high_3class", lambda x: rate_of(x, 2)),
        severity_mean=("severity_high_3class", "mean"),
    )
    .reset_index()
)

# ---- 중증도 평균을 A/B/C 단계로 변환 (상/중/하)
def stage_from_mean(sev_mean):
    # 숫자는 예시, 데이터 보고 조정해도 됨
    if sev_mean >= 1.5:
        return "A"   # 상
    elif sev_mean >= 0.8:
        return "B"   # 중
    else:
        return "C"   # 하

zone_stats["stage_ABC"] = zone_stats["severity_mean"].apply(stage_from_mean)

# ---- 위험도(risk_level) ----
def classify_risk(row):
    sev2 = row["sev2_rate"]
    sev1 = row["sev1_rate"]
    dr = row["disease_rate"]

    if sev2 >= 0.20:
        return "High"
    elif sev2 >= 0.10 or (sev1 + sev2) >= 0.40 or dr >= 0.35:
        return "Medium"
    else:
        return "Low"

def recommend_action(risk_level):
    if risk_level == "High":
        return "오늘 중 드론 방제 및 집중 관리 권장"
    elif risk_level == "Medium":
        return "1~2일 내 재촬영 및 부분 방제 검토"
    else:
        return "일반 모니터링 (우선순위 낮음)"

zone_stats["risk_level"] = zone_stats.apply(classify_risk, axis=1)
zone_stats["recommendation"] = zone_stats["risk_level"].apply(recommend_action)

st.dataframe(
    zone_stats[
        [
            "zone",
            "n_images",
            "disease_rate",
            "sev0_rate",
            "sev1_rate",
            "sev2_rate",
            "severity_mean",
            "stage_ABC",
            "risk_level",
            "recommendation",
        ]
    ],
    use_container_width=True,
)

# -------------------------------
# 4. 3×4 온실 레이아웃 (row/col 좌표)
# -------------------------------
layout_rows = []
layout_cols = []
layout_zones = []

for r_idx, r in enumerate(rows):      # y축: 0~3
    for c_idx, c in enumerate(cols):  # x축: 0~2
        layout_zones.append(f"{c}{r}")
        layout_rows.append(r_idx)
        layout_cols.append(c_idx)

layout_df = pd.DataFrame(
    {"zone": layout_zones, "row": layout_rows, "col": layout_cols}
)

zone_map = layout_df.merge(zone_stats, on="zone", how="left")
zone_map = zone_map.fillna(0)

# -------------------------------
# 5. 감염 경로 계산 함수 (값 + 거리 동시에 고려)
# -------------------------------
def manhattan_dist(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)


def compute_infection_path(zone_stats, zone_map, max_zones=6):
    """
    값(severity_mean)과 거리(row, col)를 동시에 고려한 경로 생성.

    - start: severity_mean이 가장 큰 구역
    - 이후: 아직 방문 안 한 구역 중
        1) severity_mean <= 현재 sev  (값은 내려가는 방향)
        2) 그 중에서 맨해튼 거리가 가장 가까운 칸 선택
    - 그런 칸이 없으면: 남은 구역 중 '가장 가까운 칸'으로 이동 (값 조건 완화)
    """
    zs = zone_stats.set_index("zone")
    zm = zone_map.set_index("zone")

    # 시작점: 가장 심한 구역
    start_zone = zone_stats.sort_values("severity_mean", ascending=False).iloc[0]["zone"]
    path = [start_zone]
    current = start_zone

    max_zones = max(2, min(max_zones, len(zone_stats)))  # 2 ~ N 사이로 제한

    while len(path) < max_zones:
        cur_sev = zs.loc[current, "severity_mean"]
        cur_r, cur_c = zm.loc[current, ["row", "col"]]

        # 아직 방문 안 한 구역들
        remaining = [z for z in zs.index if z not in path]
        if not remaining:
            break

        # 1차 후보: 현재보다 덜 심한 구역들
        cand = [
            z
            for z in remaining
            if zs.loc[z, "severity_mean"] <= cur_sev + 1e-8
        ]

        # 없으면 값 조건 완화: 그냥 남은 것 중에서 고름
        if not cand:
            cand = remaining

        # 거리 + 값 기준으로 최적 후보 선택
        best_zone = None
        best_score = None  # (거리, -sev) 같은 튜플로 비교

        for z in cand:
            r, c = zm.loc[z, ["row", "col"]]
            d = manhattan_dist(cur_r, cur_c, r, c)
            sev = zs.loc[z, "severity_mean"]
            score = (d, -sev)  # 거리 우선, 거리 같으면 sev 큰 쪽

            if best_score is None or score < best_score:
                best_score = score
                best_zone = z

        if best_zone is None:
            break

        path.append(best_zone)
        current = best_zone

    return path  # [start, ..., last]

# -------------------------------
# 6. Heatmap 함수들
# -------------------------------
def plot_heatmap_discrete(value_col, title, label=""):
    heat = zone_map.pivot(index="row", columns="col", values=value_col)
    values = heat.values.astype(float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    if vmin == vmax:
        vmin -= 0.01
        vmax += 0.01

    step = (vmax - vmin) / 3.0
    bounds = [vmin, vmin + step, vmin + 2 * step, vmax]
    cmap = mcolors.ListedColormap(["#3b4cc0", "#fdda45", "#b40426"])
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(4, 8))
    im = ax.imshow(heat.values, origin="upper", cmap=cmap, norm=norm)

    ax.set_title(title)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["A", "B", "C"])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["1", "2", "3", "4"])

    for _, r in zone_map.iterrows():
        ax.text(
            r["col"],
            r["row"],
            f"{r['zone']}\n{r[value_col]:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            weight="bold",
        )

    cbar = fig.colorbar(im, ax=ax, boundaries=bounds, fraction=0.035, pad=0.02)
    cbar.set_ticks(
        [
            (bounds[0] + bounds[1]) / 2,
            (bounds[1] + bounds[2]) / 2,
            (bounds[2] + bounds[3]) / 2,
        ]
    )
    cbar.set_ticklabels([f"{label} 낮음", f"{label} 보통", f"{label} 높음"])

    st.pyplot(fig)


def plot_severity_with_arrow(max_steps=6):
    """
    평균 중증도 Heatmap + '값은 내려가고, 공간은 가까운 곳'으로 이동하는 경로 시각화
    """

    # 새 경로 계산: 값 + 거리 기반
    path = compute_infection_path(zone_stats, zone_map, max_zones=max_steps)

    start_zone = path[0]          # 가장 중증도가 높은 구역 (출발점)
    end_zone = path[-1]           # 마지막 예상 감염 구역
    mid_zones = path[1:-1]        # 중간 경로 구역들

    heat = zone_map.pivot(index="row", columns="col", values="severity_mean")
    values = heat.values.astype(float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    fig, ax = plt.subplots(figsize=(4, 6))
    im = ax.imshow(heat.values, origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)

    ax.set_title("평균 중증도 및 예상 감염 경로")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["A", "B", "C"])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["1", "2", "3", "4"])
    ax.set_xlabel("열 (A–C)")
    ax.set_ylabel("행 (1–4)")

    # 각 칸에 zone / 평균 중증도 / 단계(A/B/C) 표시
    for _, r in zone_map.iterrows():
        ax.text(
            r["col"],
            r["row"],
            f"{r['zone']}\n{r['severity_mean']:.2f}\n({r['stage_ABC']})",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            weight="bold",
        )

    # 컬러바 폭 줄이기
    cbar = fig.colorbar(
        im,
        ax=ax,
        label="평균 중증도 (0=정상, 2=상)",
        fraction=0.035,
        pad=0.02,
    )

    # 🟢 출발 구역(가장 심한 곳) - 초록 테두리
    sr, sc = zone_map.loc[zone_map["zone"] == start_zone, ["row", "col"]].values[0]
    rect_start = plt.Rectangle(
        (sc - 0.5, sr - 0.5),
        1,
        1,
        fill=False,
        edgecolor="#00ff00",
        linewidth=3,
    )
    ax.add_patch(rect_start)

    # 🟡 중간 경로 구역들 - 주황 점선 테두리
    for z in mid_zones:
        rr, cc = zone_map.loc[zone_map["zone"] == z, ["row", "col"]].values[0]
        rect_mid = plt.Rectangle(
            (cc - 0.5, rr - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#ffcc00",
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(rect_mid)

    # 🔴 마지막 예상 감염 구역 - 빨간 테두리
    er, ec = zone_map.loc[zone_map["zone"] == end_zone, ["row", "col"]].values[0]
    rect_end = plt.Rectangle(
        (ec - 0.5, er - 0.5),
        1,
        1,
        fill=False,
        edgecolor="red",
        linewidth=3,
    )
    ax.add_patch(rect_end)

    # ➡ 화살표: path[0] → path[1] → path[2] → ...
    for i in range(len(path) - 1):
        z_from = path[i]
        z_to = path[i + 1]
        r1, c1 = zone_map.loc[zone_map["zone"] == z_from, ["row", "col"]].values[0]
        r2, c2 = zone_map.loc[zone_map["zone"] == z_to, ["row", "col"]].values[0]

        ax.annotate(
            "",
            xy=(c2, r2),
            xytext=(c1, r1),
            arrowprops=dict(arrowstyle="->", color="red", lw=2.5),
        )

    st.pyplot(fig)

    # 🔤 텍스트 설명
    st.markdown(f"🟢 **출발 구역(현재 가장 중증도가 높은 구역)**: `{start_zone}`")
    st.markdown(f"🔴 **마지막 예상 감염 구역**: `{end_zone}`")

    if len(path) > 1:
        path_str = " → ".join(path)
        st.markdown(f"🟡 **예상 감염 경로:** {path_str}")
    else:
        st.markdown("🟡 추가 예상 감염 경로를 계산할 수 없습니다.")

# -------------------------------
# 7. 시각화 탭
# -------------------------------
st.markdown("### 4️⃣ 온실 병해·중증도 지도 및 감염 경로 시각화")

tab1, tab2, tab3 = st.tabs(
    ["병해 발생 비율", "중증도 + 감염 경로", "중증도 상(high) 비율"]
)

with tab1:
    plot_heatmap_discrete("disease_rate", "구역별 병해 발생 비율", label="병해율")

with tab2:
    plot_severity_with_arrow(max_steps=path_len)

with tab3:
    plot_heatmap_discrete("sev2_rate", "구역별 중증도 상(high) 비율", label="sev2")

# -------------------------------
# 8. 드론 방제 우선순위 Top-K
# -------------------------------
st.markdown("### 5️⃣ 드론 방제 우선순위 Top-K 구역")

top_zones = zone_stats.sort_values("sev2_rate", ascending=False).head(top_k)
st.dataframe(
    top_zones[
        [
            "zone",
            "n_images",
            "disease_rate",
            "sev0_rate",
            "sev1_rate",
            "sev2_rate",
            "severity_mean",
            "stage_ABC",
            "risk_level",
            "recommendation",
        ]
    ],
    use_container_width=True,
)

# -------------------------------
# 9. 오늘의 방제 계획 요약
# -------------------------------
st.markdown("### 6️⃣ 오늘의 방제 계획 제안")

high_zones = zone_stats[zone_stats["risk_level"] == "High"].sort_values(
    "sev2_rate", ascending=False
)
med_zones = zone_stats[zone_stats["risk_level"] == "Medium"].sort_values(
    "sev2_rate", ascending=False
)
low_zones = zone_stats[zone_stats["risk_level"] == "Low"].sort_values(
    "sev2_rate", ascending=False
)

if len(high_zones) == 0:
    st.write("✅ High 위험 구역은 없습니다. 전체적으로 심각한 병해는 관찰되지 않습니다.")
else:
    st.write("🚨 **가장 먼저 방제가 필요한 High 위험 구역:**")
    for _, r in high_zones.iterrows():
        st.write(
            f"- {r['zone']}: 상(high) 비율={r['sev2_rate']:.2f}, "
            f"평균 중증도={r['severity_mean']:.2f} → {r['recommendation']}"
        )

if len(med_zones) > 0:
    st.write("🟡 **중간 수준의 위험(Medium) 구역:**")
    for _, r in med_zones.iterrows():
        st.write(
            f"- {r['zone']}: 상(2) 비율={r['sev2_rate']:.2f}, "
            f"평균 중증도={r['severity_mean']:.2f} → {r['recommendation']}"
        )

if len(low_zones) > 0:
    st.write("🟢 **Low 구역:** 일반 모니터링만 수행해도 무방합니다.")
