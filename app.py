import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import heapq
import zlib
import os
from matplotlib import font_manager

# 한글 폰트(배포 서버용)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "fonts", "NanumGothic-Regular.ttf")

# 폰트 등록
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = "NanumGothic"

plt.rcParams["axes.unicode_minus"] = False

# 기본 설정 
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
 
# 사이드바
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
route_alpha = 0


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

# 파일 업로드
uploaded_file = st.file_uploader(
    " 입력 파일 업로드", type=["csv"]
)

if uploaded_file is None:
    st.info("👆 위에 중증도 결과 CSV 파일을 업로드해 주세요.")
    st.stop()

df = pd.read_csv(uploaded_file)

# 중증도 컬럼 찾기 
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


# 12구역 가상 온실 배정 
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
        # hash()는 실행할 때마다 값이 바뀌므로, 항상 같은 값을 주는 crc32 사용
        return zones[zlib.crc32(stem.encode("utf-8")) % n_zones]
    df_zone["zone"] = df_zone["stem"].apply(assign_zone_hash)

st.write("구역 배정 예시:")
st.dataframe(df_zone.head(), use_container_width=True)

df_merged = df.merge(df_zone, on="stem")

# 구역별 병해, 중증도 통계 
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

# 중증도 평균 A/B/C 단계로 변환 (상/중/하)
def stage_from_mean(sev_mean):
    # 숫자는 예시, 데이터 보고 조정해도 됨
    if sev_mean >= 1.5:
        return "A"   # 상
    elif sev_mean >= 0.8:
        return "B"   # 중
    else:
        return "C"   # 하

zone_stats["stage_ABC"] = zone_stats["severity_mean"].apply(stage_from_mean)

# 위험도
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


# 3×4 온실 레이아웃 (row/col 좌표)
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


# 5. 감염 경로 계산 함수 (값 + 거리 동시에 고려)
# 맨해튼 거리
def manhattan_dist(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)

# 그리디 알고리즘 
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

    # 시작점 : 가장 심한 구역
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

        # 1차 후보 : 현재보다 덜 심한 구역들
        cand = [
            z
            for z in remaining
            if zs.loc[z, "severity_mean"] <= cur_sev + 1e-8
        ]

        # 없으면 값 조건 완화 : 그냥 남은 것 중에서 고름
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


# Dijkstra / A* 경로 탐색 함수

def get_severity_norm():
    """
    각 구역의 평균 중증도 severity_mean을 0~1로 정규화.
    경로 탐색 비용 계산에 사용.
    """
    max_sev = zone_map["severity_mean"].max()

    if max_sev > 0:
        sev_norm = zone_map.set_index("zone")["severity_mean"] / max_sev
    else:
        sev_norm = zone_map.set_index("zone")["severity_mean"] * 0

    return sev_norm.to_dict()


def get_neighbors(zone):
    """
    현재 zone에서 상하좌우로 인접한 구역만 반환.
    예: A1의 이웃은 B1, A2
    """
    cur = zone_map[zone_map["zone"] == zone].iloc[0]
    r, c = cur["row"], cur["col"]

    temp = zone_map.copy()
    temp["dist"] = abs(temp["row"] - r) + abs(temp["col"] - c)

    neighbors = temp[temp["dist"] == 1]["zone"].tolist()

    return neighbors


def heuristic(a, b):
    """
    A*의 h(n): 현재 구역에서 목표 구역까지의 맨해튼 거리.
    격자 이동이므로 맨해튼 거리를 사용.
    """
    za = zone_map[zone_map["zone"] == a].iloc[0]
    zb = zone_map[zone_map["zone"] == b].iloc[0]

    return abs(za["row"] - zb["row"]) + abs(za["col"] - zb["col"])


def move_cost(next_zone, alpha=0.5):
    """
    이동 비용 w(u, v).

    w(u, v) = 1 + α × (1 - severity_norm(v))

    다음 구역의 중증도가 높을수록 severity_norm(v)가 커지고,
    그러면 이동 비용이 작아진다.
    즉, 중증도 높은 구역을 경로상에서 더 선호하게 된다.
    """
    sev_norm = get_severity_norm()
    return 1 + alpha * (1 - sev_norm[next_zone])


def search_path(start, goal, algorithm="dijkstra", alpha=0.5):
    """
    Dijkstra 또는 A*로 start → goal 경로를 찾는다.

    Dijkstra:
        f(n) = g(n)

    A*:
        f(n) = g(n) + h(n)
    """
    pq = []
    heapq.heappush(pq, (0, start))

    g = {start: 0}
    parent = {start: None}
    visited = set()

    while pq:
        _, current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)

        if current == goal:
            break

        for nxt in get_neighbors(current):
            new_g = g[current] + move_cost(nxt, alpha)

            if new_g < g.get(nxt, float("inf")):
                g[nxt] = new_g
                parent[nxt] = current

                if algorithm == "dijkstra":
                    priority = new_g

                elif algorithm == "astar":
                    priority = new_g + heuristic(nxt, goal)

                else:
                    raise ValueError("algorithm은 'dijkstra' 또는 'astar'만 가능")

                heapq.heappush(pq, (priority, nxt))

    if goal not in parent:
        return [], float("inf"), len(visited)

    path = []
    cur = goal

    while cur is not None:
        path.append(cur)
        cur = parent[cur]

    path.reverse()

    return path, g[goal], len(visited)


def get_algorithm_route(algorithm="proposed", max_steps=6, alpha=0.5):
    """
    기존 알고리즘 / Dijkstra / A* 경로를 계산.

    비교 기준:
    - 기존 Proposed 경로를 먼저 계산
    - Proposed의 시작점과 마지막 도착점을 동일하게 사용
    - Dijkstra와 A*는 같은 start → goal 사이에서 경로만 다르게 탐색
    """
    proposed_path = compute_infection_path(zone_stats, zone_map, max_zones=max_steps)

    start_zone = proposed_path[0]
    goal_zone = proposed_path[-1]

    if algorithm == "proposed":
        path = proposed_path

        total_cost = 0
        for i in range(len(path) - 1):
            total_cost += move_cost(path[i + 1], alpha)

        visited_count = len(path)

    elif algorithm in ["dijkstra", "astar"]:
        path, total_cost, visited_count = search_path(
            start=start_zone,
            goal=goal_zone,
            algorithm=algorithm,
            alpha=alpha
        )

    else:
        raise ValueError("algorithm은 'proposed', 'dijkstra', 'astar' 중 하나여야 함")

    return {
        "algorithm": algorithm,
        "path": path,
        "start_zone": start_zone,
        "goal_zone": goal_zone,
        "total_cost": total_cost,
        "visited_count": visited_count,
        "path_length": max(0, len(path) - 1),
    }


# Heatmap 
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


def plot_severity_with_algorithm(algorithm="proposed", max_steps=6, alpha=0.5):
    """
    기존 '평균 중증도 및 예상 감염 경로' 히트맵과 같은 형태.
    단, path 계산 알고리즘만 Proposed / Dijkstra / A*로 변경.
    """
    result = get_algorithm_route(
        algorithm=algorithm,
        max_steps=max_steps,
        alpha=alpha
    )

    path = result["path"]

    if len(path) == 0:
        st.error("경로를 계산할 수 없습니다.")
        return

    start_zone = path[0]
    end_zone = path[-1]
    mid_zones = path[1:-1]

    # 기존 plot_severity_with_arrow와 동일하게 severity_mean 히트맵 사용
    heat = zone_map.pivot(index="row", columns="col", values="severity_mean")

    values = heat.values.astype(float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    fig, ax = plt.subplots(figsize=(4, 6))
    im = ax.imshow(
        heat.values,
        origin="upper",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax
    )

    title_dict = {
        "proposed": "평균 중증도 및 예상 감염 경로 - Proposed",
        "dijkstra": "평균 중증도 및 예상 감염 경로 - Dijkstra",
        "astar": "평균 중증도 및 예상 감염 경로 - A*",
    }

    ax.set_title(title_dict[algorithm])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["A", "B", "C"])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["1", "2", "3", "4"])
    ax.set_xlabel("열 (A–C)")
    ax.set_ylabel("행 (1–4)")

    # 기존과 동일하게 zone / 평균 중증도 / 단계 표시
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

    fig.colorbar(
        im,
        ax=ax,
        label="평균 중증도 (0=정상, 2=상)",
        fraction=0.035,
        pad=0.02,
    )

    # 시작 구역: 초록 테두리
    sr, sc = zone_map.loc[
        zone_map["zone"] == start_zone,
        ["row", "col"]
    ].values[0]

    ax.add_patch(
        plt.Rectangle(
            (sc - 0.5, sr - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#00ff00",
            linewidth=3,
        )
    )

    # 중간 경로: 노란 점선 테두리
    for z in mid_zones:
        rr, cc = zone_map.loc[
            zone_map["zone"] == z,
            ["row", "col"]
        ].values[0]

        ax.add_patch(
            plt.Rectangle(
                (cc - 0.5, rr - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#ffcc00",
                linewidth=2,
                linestyle="--",
            )
        )

    # 마지막 구역 : 빨간 테두리
    er, ec = zone_map.loc[
        zone_map["zone"] == end_zone,
        ["row", "col"]
    ].values[0]

    ax.add_patch(
        plt.Rectangle(
            (ec - 0.5, er - 0.5),
            1,
            1,
            fill=False,
            edgecolor="red",
            linewidth=3,
        )
    )

    # 화살표
    for i in range(len(path) - 1):
        z_from = path[i]
        z_to = path[i + 1]

        r1, c1 = zone_map.loc[
            zone_map["zone"] == z_from,
            ["row", "col"]
        ].values[0]

        r2, c2 = zone_map.loc[
            zone_map["zone"] == z_to,
            ["row", "col"]
        ].values[0]

        ax.annotate(
            "",
            xy=(c2, r2),
            xytext=(c1, r1),
            arrowprops=dict(arrowstyle="->", color="red", lw=2.5),
        )

    st.pyplot(fig)

    st.markdown(f"**알고리즘:** `{algorithm}`")
    st.markdown(f"🟢 **출발 구역:** `{start_zone}`")
    st.markdown(f"🔴 **마지막 예상 감염 구역:** `{end_zone}`")
    st.markdown(f"🟡 **예상 감염 경로:** {' → '.join(path)}")
    st.markdown(f"**총 이동 비용:** `{result['total_cost']:.3f}`")
    st.markdown(f"**경로 길이:** `{result['path_length']}`")
    st.markdown(f"**탐색 노드 수:** `{result['visited_count']}`")


def compare_algorithm_table(max_steps=6, alpha=0.5):
    """
    Proposed / Dijkstra / A* 결과를 표로 비교.
    """
    rows = []

    for alg in ["proposed", "dijkstra", "astar"]:
        result = get_algorithm_route(
            algorithm=alg,
            max_steps=max_steps,
            alpha=alpha
        )

        rows.append({
            "algorithm": alg,
            "path": " → ".join(result["path"]),
            "path_length": result["path_length"],
            "total_cost": result["total_cost"],
            "visited_count": result["visited_count"],
            "start_zone": result["start_zone"],
            "goal_zone": result["goal_zone"],
        })

    compare_df = pd.DataFrame(rows)
    st.dataframe(compare_df, use_container_width=True)



# 시각화 탭
st.markdown("### 4️⃣ 온실 병해·중증도 지도 및 감염 경로 알고리즘 비교")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "병해 발생 비율",
        "Proposed 경로",
        "Dijkstra 경로",
        "A* 경로",
        "알고리즘 비교표",
        "중증도 상(high) 비율",
    ]
)

with tab1:
    plot_heatmap_discrete(
        "disease_rate",
        "구역별 병해 발생 비율",
        label="병해율"
    )

with tab2:
    plot_severity_with_algorithm(
        algorithm="proposed",
        max_steps=path_len,
        alpha=route_alpha
    )

with tab3:
    plot_severity_with_algorithm(
        algorithm="dijkstra",
        max_steps=path_len,
        alpha=route_alpha
    )

with tab4:
    plot_severity_with_algorithm(
        algorithm="astar",
        max_steps=path_len,
        alpha=route_alpha
    )

with tab5:
    compare_algorithm_table(
        max_steps=path_len,
        alpha=route_alpha
    )

with tab6:
    plot_heatmap_discrete(
        "sev2_rate",
        "구역별 중증도 상(high) 비율",
        label="sev2"
    )



# 드론 방제 우선순위 Top-K
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


# 오늘의 방제 계획 요약
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
