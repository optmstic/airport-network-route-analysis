# Combined script: exploration + analysis + plots.
# Run with: python main.py

import os
from collections import defaultdict, Counter
from math import radians, sin, cos, asin, sqrt, atan2, degrees

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SETUP
# ============================================================

DATA_DIR = "data"
PLOTS_DIR = "plots"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

# Three CSVs from a single Kaggle dataset. keep_default_na=False prevents
# pandas from reading the string "NA" (North America) as a missing value.
datasets = {}
for name in ["airports", "routes", "airlines"]:
    path = os.path.join(DATA_DIR, f"{name}.csv")
    datasets[name] = pd.read_csv(path, low_memory=False,
                                 keep_default_na=False, na_values=[""])

airports = datasets["airports"]
routes = datasets["routes"]
airlines = datasets["airlines"]


# ============================================================
# 3. EXPLORATION: quick look at each table
# ============================================================

# Shape + columns for each table.
for name, df in datasets.items():
    print("=" * 70)
    print(name.upper())
    print("=" * 70)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print()

# Value counts on the columns we care about.
value_count_cols = [
    (airports, "type",              "Airport types"),
    (airports, "scheduled_service", "Scheduled service"),
    (airports, "continent",         "Continent"),
    (routes,   "stops",             "Stops per route"),
    (airlines, "active",            "Airlines active flag"),
]
for df, col, title in value_count_cols:
    print(f"\n-- {title} --")
    print(df[col].value_counts(dropna=False))

# How many distinct values in each key ID column.
unique_cols = [
    (airports, "iso_country", "Countries in airports"),
    (routes,   "airline_iata", "Airlines in routes"),
    (routes,   "src_iata",     "Source airports in routes"),
    (routes,   "dst_iata",     "Destination airports in routes"),
]
print("\n-- Distinct values --")
for df, col, label in unique_cols:
    print(f"  {label}: {df[col].nunique()}")
print(f"\nAirports with IATA code: {airports['iata_code'].notna().sum()}")

print("\n-- Distance (km) summary --")
print(routes["distance_km"].describe())

print("\n-- Missing values per column (routes) --")
print(routes.isna().sum())

# Top 10 countries by airport count (full dataset) and by large airports.
print("\n-- Top 10 countries by total airports --")
print(airports["iso_country"].value_counts().head(10))
print("\n-- Top 10 countries by LARGE airports --")
print(airports[airports["type"] == "large_airport"]
      ["iso_country"].value_counts().head(10))

# Top 10 airlines by number of routes operated (gives a sense of which
# carriers dominate the edge list before we collapse duplicates).
print("\n-- Top 10 airlines by route count --")
print(routes["airline_iata"].value_counts().head(10))

# How many distinct carriers operate each (src, dst)? Proxy for codeshare
# density: high values mean the route is flown by many airlines.
codeshare = (routes.dropna(subset=["src_iata", "dst_iata"])
             .groupby(["src_iata", "dst_iata"])["airline_iata"]
             .nunique())
print(f"\nAirlines per route: mean {codeshare.mean():.2f}, "
      f"max {codeshare.max()}, "
      f"routes with >=5 carriers: {(codeshare >= 5).sum()}")


# ============================================================
# 4. SCOPE DECISION: keep only LARGE airports
# ============================================================
# Large airports are 1.4% of the full table but cover almost all commercial
# traffic. Restricting the network to them keeps it small and readable.

large_iata = set(airports[airports["type"] == "large_airport"]["iata_code"].dropna())
print("\n" + "=" * 70)
print("LARGE-AIRPORT COVERAGE")
print("=" * 70)
print(f"Large airports with IATA code: {len(large_iata)}")

# Coverage check: how many route rows touch a large airport.
r_valid = routes.dropna(subset=["src_iata", "dst_iata"])
r_valid = r_valid[r_valid["distance_km"] > 0]
src_large = r_valid["src_iata"].isin(large_iata)
dst_large = r_valid["dst_iata"].isin(large_iata)
for label, mask in [("both endpoints large", src_large & dst_large),
                    ("at least one large",   src_large | dst_large)]:
    print(f"  {label}: {mask.sum()} ({mask.mean()*100:.1f}%)")


# ============================================================
# 5. CLEAN ROUTES AND BUILD THE NETWORK
# ============================================================

# Keep only routes with valid coordinates, positive distance, and both
# endpoints classified as large airports.
needed_cols = ["src_iata", "dst_iata", "src_lat", "src_lon", "dst_lat", "dst_lon"]
r = routes.dropna(subset=needed_cols).copy()
r = r[r["distance_km"] > 0]
r = r[r["src_iata"].isin(large_iata) & r["dst_iata"].isin(large_iata)]
print(f"\nRoute rows after filtering: {len(r)}")

# Edge list: one row per (src, dst), shortest reported distance.
edges = r.groupby(["src_iata", "dst_iata"], as_index=False)["distance_km"].min()
print(f"Unique direct connections: {len(edges)}")

# Adjacency list for path-finding.
graph = defaultdict(list)
for row in edges.itertuples(index=False):
    graph[row.src_iata].append((row.dst_iata, row.distance_km))

# Coordinates lookup per airport.
coords = {}
for row in r.itertuples(index=False):
    coords[row.src_iata] = (row.src_lat, row.src_lon)
    coords[row.dst_iata] = (row.dst_lat, row.dst_lon)

# Airport metadata lookup (IATA -> name, country). Built once to avoid
# scanning the 85k-row airports table on every print.
airport_lookup = {}
for iata, name, country in zip(airports["iata_code"],
                               airports["name"],
                               airports["iso_country"]):
    if pd.notna(iata):
        ascii_name = str(name).encode("ascii", "ignore").decode()
        airport_lookup[iata] = (ascii_name, country)


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    """Straight-line distance (km) between two lat/lon points on Earth."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def great_circle_points(lat1, lon1, lat2, lon2, n=100):
    """N points along the shortest path on the globe between A and B.
    Used to draw curved lines on the flat map."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    d = 2 * asin(sqrt(sin((lat2 - lat1) / 2) ** 2
                      + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2))
    if d == 0:
        return [(degrees(lat1), degrees(lon1))]
    pts = []
    for f in np.linspace(0, 1, n):
        A = sin((1 - f) * d) / sin(d)
        B = sin(f * d) / sin(d)
        x = A * cos(lat1) * cos(lon1) + B * cos(lat2) * cos(lon2)
        y = A * cos(lat1) * sin(lon1) + B * cos(lat2) * sin(lon2)
        z = A * sin(lat1) + B * sin(lat2)
        pts.append((degrees(atan2(z, sqrt(x * x + y * y))),
                    degrees(atan2(y, x))))
    return pts


def paths_upto_2_stops(src, dst):
    """All simple paths from src to dst with 0, 1 or 2 stops.
    Returns list of (path_as_airport_list, total_distance_km)."""
    results = []
    # 0 stops: direct flight.
    for nxt, d in graph.get(src, []):
        if nxt == dst:
            results.append(([src, dst], d))
    # 1 stop.
    for mid, d1 in graph.get(src, []):
        if mid in (dst, src):
            continue
        for nxt, d2 in graph.get(mid, []):
            if nxt == dst:
                results.append(([src, mid, dst], d1 + d2))
    # 2 stops.
    for mid1, d1 in graph.get(src, []):
        if mid1 in (dst, src):
            continue
        for mid2, d2 in graph.get(mid1, []):
            if mid2 in (dst, src, mid1):
                continue
            for nxt, d3 in graph.get(mid2, []):
                if nxt == dst:
                    results.append(([src, mid1, mid2, dst], d1 + d2 + d3))
    return results


# Cost projection models. The dataset has no prices, no dates, no fare
# classes and no seat inventory, so a real fare estimator is not possible.
# Real ticket prices depend on booking window, season, demand, promotions,
# airline, cabin class and O&D pricing rules — none of which are in the
# data. Building a pseudo-realistic model on top of absent ground truth
# would stack fiction on fiction, so we deliberately keep this simple.
#
# What this IS: a *structural comparator*. Under a single transparent
# rule, it lets us rank options for the same pair and ask "does accepting
# an extra stop look worth it in the shape of the network?". The absolute
# numbers are not fares — only the relative ordering between options for
# the same pair is meaningful.
#
# The only enrichment that would stay honest given this dataset is
# deriving flight *time* from distance and a typical cruise speed
# (~850 km/h plus a fixed taxi/climb overhead), since time is a function
# of the geography we actually have. Prices are not.
BASE_FARE = 50.0        # fixed per trip (fees, booking)
RATE_PER_KM = 0.10      # variable (fuel, crew, aircraft)
STOP_COST = 30.0        # flat euros per intermediate stop (handling, transfer)

# Proportional: per-stop penalty grows with the trip length, so long-haul
# does not get "free" stops in the accounting. Used as a sensitivity check.
STOP_PCT_PER_KM = 0.01  # 1% of distance added per stop


def estimate_cost(distance_km, num_stops):
    """Linear structural comparator in euros. Not a fare estimate."""
    return BASE_FARE + RATE_PER_KM * distance_km + STOP_COST * num_stops


def estimate_cost_nl(distance_km, num_stops):
    """Non-linear model: per-stop penalty scales with distance."""
    return (BASE_FARE + RATE_PER_KM * distance_km
            + STOP_PCT_PER_KM * distance_km * num_stops)


# ============================================================
# 7. ANALYSIS: specific airport pairs (distance + cost)
# ============================================================

test_pairs = [
    ("LIS", "HND"),   # Lisbon -> Tokyo Haneda
    ("LIS", "JFK"),   # Lisbon -> New York
    ("OPO", "GRU"),   # Porto -> Sao Paulo
    ("LIS", "SYD"),   # Lisbon -> Sydney
    ("MAD", "PEK"),   # Madrid -> Beijing
    ("LHR", "AKL"),   # London -> Auckland
    ("LIS", "CDG"),   # Lisbon -> Paris
]

print("\n" + "=" * 70)
print("ANALYSIS OF SPECIFIC PAIRS (distance + cost projection)")
print("=" * 70)
print(f"Cost model: {BASE_FARE:.0f} EUR + {RATE_PER_KM:.2f} EUR/km "
      f"+ {STOP_COST:.0f} EUR/stop")

pair_summaries = []

for src, dst in test_pairs:
    if src not in coords or dst not in coords:
        print(f"\n{src} -> {dst}: airport not in network")
        continue

    gc = haversine(*coords[src], *coords[dst])
    paths = paths_upto_2_stops(src, dst)

    # Single pass: count paths and track best (shortest) at each stop level.
    level_counts = {0: 0, 1: 0, 2: 0}
    best_per_level = {}  # level -> (path, distance_km, cost_eur)
    for path, dist in paths:
        lv = len(path) - 2
        level_counts[lv] += 1
        if lv not in best_per_level or dist < best_per_level[lv][1]:
            best_per_level[lv] = (path, dist, estimate_cost(dist, lv))

    print(f"\n{src} -> {dst}  (straight line {gc:.0f} km)")
    for lv in [0, 1, 2]:
        if lv in best_per_level:
            path, dist, cost = best_per_level[lv]
            print(f"  {lv}-stop: {level_counts[lv]:>5} option(s) | "
                  f"best {dist:5.0f} km  ~EUR {cost:5.0f}  via "
                  f"{' -> '.join(path)}")
        else:
            print(f"  {lv}-stop: no path available")

    pair_summaries.append({
        "src": src, "dst": dst, "gc": gc,
        "best_per_level": best_per_level,
    })


# ============================================================
# 7a. SENSITIVITY: linear vs proportional stop cost
# ============================================================
# The claim "stops weigh less on long-haul" is partly an artefact of the
# flat-fee-per-stop model. Here we compare it to a model where the stop
# penalty scales with distance, to see if the conclusion survives.

print("\n" + "=" * 70)
print("COST-MODEL SENSITIVITY (linear vs proportional per-stop)")
print("=" * 70)
print(f"Linear:       {BASE_FARE:.0f} + {RATE_PER_KM:.2f}/km + {STOP_COST:.0f}/stop")
print(f"Proportional: {BASE_FARE:.0f} + {RATE_PER_KM:.2f}/km "
      f"+ {STOP_PCT_PER_KM*100:.0f}% of km per stop")

# Only for the two reference pairs used in the presentation.
for src, dst in [("LIS", "CDG"), ("MAD", "PEK")]:
    summary = next(s for s in pair_summaries
                   if s["src"] == src and s["dst"] == dst)
    direct = summary["best_per_level"].get(0)
    two_stop = summary["best_per_level"].get(2)
    if not direct or not two_stop:
        continue
    _, d0, c0_lin = direct
    _, d2, c2_lin = two_stop
    c0_nl = estimate_cost_nl(d0, 0)
    c2_nl = estimate_cost_nl(d2, 2)
    pct_lin = (c2_lin / c0_lin - 1) * 100
    pct_nl = (c2_nl / c0_nl - 1) * 100
    print(f"\n{src} -> {dst}")
    print(f"  Direct  {d0:5.0f} km | linear EUR {c0_lin:6.0f} | "
          f"proportional EUR {c0_nl:6.0f}")
    print(f"  2-stop  {d2:5.0f} km | linear EUR {c2_lin:6.0f} ({pct_lin:+.0f}%) | "
          f"proportional EUR {c2_nl:6.0f} ({pct_nl:+.0f}%)")


# ============================================================
# 7b. ANALYSIS: LIS -> SYD eastbound vs westbound
# ============================================================
# Eastbound = goes via Europe/Asia (intermediate stops between LIS and SYD
# in longitude). Westbound = goes via the Americas/Pacific (at least one
# intermediate stop with longitude in the American range).

print("\n" + "=" * 70)
print("LIS -> SYD: EASTBOUND vs WESTBOUND (<=2 stops)")
print("=" * 70)

src_dir, dst_dir = "LIS", "SYD"
all_lis_syd = paths_upto_2_stops(src_dir, dst_dir)


def in_americas(code):
    """True if the airport sits in the American longitude range (-170, -30)."""
    _, lon = coords[code]
    return -170 <= lon <= -30


direction_paths = {"east": [], "west": []}
for path, dist in all_lis_syd:
    stops = path[1:-1]
    bucket = "west" if any(in_americas(s) for s in stops) else "east"
    direction_paths[bucket].append((path, dist))

for bucket in ["east", "west"]:
    bucket_paths = direction_paths[bucket]
    print(f"\n{bucket.capitalize()}bound: {len(bucket_paths)} option(s)")
    if not bucket_paths:
        print("  (no viable path in this direction with <=2 stops)")
        continue
    best_path, best_dist = min(bucket_paths, key=lambda x: x[1])
    n_stops = len(best_path) - 2
    cost = estimate_cost(best_dist, n_stops)
    print(f"  Best: {best_dist:.0f} km, {n_stops} stop(s), ~EUR {cost:.0f}")
    print(f"  Route: {' -> '.join(best_path)}")


# ============================================================
# 8. ANALYSIS: random-sample coverage, detours and hubs
# ============================================================

print("\n" + "=" * 70)
print("RANDOM-SAMPLE ANALYSIS")
print("=" * 70)

# 60 random large airports that are actually in the network.
sched = airports[(airports["scheduled_service"] == "yes")
                 & (airports["iata_code"].notna())
                 & (airports["type"] == "large_airport")
                 ]["iata_code"].tolist()
sched_in_graph = [a for a in sched if a in graph]

rng = np.random.default_rng(42)
sample = rng.choice(sched_in_graph, size=60, replace=False)

# For each ordered pair in the sample: best path, detour factor, hubs used.
detours = []
hub_counter = Counter()
with_route = 0
no_route = 0
for src in sample:
    for dst in sample:
        if src == dst:
            continue
        paths = paths_upto_2_stops(src, dst)
        if not paths:
            no_route += 1
            continue
        with_route += 1
        best_path, best_dist = min(paths, key=lambda x: x[1])
        gc = haversine(*coords[src], *coords[dst])
        # Skip tiny distances where the ratio becomes meaningless.
        if gc > 100:
            detours.append(best_dist / gc)
        # Intermediate nodes = everything except endpoints.
        for hub in best_path[1:-1]:
            hub_counter[hub] += 1

print(f"Pairs tested: {len(sample) * (len(sample) - 1)}")
print(f"  with route <=2 stops: {with_route}")
print(f"  without route:        {no_route}")
if detours:
    print(f"Detour factor  mean: {np.mean(detours):.3f}  "
          f"median: {np.median(detours):.3f}")
else:
    print("Detour factor: no pairs with distance > 100 km")

print("\nTop 15 hubs used as intermediate stop on the best path:")
for hub, count in hub_counter.most_common(15):
    name, country = airport_lookup.get(hub, ("?", "?"))
    print(f"  {hub}  {count:4d}  {name} ({country})")


# ============================================================
# 9. ANALYSIS: top hubs by degree and distance buckets
# ============================================================

# Degree = number of unique direct connections (in + out).
out_deg = edges.groupby("src_iata").size()
in_deg = edges.groupby("dst_iata").size()
total_deg = out_deg.add(in_deg, fill_value=0).sort_values(ascending=False)

print("\n" + "=" * 70)
print("TOP 15 AIRPORTS BY DIRECT-ROUTE DEGREE (in + out)")
print("=" * 70)
for iata, deg in total_deg.head(15).items():
    name, country = airport_lookup.get(iata, ("?", "?"))
    print(f"  {iata}  {int(deg):4d}  {name} ({country})")

# Distance buckets: count routes in each distance range.
buckets = [
    (0,     500,   "very short (<500 km)"),
    (500,   1500,  "short (500-1500 km)"),
    (1500,  4000,  "medium (1500-4000 km)"),
    (4000,  99999, "long (>4000 km)"),
]
print("\n-- Distance buckets over unique direct connections --")
for low, high, label in buckets:
    n = ((edges["distance_km"] >= low) & (edges["distance_km"] < high)).sum()
    print(f"  {label}: {n}")

# ---- Extremes: longest direct flight and shortest forced-2-stop pair ----

# Longest single-segment direct connection in the whole large-airport network.
longest_direct = edges.loc[edges["distance_km"].idxmax()]
ld_src, ld_dst = longest_direct["src_iata"], longest_direct["dst_iata"]
print("\n-- Longest direct flight (no stops) --")
print(f"  {ld_src} -> {ld_dst}  {longest_direct['distance_km']:.0f} km")
print(f"  {ld_src}: {airport_lookup.get(ld_src, ('?', '?'))[0]}")
print(f"  {ld_dst}: {airport_lookup.get(ld_dst, ('?', '?'))[0]}")

# Shortest pair that NEEDS 2 stops: no direct edge and no 1-stop connection.
# Precompute reachability within 0 or 1 stops, then scan 2-stop totals.
reachable_01 = {}
nodes = list(graph.keys())
for src in nodes:
    seen = set()
    for mid, _ in graph.get(src, []):
        seen.add(mid)
        for nxt, _ in graph.get(mid, []):
            if nxt != src:
                seen.add(nxt)
    reachable_01[src] = seen

min_total = float("inf")
min_path = None
for src in nodes:
    src_reach = reachable_01[src]
    for mid1, d1 in graph.get(src, []):
        if mid1 == src:
            continue
        for mid2, d2 in graph.get(mid1, []):
            if mid2 in (src, mid1):
                continue
            for dst, d3 in graph.get(mid2, []):
                if dst in (src, mid1, mid2):
                    continue
                # Only keep pairs where 2 stops is the minimum needed.
                if dst in src_reach:
                    continue
                total = d1 + d2 + d3
                if total < min_total:
                    min_total = total
                    min_path = [src, mid1, mid2, dst]

print("\n-- Shortest route that REQUIRES 2 stops --")
if min_path:
    print(f"  {' -> '.join(min_path)}  total {min_total:.0f} km")
    for code in min_path:
        name, country = airport_lookup.get(code, ("?", "?"))
        print(f"    {code}: {name} ({country})")
else:
    print("  (no such pair found)")


# Sanity check: does our Haversine match the distance_km column? Sampled
# from r_valid (all routes with coords and positive distance) rather than
# r (large-airport subset), so this validates the column more broadly.
sample_rows = r_valid.sample(5000, random_state=0)
recalc = np.array([haversine(a, b, c, d) for a, b, c, d
                   in zip(sample_rows.src_lat, sample_rows.src_lon,
                          sample_rows.dst_lat, sample_rows.dst_lon)])
diff = np.abs(recalc - sample_rows["distance_km"].values)
print(f"\nDistance check on 5,000 rows -> mean diff {diff.mean():.2f} km, "
      f"max {diff.max():.2f} km")


# ============================================================
# 10. PLOTS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

# Large airports with IATA — used as background dots on the maps.
large_df = airports[(airports["type"] == "large_airport")
                    & (airports["iata_code"].notna())]


# --- Plot 1: distance distribution histogram -------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(edges["distance_km"], bins=60, color="#3b7dd8", edgecolor="white")
for value, color, lbl in [(edges["distance_km"].median(), "red",    "Median"),
                          (edges["distance_km"].mean(),   "orange", "Mean")]:
    ax.axvline(value, color=color, linestyle="--",
               label=f"{lbl} {value:.0f} km")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Number of direct routes")
ax.set_title("Distribution of direct-route distances (large airports)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "01_distance_distribution.png"), dpi=130)
plt.close(fig)


# --- Plot 2: top 15 hubs by degree -----------------------------------------
top15 = total_deg.head(15)
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top15.index[::-1], top15.values[::-1], color="#2a9d8f")
for i, v in enumerate(top15.values[::-1]):
    ax.text(v + 3, i, str(int(v)), va="center", fontsize=8)
ax.set_xlabel("Direct routes (in + out)")
ax.set_title("Top 15 airports by number of direct routes")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "02_top_hubs_by_degree.png"), dpi=130)
plt.close(fig)


# --- Plot 3: airport types breakdown ---------------------------------------
types = airports["type"].value_counts()
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(types.index, types.values, color="#e76f51")
for i, v in enumerate(types.values):
    ax.text(i, v + 400, f"{v:,}", ha="center", fontsize=8)
ax.set_ylabel("Number of airports")
ax.set_title("Airports by type (full dataset)")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "03_airport_types.png"), dpi=130)
plt.close(fig)


# --- Plot 4: world map of large airports by continent ----------------------
palette = {"EU": "#1f77b4", "AS": "#ff7f0e", "NA": "#2ca02c",
           "SA": "#d62728", "AF": "#9467bd", "OC": "#8c564b", "AN": "#7f7f7f"}
fig, ax = plt.subplots(figsize=(12, 6))
for cont, color in palette.items():
    sub = large_df[large_df["continent"] == cont]
    ax.scatter(sub["longitude_deg"], sub["latitude_deg"],
               s=5, c=color, label=cont, alpha=0.7)
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Large airports by continent")
ax.legend(markerscale=2, loc="lower left", fontsize=8)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "04_airports_world_map.png"), dpi=130)
plt.close(fig)


# --- Plot 5: example route LIS -> SYD on the world map ---------------------
# Reuse the best path already computed for LIS-SYD in section 7 instead of
# recalculating all paths here.
SRC, DST = "LIS", "SYD"
lis_syd = next(s for s in pair_summaries if s["src"] == SRC and s["dst"] == DST)
best_path, best_dist, _ = min(lis_syd["best_per_level"].values(),
                              key=lambda x: x[1])
gc_dist = lis_syd["gc"]

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(large_df["longitude_deg"], large_df["latitude_deg"],
           s=2, c="lightgrey", alpha=0.5)

# Red dashed: great-circle between endpoints.
gc_lats, gc_lons = zip(*great_circle_points(*coords[SRC], *coords[DST]))
ax.plot(gc_lons, gc_lats, "r--", linewidth=1.5,
        label=f"Straight line ({gc_dist:.0f} km)")

# Blue solid: each leg of the best path, drawn as its own arc.
for a, b in zip(best_path[:-1], best_path[1:]):
    seg_lats, seg_lons = zip(*great_circle_points(*coords[a], *coords[b]))
    ax.plot(seg_lons, seg_lats, "b-", linewidth=2)

# Label every stop.
for code in best_path:
    lat, lon = coords[code]
    ax.plot(lon, lat, "ko", markersize=6)
    ax.annotate(code, (lon, lat), xytext=(5, 5),
                textcoords="offset points", fontsize=10, fontweight="bold")

ax.plot([], [], "b-", linewidth=2,
        label=f"Best path {' -> '.join(best_path)} ({best_dist:.0f} km)")
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"{SRC} -> {DST}: straight line vs best path with up to 2 stops")
ax.legend(loc="lower left", fontsize=9)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "05_route_lis_syd.png"), dpi=130)
plt.close(fig)


# --- Plot 6: detour-factor distribution ------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(detours, bins=40, range=(1, 3), color="#6a4c93", edgecolor="white")
for value, color, lbl in [(np.median(detours), "red",    "Median"),
                          (np.mean(detours),   "orange", "Mean")]:
    ax.axvline(value, color=color, linestyle="--",
               label=f"{lbl} {value:.2f}")
ax.set_xlabel("Detour factor (real path / straight line)")
ax.set_ylabel("Number of airport pairs")
ax.set_title("How much longer is the best 2-stop path vs the straight line?")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "06_detour_distribution.png"), dpi=130)
plt.close(fig)


# --- Plot 7: top transfer hubs from the random-pair experiment -------------
top_hubs = hub_counter.most_common(15)
hub_labels = [h for h, _ in top_hubs]
hub_values = [c for _, c in top_hubs]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(hub_labels[::-1], hub_values[::-1], color="#f4a261")
for i, v in enumerate(hub_values[::-1]):
    ax.text(v + 1, i, str(v), va="center", fontsize=8)
ax.set_xlabel("Times used as intermediate stop (best path)")
ax.set_title("Top 15 transfer hubs on the shortest 2-stop paths")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "07_top_transfer_hubs.png"), dpi=130)
plt.close(fig)


# --- Plot 8: projected cost by stop level for each test pair ---------------
# Grouped bars: for each pair, cost of best path with 0, 1, 2 stops.
# Missing bars = no path exists at that stop level.
pair_labels = [f"{s['src']}-{s['dst']}" for s in pair_summaries]
x = np.arange(len(pair_labels))
width = 0.27
colors = {0: "#2a9d8f", 1: "#e9c46a", 2: "#e76f51"}

fig, ax = plt.subplots(figsize=(12, 6))
for i, lv in enumerate([0, 1, 2]):
    costs = [s["best_per_level"][lv][2] if lv in s["best_per_level"] else 0
             for s in pair_summaries]
    bars = ax.bar(x + (i - 1) * width, costs, width,
                  label=f"{lv} stop(s)", color=colors[lv])
    for b, c in zip(bars, costs):
        if c > 0:
            ax.text(b.get_x() + b.get_width() / 2, c + 20,
                    f"EUR {c:.0f}", ha="center", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(pair_labels, rotation=20)
ax.set_ylabel("Projected cost (EUR)")
ax.set_title(f"Cost projection by stop level "
             f"(EUR {BASE_FARE:.0f} + {RATE_PER_KM:.2f}/km + {STOP_COST:.0f}/stop)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "08_cost_by_pair.png"), dpi=130)
plt.close(fig)

print(f"All 8 plots saved in ./{PLOTS_DIR}/")
