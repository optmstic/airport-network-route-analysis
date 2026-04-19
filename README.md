# pds — Flight Routes with Up to Two Stops

Given two airports, list every route with at most two stops, rank them by total distance, and compare against the great-circle line. Aggregates across many pairs to surface the world's most-used transfer hubs.

**Course:** Programming for Data Science

---

## Data

Single public Kaggle dataset (no scraping):

| File | Rows | Notes |
|------|-----:|-------|
| `airports.csv` | 85,116 | 246 countries · **1,188** large airports |
| `routes.csv`   | 66,379 | 564 airlines · direct flights only |
| `airlines.csv` | 1,536  | 1,015 active |

**Caveat:** static snapshot, no dates. Answers are *paths in the network*, not flights per year.

**Quality check:** the `stops` column is `0` in 66,368 of 66,379 rows, so multi-stop routes are built by chaining direct flights. The `distance_km` column was validated against the haversine formula on 5,000 rows — mean diff **0.25 km**, max **0.50 km**.

---

## Method

1. **Filter** to large airports with IATA and scheduled service. They're only 1.4% of the file but carry **76%** of routes (both endpoints) and **96%** (at least one). Coverage: 219 countries, every inhabited continent. Network: **1,073 airports · 25,243 unique edges**.
2. **Search** all paths with 0, 1 or 2 intermediate nodes for a given pair; sort by total km.
3. **Compare** the best path against the great-circle distance — detour factor = `actual / great-circle`.
4. **Aggregate** stops used across many pairs to rank transfer hubs.
5. **Cost projection** (structural comparator, *not* a fare estimator, since the data has no prices):
   ```
   cost (€) = 50 + 0.10 × distance + 30 × stops
   ```
   A sensitivity variant (`0.01 × distance × stops` instead of a flat fee) shows that "stops weigh less on long-haul" is an artifact of the flat-fee choice, not a property of the data.

---

## Results

| # | Question | Answer |
|---|----------|--------|
| Q1 | Routes with ≤ 2 stops | LIS–CDG: **4,306** · LIS–SYD: **281** |
| Q2 | Detour from straight line | mean **21%** · median **6%** |
| Q3 | Top transfer hubs | CDG · IST · GRU · SVO · PEK · AMS · MEX |
| Q4 | Reachable in ≤ 2 stops | **82%** of pairs yes, **18%** no |
| Q5 | Cost of adding stops | LIS–CDG 197 € → 266 € (+35%) · MAD–PEK 970 € → 1,031 € (+6%) |
| Q6 | Extremes | Longest direct **SYD–DFW 13,808 km** · shortest pair forced to 2 stops **BLA–TAB 491 km** |

**Lisbon → Sydney** (18,176 km straight-line):
- East via Dubai — 18,182 km, 1 stop, **+0.03%** over the great circle.
- West via Americas/Pacific — 21,261 km, 2 stops, **+17%**.

The east-bound route wins — literally why flights from Europe to Australia go through the Gulf.

---

## Limits

- **Not ticket prices.** The cost model ranks options under a single transparent rule; it isn't a fare.
- **No dates, no schedules.** A path in the graph is not a bookable itinerary — no connection times, no interlining agreements.
- **Only large airports, only up to 2 stops.** "No route" means "no route within these constraints."
- **Great-circle ≠ flight track.** Real aircraft deviate for airspace, ETOPS, weather.

To turn this into a real planner you'd need: flight time (derivable from distance), minimum connection times, interlining rules, frequency, plus external data for prices (OAG/ITA/scraping).

---

## Layout

```
.
├── main.py           # exploration + analysis + plotting
├── data/             # Kaggle CSVs
├── plots/            # 8 output figures
└── README.md
```

---

## Run

Python 3.10+ with `pandas`, `numpy`, `matplotlib`.

```bash
pip install pandas numpy matplotlib
python main.py
```

Prints the analysis to stdout and writes figures to `plots/`.
