# Programming for Data Science — Flight Routes with Up to Two Stops

Given two airports, list every route with at most two stops, rank them by total distance, and compare against the great-circle line. Aggregates across many pairs to surface the world's most-used transfer hubs.

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

**The network is remarkably efficient.** Accepting up to two stops costs very little distance: the best 2-stop route is only **6% longer than the straight line** (median across 3,540 pairs tested). Half the time, the detour is essentially invisible.

**Seven airports carry global transfers.** Ranked by appearances as the middle stop on the shortest path: Paris CDG, Istanbul, São Paulo Guarulhos, Moscow Domodedovo, Beijing, Amsterdam, Mexico City. European hubs and the Gulf dominate intercontinental connections; US airports are in the top-degree list (JFK, ATL, EWR) but rarely act as international transfer points — they mostly feed domestic traffic.

**The world is 82% reachable in ≤ 2 stops.** Of 3,540 large-airport pairs sampled, 2,886 have a viable path; the remaining 654 involve peripheral regions that need three hops or more. "Two stops" is close to, but not quite, universal coverage.

**Route volume collapses with distance.** Lisbon–Paris has **4,306** routes up to 2 stops; Lisbon–Sydney has only **281**. Far-flung pairs are served by a thin set of corridors.

**Extra stops are cheap on long-haul, expensive on short-haul — partly a modelling artifact.** Under the flat-fee cost rule, Madrid–Beijing barely moves (970 € → 1,031 €, **+6%**) while Lisbon–Paris jumps sharply (197 € → 266 €, **+35%**). Switch the per-stop fee to a percentage of distance and the gap disappears (+19% vs +21%). The claim *"stops hurt less on long-haul"* is a property of the formula, not the geography.

**Network extremes.** Longest non-stop flight: **Sydney–Dallas, 13,808 km**. Shortest pair that *forces* two stops: Barcelona (Venezuela) → Porlamar → Port of Spain → Tobago, a total of **491 km** — the geography leaves no shorter option within the large-airport network.

### Case study: Lisbon → Sydney

Straight-line distance: **18,176 km**.

- East via Dubai — LIS → DXB → SYD, 18,182 km, 1 stop → **+0.03%** over the great circle.
- West via Montreal and San Francisco — LIS → YUL → SFO → SYD, 21,261 km, 2 stops → **+17%**.

The east-bound route wins by a huge margin — an extra 3,000 km and one extra stop the other way. This is literally why flights from Europe to Australia go through the Gulf and not across the Pacific.

---

## Limits

- **Not ticket prices.** The cost model ranks options under a single transparent rule; it isn't a fare.
- **No dates, no schedules.** A path in the graph is not a bookable itinerary — no connection times, no interlining agreements.
- **Only large airports, only up to 2 stops.** "No route" means "no route within these constraints."
- **Great-circle ≠ flight track.** Real aircraft deviate for airspace, ETOPS, weather.

To turn this into a real planner you'd need: flight time (derivable from distance), minimum connection times, interlining rules, frequency, plus external data for prices (OAG/ITA/scraping).

---

## Layout
.
├── data/             # Kaggle CSVs
├── plots/            # 8 output figures
└── main.py           # exploration + analysis + plotting

---

## Run

Python 3.10+ with `pandas`, `numpy`, `matplotlib`.

```bash
pip install pandas numpy matplotlib
python main.py
```

Prints the analysis to stdout and writes figures to `plots/`.
