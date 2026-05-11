# Airport Network Route Analysis

Academic project for **Programming for Data Science**.

This project analyses the global airport network as a graph and finds viable routes between airports with **up to two intermediate stops**. It ranks route alternatives by total distance, compares them with the great-circle baseline, and identifies the transfer hubs that appear most often in efficient paths.

---

## What this project does

- Builds a flight-route network from public airport, airline, and route datasets
- Filters the network to large airports with scheduled service and IATA codes
- Finds direct, one-stop, and two-stop paths between airport pairs
- Calculates route distance and detour against the great-circle distance
- Surfaces major global transfer hubs from sampled route pairs
- Produces exploratory plots and route case studies

---

## Data

The project uses a public Kaggle-style static snapshot:

| File | Description |
|---|---|
| `data/airports.csv` | Airport metadata, including location, country, type, and IATA code |
| `data/routes.csv` | Direct route links between airports |
| `data/airlines.csv` | Airline metadata and active status |

Important limitation: the data represents network links, not live flight schedules or bookable itineraries.

---

## Methodology

1. **Network preparation**  
   Keep large airports with scheduled service and valid IATA codes.

2. **Graph search**  
   Enumerate routes with 0, 1, or 2 intermediate stops.

3. **Distance comparison**  
   Compare each candidate route with the great-circle distance between origin and destination.

4. **Hub analysis**  
   Aggregate intermediate stops across sampled pairs to identify structurally important transfer airports.

5. **Cost sensitivity**  
   Apply a simple transparent route-cost formula to compare how stop penalties affect short-haul and long-haul trips.

---

## Key findings

- Many international airport pairs are reachable with at most two stops.
- Efficient routes are often only slightly longer than the great-circle baseline.
- A small number of airports account for a large share of efficient transfer paths.
- Long-haul routing patterns strongly favour established intercontinental hubs.
- The apparent cost of extra stops depends heavily on the chosen cost model.

---

## Repository structure

```text
.
├── data/                 # Airport, route, and airline datasets
├── plots/                # Generated visual outputs
├── main.py               # Main analysis script
└── README.md
```

---

## How to run

Requirements: Python 3.10+.

```bash
pip install pandas numpy matplotlib
python main.py
```

The script prints the analysis to the terminal and writes figures to `plots/`.

---

## Technologies

`Python` · `pandas` · `NumPy` · `Matplotlib` · `Graph Analysis` · `Geospatial Analysis`

---

## Notes

This is an academic network-analysis project. It does not estimate real ticket prices, travel times, minimum connection times, airline alliances, or schedule feasibility.
