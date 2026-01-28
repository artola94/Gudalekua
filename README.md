# Gudalekua: War Room Simulator

Gudalekua is a tactical simulation engine that models the evolution of combat fronts. Using historical data, geolocation metrics, and offensive pressure factors, it projects the probability of capturing strategic objectives (cities) under different intensity scenarios.

## Features

* **Multi-Scenario Simulation:** Three intensity levels for "what if" analysis:
    * **Conservative (x0.5):** High resistance, slowed advance.
    * **Inertial (x1.0):** Linear projection based on current trend.
    * **Aggressive (x1.5):** Defensive collapse scenario.
* **Advanced Metrics:**
    * `encirclement`: Factor (0-1) representing logistical isolation.
    * `dist_to_front`: Dynamic distance to the combat front.
    * `prob_capture`: Cumulative probability of city capture.
* **Temporal Resolution:** Day-by-day simulation up to 60 days.

## Installation

**Requirements:** Python 3.8+

1. Clone the repository:
   ```bash
   git clone https://github.com/artola94/Gudalekua.git
   cd Gudalekua
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Sources

This project requires external datasets not included in the repository due to size. Download them separately:

| Dataset | Source | Location |
|---------|--------|----------|
| NZZ Front Line Maps | [nzzdev/nzz-maps-ukraine-war](https://github.com/nzzdev/nzz-maps-ukraine-war) | `data/nzz-maps-master/` |
| Ukraine War Losses | [PetroIvaniuk/2022-Ukraine-Russia-War-Dataset](https://github.com/PetroIvaniuk/2022-Ukraine-Russia-War-Dataset) | `data/Russia-Ukraine-main/` |
| Ukrainian Cities | OpenStreetMap or similar | `data/ukranian_cities.geojson` |

### Generating the Dataset

Once source data is in place, run the pipeline (scripts can be run from any directory):

```bash
python scripts/process_data.py
python scripts/process_losses.py
python scripts/extract_physical.py
python scripts/merger.py
python scripts/wardatasetprepper.py
```

This generates `data/dataset_to_train.csv` required by the simulator.

## How It Works

The simulation evaluates each city daily based on three rules:

1. **Offensive Pressure:** Calculated from recent average advance (`delta_real`), adjusted by the scenario multiplier.
2. **Encirclement Factor:** When encirclement exceeds 90%, capture probability increases exponentially due to logistical isolation.
3. **Capture Condition:** A city is marked "OCCUPIED" when `prob_capture > 0.5`. The simulation stops and records the fall date.

### Example Output

Simulation for **Pokrovsk** (data: 2026-01-03):

```
>>> SIMULATING: Pokrovsk | Scenario: INERTIAL (x1.0) | 60 Days
!!! CITY FALLS ON DAY 18 !!!

--- RESULTS (Excerpt) ---
    day  dist_to_front  delta_real  encirclement  prob_capture  status
0     1              0        11.0         0.915         0.434   FREE
1     2              0        13.3         0.915         0.434   FREE
...
17   18              0        13.3         0.915         0.542   OCCUPIED
```

With 91.5% encirclement and constant pressure, the defense collapses on day 18.

## Usage

### Running the Simulator

```bash
python scripts/war_simulator.py
```

### Training Models

To retrain the ML models from scratch:

```bash
python scripts/cascade1.py  # Model A: Front dynamics
python scripts/cascade2.py  # Model B: Encirclement
python scripts/cascade3.py  # Model C: Capture probability
```

Models are saved to the `models/` directory.

## Repository Structure

```
Gudalekua/
├── data/                   # Data sources (not tracked)
├── models/                 # Trained ML models (not tracked)
├── scripts/
│   ├── war_simulator.py    # Main simulation engine
│   ├── process_data.py     # Geospatial processing
│   ├── process_losses.py   # Losses aggregation
│   ├── extract_physical.py # City attributes
│   ├── merger.py           # Data merge pipeline
│   ├── wardatasetprepper.py# Dataset preparation
│   ├── cascade1.py         # Model A training
│   ├── cascade2.py         # Model B training
│   └── cascade3.py         # Model C training
├── requirements.txt
├── LICENSE
└── README.md
```

## Roadmap

* Integration of seasonal weather factors
* Automatic comparison between predictions and historical data for model calibration

## License

MIT License - see [LICENSE](LICENSE) for details.
