# SeismicMap

SeismicMap is a Streamlit application for visualizing seismic design parameters (S<sub>MS</sub>, S<sub>M1</sub>, S<sub>DS</sub>, S<sub>D1</sub>) based on ASCE 7-16 and ASCE 7-22 data.

## Data

All input data is stored in the `data/` folder:

* `ASCE7-16_summarySA.csv` – Summary spectral acceleration values (ASCE 7-16)
* `ASCE7-22_summarySA.csv` – Summary spectral acceleration values (ASCE 7-22)
* `ASCE7_ratio.csv` – Ratio of 7-22 to 7-16 values
* `USA_map.png` – Base map of the continental United States
* `USA_map_transparent.png` – Transparent overlay mask for the U.S. map

## Installation

1. Clone or download the repository and navigate into its root directory.
2. Create and activate a Python virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Use the dropdown menus to select the ASCE code version (7‑16, 7‑22, or ratio) and the seismic parameter (S\_MS, S\_M1, S\_DS, S\_D1).

## Project Structure

```
├── app.py               # Streamlit application code
├── data/                # Folder containing input datasets and map images
│   ├── ASCE7-16_summarySA.csv
│   ├── ASCE7-22_summarySA.csv
│   ├── ASCE7_ratio.csv
├── README.md            # This file
└── requirements.txt     # Python package dependencies
```
