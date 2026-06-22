# Private Markets Portfolio Monitor

An interactive dashboard for fund-of-funds analytics: fund-level IRR/MOIC/DPI/RVPI/TVPI, portfolio exposure by vintage/strategy/geography, and a forward liquidity projection model.

**Live demo:** _(add your Streamlit Cloud URL here after deploying)_

## What it does

- Loads fund-level data (capital calls, distributions, NAV) from an Excel workbook
- Computes Net IRR (XIRR on irregular cash-flow dates), MOIC, DPI, RVPI, TVPI per fund
- Aggregates portfolio exposure by vintage year, strategy, and geography
- Computes HHI concentration and effective fund count
- Projects 5-year forward capital calls and distributions under adjustable assumptions
- Exports Power BI-ready CSV tables

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app loads the bundled `fund_data_input.xlsx` sample dataset by default, or you can upload your own file with the same four-sheet structure (`Funds`, `Capital_Calls`, `Distributions`, `NAV`).

## Stack

Python · pandas · numpy · scipy (XIRR solver) · Plotly · Streamlit · openpyxl
