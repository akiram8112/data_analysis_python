
# Monte Carlo Simulation for Incident Cost Forecasting

This Streamlit app allows users to perform **Monte Carlo simulations** to forecast incident-related costs based on historical incident and cost data from Excel spreadsheets.

It combines two types of simulations:
- **Total Projection** — Estimate future total costs for total data using average trends.
- **Multi-Year Summary** — Simulate and summarize cost ranges for each year in your dataset, with growth and volatility controls.

---

## 📁 File Requirements

The app expects an **Excel file** (`.xlsx` or `.xls`) with the following sheets:

### Sheet: `INCIDENTS`
| Column           | Required? | Description                    |
|------------------|-----------|--------------------------------|
| `Date of Injury` | ✅        | Date of each incident          |

### Sheet: `COSTS`
| Column           | Required? | Description                     |
|------------------|-----------|---------------------------------|
| `Date of Injury` | ✅        | Date associated with the cost   |
| `Incurred`       | ✅        | Total cost incurred (numeric)   |

Make sure:
- Dates are formatted correctly.
- Costs are numeric or formatted as currency.

---

## 🚀 Features

- Upload or input path to an Excel file
- Select simulation mode:
  - **Total Projection**
    - Adjustable event and cost volatility
    - Histogram of simulated total costs
  - **Multi-Year Summary**
    - Growth rate and volatility controls
    - Percentile-based summaries per year
    - Export results to `.csv` and `.xlsx`
    - Interactive Altair chart and histogram for latest year

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/monte-carlo-simulator.git
cd monte-carlo-simulator
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```txt
streamlit
pandas
numpy
matplotlib
altair
openpyxl
```

### 3. Run the App
```bash
streamlit run app.py
```

---

## 📂 Outputs

If enabled in Multi-Year mode, simulation results are saved in:
```
simulation_output/
├── monte_carlo_summary.csv
└── monte_carlo_summary.xlsx
```

---

## ⚙️ Controls Overview

| Control                  | Description                                         |
|--------------------------|-----------------------------------------------------|
| File Upload or Path      | Choose whether to upload a file or enter a file path |
| Simulation Mode          | Choose between Total or Multi-Year analysis  |
| Volatility Sliders       | Adjust randomness for events and costs             |
| Growth Rate (Multi-Year) | Annual % increase in incident count                |
| Export Option            | Save results as Excel and CSV                      |

---

## 📊 Sample Outputs

- 📈 Line chart of simulation percentiles by year
- 📉 Histogram of simulated total costs
- 🧮 Summary statistics (5th, 50th, 95th percentiles, mean, max)

---

## 🛠️ Troubleshooting

- Ensure the Excel file contains both required sheets: `INCIDENTS` and `COSTS`.
- Costs must be numeric. If not, try cleaning or reformatting the spreadsheet.
- If you're using file paths, ensure they're correct and accessible from your system.

---

## 📝 License

MIT License. Use freely with attribution.

---

## 🤝 Contributing

Contributions, bug reports, and feature suggestions are welcome! Open an issue or submit a pull request.
