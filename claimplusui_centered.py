# --- Import necessary libraries ---
import streamlit as st # Streamlit for building web apps
import pandas as pd # Pandas for data manipulation
import numpy as np # NumPy for numerical computations
import matplotlib.pyplot as plt # Matplotlib for plotting histograms
import altair as alt # Altair for interactive charts
import os # OS module for file path checking and directory operations

# -----------------------------
# --- Shared Utility Functions
# -----------------------------
# Load incident and cost data from Excel file
def load_data(file, from_upload=True, outcome_factor="All Events"):
    try:
         # Read Excel sheets depending on whether uploaded or from path
        if from_upload:
            incidents_df = pd.read_excel(file, sheet_name='INCIDENTS')
            costs_df = pd.read_excel(file, sheet_name='COSTS')
            expected_claims_columns = {
        'I': 'Primary Mechanism',
        'J': 'Secondary Mechanism',
        'E': 'Primary Nature of Injury',
        'F': 'Secondary Nature of Injury',
        'G': 'Primary Bodily Location',
        'H': 'Secondary Bodily Location'
         }
    # Only rename if needed (your Excel file may already have readable headers)
        for col_letter, new_name in expected_claims_columns.items():
            if col_letter in costs_df.columns:
                costs_df.rename(columns={col_letter: new_name}, inplace=True)
                ##################
            
        else:
            incidents_df = pd.read_excel(file, sheet_name='INCIDENTS', engine='openpyxl')
            costs_df = pd.read_excel(file, sheet_name='COSTS', engine='openpyxl')
        # Convert date strings to datetime objects
        incidents_df['Date of Injury'] = pd.to_datetime(incidents_df['Date of Injury'], errors='coerce')
        costs_df['Date of Injury'] = pd.to_datetime(costs_df['Date of Injury'], errors='coerce')
        # Convert cost strings to numeric
        costs_df['Incurred'] = pd.to_numeric(costs_df['Incurred'].replace('[^\d.]', '', regex=True), errors='coerce')
        # Extract year from date
        source_df = costs_df if outcome_factor == "Claims" else incidents_df
        source_df["Date of Injury"] = pd.to_datetime(source_df["Date of Injury"], errors='coerce')
        source_df["Year"] = source_df["Date of Injury"].dt.year
        # incidents_df['Year'] = incidents_df['Date of Injury'].dt.year
        costs_df['Year'] = costs_df['Date of Injury'].dt.year
         # If Outcome Factor is "Claims", use costs_df for all injury date references
        if outcome_factor == "Claims":
            return costs_df.dropna(subset=['Date of Injury', 'Incurred']), costs_df.dropna(subset=['Date of Injury', 'Incurred'])
        else:
            return incidents_df.dropna(subset=['Date of Injury']), costs_df.dropna(subset=['Date of Injury', 'Incurred'])

    except Exception as e:
        st.error(f"Error loading data: {e}") # Display error in Streamlit
        return None, None
# Calculate average events, cost per event, and std deviation
def calculate_statistics(incidents_df, costs_df, outcome_factor="All Events"):
     # Count incidents per year
    if outcome_factor == "Claims":
        # Count claims from COSTS sheet
        incident_counts = costs_df['Date of Injury'].dt.year.value_counts().sort_index()
        incident_counts_df = incident_counts.rename_axis('Year').reset_index(name='Incident Count')
    else:
        # Count all events from INCIDENTS sheet
        incident_counts = incidents_df['Date of Injury'].dt.year.value_counts().sort_index()
        incident_counts_df = incident_counts.rename_axis('Year').reset_index(name='Incident Count')
    # Summarize total cost and average cost per incident
    cost_summary = costs_df.groupby('Year').agg(
        Total_Cost=('Incurred', 'sum'),
        Incident_Count=('Date of Injury', 'count')
    ).reset_index()
    cost_summary['Avg_Cost_Per_Incident'] = cost_summary['Total_Cost'] / cost_summary['Incident_Count']
    # Merge incident and cost summaries
    merged_df = pd.merge(incident_counts_df, cost_summary[['Year', 'Avg_Cost_Per_Incident']], on='Year', how='left')
    # Calculate averages and standard deviation
    avg_events_per_year = merged_df['Incident Count'].mean()
    avg_cost_per_event = merged_df['Avg_Cost_Per_Incident'].mean()
    std_dev_cost = merged_df['Avg_Cost_Per_Incident'].std()
    
    return avg_events_per_year, avg_cost_per_event, std_dev_cost

# Run Monte Carlo simulation to model total cost variability
def run_monte_carlo_simulation(avg_events, avg_cost, std_dev_cost, 
                               event_volatility=1.0, cost_volatility=1.0, 
                               num_simulations=10000):
     # Simulate number of events
    events = np.random.normal(avg_events * event_volatility, 
                              avg_events * 0.2 * event_volatility, 
                              num_simulations)
    events = np.maximum(events, 0) # Prevent negative events

    # Simulate cost per event
    costs = np.random.normal(avg_cost * cost_volatility, 
                             std_dev_cost * cost_volatility, 
                             num_simulations)
    costs = np.maximum(costs, 0) # Prevent negative costs

    return events * costs  # Total simulated costs

# Plot histogram of simulated data with key percentiles
def plot_distribution(data, title="Distribution"):
    fig, ax = plt.subplots(figsize=(10, 6)) # Set figure size
    ax.hist(data, bins=50, color='skyblue', edgecolor='black', alpha=0.7) # Histogram
    ax.set_title(title)
    ax.set_xlabel("Total Annual Cost")
    ax.set_ylabel("Frequency")

    # Plot 5th, 50th, 95th percentiles
    percentiles = np.percentile(data, [5, 50, 95])
    for p, c, l in zip(percentiles, ['r', 'g', 'r'], ['5th', 'Median', '95th']):
        ax.axvline(p, color=c, linestyle='--', label=f'{l}: ${p:,.0f}')
    ax.legend()
    return fig

# -----------------------------
# --- Main Streamlit App
# -----------------------------
# Set Streamlit page config
st.set_page_config(
    page_title="Monte Carlo Cost Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom style for aesthetics
st.markdown("""
    <style>
        /* Apply center alignment to all table cells and headers in dataframes */
        .stDataFrame tbody td, .stDataFrame thead th {
            text-align: center !important;
            vertical-align: middle !important;
        }
        .stDataFrame tbody td div, .stDataFrame thead th div {
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
""", unsafe_allow_html=True)
# Title of the Streamlit app
st.title("Monte Carlo Simulation for Events Cost Forecasting")
st.markdown("""
Use this app to simulate projected incident costs using historical data.
Adjust parameters and explore cost forecasts by category or year.
""")
# Outcome Factor selection
# outcome_factor = st.radio("Select Outcome Factor", ["All Events", "Claims"], horizontal=True)
# User chooses simulation mode
# mode = st.radio("Select Simulation Mode", ["Total Projection", "Multi-Year Summary"], horizontal=True)

# Outcome Factor & Mode
st.sidebar.header("Simulation Setup")
outcome_factor = st.sidebar.radio("Outcome Factor", ["All Events", "Claims"])
mode = st.sidebar.radio("Simulation Mode", ["Total Projection", "Multi-Year Summary"])
category_option = None


# Show event category dropdown only for "Total Projection"
if mode == "Total Projection":
    category_option = st.sidebar.selectbox("Select Event Category", ["All Events", "Mechanism", "Nature", "Bodily Location"])
else:
    category_option = None  # Or skip defining it if not needed

# category_columns = {
#     "Mechanism": ["Primary Mechanism", "Secondary Mechanism"],
#     "Nature": ["Primary Nature of Injury", "Secondary Nature of Injury"],
#     "Bodily Location": ["Primary Bodily Location", "Secondary Bodily Location"]
# }

# incident_category_map = {
#     "Mechanism": ["Primary Mechanism", "Secondary Mechanism"],
#     "Nature": ["Primary Nature of Injury", "Secondary Nature of Injury"],
#     "Bodily Location": ["Primary Bodily Location", "Secondary Bodily Location"]
# }
# Dynamic category mapping based on outcome factor
if outcome_factor == "Claims":
    category_columns = {
        "Mechanism": ["Primary Mechanism", "Secondary Mechanism"],  # Cols I & J
        "Nature": ["Primary Nature of Injury", "Secondary Nature of Injury"],  # Cols E & F
        "Bodily Location": ["Primary Bodily Location", "Secondary Bodily Location"]  # Cols G & H
    }
    incident_category_map = {
        "Mechanism": ["Primary Mechanism", "Secondary Mechanism"],
        "Nature": ["Primary Nature of Injury", "Secondary Nature of Injury"],
        "Bodily Location": ["Primary Bodily Location", "Secondary Bodily Location"]
    }   
    # Use actual column *names* from COSTS dataframe, not Excel letters
    # So map them to expected column names
    cost_sheet_column_names = {
        "I": "Primary Mechanism",
        "J": "Secondary Mechanism",
        "E": "Primary Nature of Injury",
        "F": "Secondary Nature of Injury",
        "G": "Primary Bodily Location",
        "H": "Secondary Bodily Location"
    }
else:
    incident_category_map = {
        "Mechanism": ["Primary Mechanism", "Secondary Mechanism"],
        "Nature": ["Primary Nature of Injury", "Secondary Nature of Injury"],
        "Bodily Location": ["Primary Bodily Location", "Secondary Bodily Location"]
    }   
# Choose file input method
# use_upload = st.checkbox("Use File Upload", value=True)

# # Upload mode
# if use_upload:
#     uploaded_file = st.file_uploader("Upload Excel file with INCIDENTS and COSTS sheets", type=['xlsx', 'xls'])
#     file_valid = uploaded_file is not None # Check if file is provided
# else:
#     # Local path input mode
#     file_path = st.text_input("Path to Excel File", "Cleaned Data.xlsx")
#     file_valid = os.path.exists(file_path) # Check if file exists
# Sidebar inputs for file
st.sidebar.header("Upload Data")
use_upload = st.sidebar.checkbox("Use File Upload", value=True)
if use_upload:
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
    file_valid = uploaded_file is not None
else:
    file_path = st.sidebar.text_input("Path to Excel File", "Cleaned Data.xlsx")
    file_valid = os.path.exists(file_path)
# Proceed only if a valid file is given
if file_valid:
    # Load data
    incidents_df, costs_df = load_data(uploaded_file if use_upload else file_path, from_upload=use_upload, outcome_factor=outcome_factor)

    if incidents_df is None or costs_df is None:
        st.stop() # Stop execution if loading failed

    # --- Total Projection Mode ---
    if mode == "Total Projection":
        st.subheader("Simulation Parameters")
        avg_events, avg_cost, std_dev_cost = calculate_statistics(incidents_df, costs_df)

        # Display computed statistics
        st.write(f"Average Events per Year: {avg_events:.2f}")
        st.write(f"Average Cost per Event: ${avg_cost:,.2f}")
        st.write(f"Standard Deviation of Cost: ${std_dev_cost:,.2f}")

        # User-adjustable sliders
        col1, col2 = st.columns(2)
        with col1:
            event_volatility = st.slider("Event Volatility", 0.5, 2.0, 1.0, 0.1)
        with col2:
            cost_volatility = st.slider("Cost Volatility", 0.5, 2.0, 1.0, 0.1)

        # Run simulation on button click
        if st.button("Run Total Simulation"):
            if category_option == "All Events":
                total_costs = run_monte_carlo_simulation(
                    avg_events, avg_cost, std_dev_cost, event_volatility, cost_volatility
                )
                st.write(f"Mean Cost: ${np.mean(total_costs):,.2f}")
                st.pyplot(plot_distribution(total_costs))
            else:
                st.subheader(f"Simulation by {category_option}")
                category_cols = incident_category_map[category_option]
                output_dir = "simulation_output"
                source_df = costs_df if outcome_factor == "Claims" else incidents_df
                melted_df = pd.melt(source_df[['Date of Injury'] + category_cols],
                                    id_vars=['Date of Injury'],
                                    value_vars=category_cols,
                                    var_name='Type', value_name='Category').dropna()

                merged_df = pd.merge(melted_df, costs_df[['Date of Injury', 'Incurred']], on='Date of Injury')

                category_stats = merged_df.groupby('Category').agg(
                    Count=('Incurred', 'count'),
                    Avg_Cost=('Incurred', 'mean'),
                    Std_Cost=('Incurred', 'std')
                ).reset_index()
                
                st.dataframe(category_stats.round(2))
                # os.makedirs(output_dir, exist_ok=True)
                # category_stats.to_csv(os.path.join(output_dir, "simulation by .csv"))
                # category_stats.to_excel(os.path.join(output_dir, "simulation by {category_option}.xlsx"))
                # st.success(f"Results exported to `{output_dir}`")
                results = []
                for _, row in category_stats.iterrows():
                    sim_costs = np.random.normal(row['Avg_Cost'] * cost_volatility,
                                                 row['Std_Cost'] * cost_volatility, 10000)
                    sim_costs = np.clip(sim_costs, 1, None)
                    total_costs = row['Count'] * sim_costs
                    results.append({
                        'Category': row['Category'],
                        'Mean': np.mean(total_costs),
                        '5th Percentile': np.percentile(total_costs, 5),
                        'Median': np.percentile(total_costs, 50),
                        '95th Percentile': np.percentile(total_costs, 95),
                        'Max': np.max(total_costs)
                    })

                results_df = pd.DataFrame(results).sort_values("Mean", ascending=False)
                st.subheader("Simulation Results by Category")
                st.dataframe(results_df.round(2))
                # os.makedirs(output_dir, exist_ok=True)
                # category_stats.to_csv(os.path.join(output_dir, "simulation results by {category_option}.csv"))
                # category_stats.to_excel(os.path.join(output_dir, "simulation results by {category_option}.xlsx"))
                # st.success(f"Results exported to `{output_dir}`")
                st.altair_chart(
                    alt.Chart(results_df).mark_bar().encode(
                        x=alt.X("Category:N", sort='-y'),
                        y="Mean:Q",
                        tooltip=["Category", "Mean", "5th Percentile", "95th Percentile"]
                    ).properties(width=800, height=400),
                    use_container_width=True
                )
    # --- Multi-Year Summary Mode ---
    elif mode == "Multi-Year Summary":
        # Sidebar controls for advanced simulation
        st.sidebar.header("Multi-Year Controls")
        n_simulations = st.sidebar.number_input("Number of Simulations", 10000, 100000, 10000, 1000)
        cost_volatility = st.sidebar.slider("Cost Volatility Multiplier", 0.1, 3.0, 1.0, 0.1)
        incident_growth_rate = st.sidebar.slider("Incident Growth Rate (annual %)", 0.0, 0.5, 0.05, 0.01)
        export_results = st.sidebar.checkbox("Export Results", True)
        output_dir = "simulation_output"
         # Group data by year
         # Choose source dataframe based on outcome factor
        source_df = costs_df if outcome_factor == "Claims" else incidents_df

        # Ensure the 'Date of Injury' column is datetime
        source_df["Date of Injury"] = pd.to_datetime(source_df["Date of Injury"], errors='coerce')

        # Drop rows with invalid dates
        valid_dates_df = source_df.dropna(subset=["Date of Injury"])

        # Extract year
        valid_dates_df["Year"] = valid_dates_df["Date of Injury"].dt.year

        # Count events per year
        annual_events = valid_dates_df["Year"].value_counts().sort_index()
        # annual_events = incidents_df.groupby('Year').size()
        annual_total_cost = costs_df.groupby('Year')['Incurred'].sum()
        annual_average_cost = annual_total_cost / annual_events
        annual_std_cost = costs_df.groupby('Year')['Incurred'].std()

         # Combine into one DataFrame
        combined_data = pd.DataFrame({
            'Annual Events': annual_events,
            'Annual Total Cost': annual_total_cost,
            'Annual Average Cost': annual_average_cost,
            'Annual Std Dev': annual_std_cost
        }).dropna()

        st.subheader("Combined Annual Data")
        st.dataframe(combined_data.round(2))

        # Run simulations per year
        simulation_results = {}
        for i, (year, row) in enumerate(combined_data.iterrows()):
            n_events = int(row['Annual Events'] * ((1 + incident_growth_rate) ** i))
            avg_cost = row['Annual Average Cost']
            std_dev = row['Annual Std Dev'] * cost_volatility

            sim_counts = np.random.poisson(lam=n_events, size=n_simulations)
            sim_costs = np.random.normal(loc=avg_cost, scale=std_dev, size=n_simulations)
            sim_costs = np.clip(sim_costs, 0, None) # No negative costs
            total_costs = sim_counts * sim_costs

            # Save key metrics
            simulation_results[year] = {
                '5th Percentile': np.percentile(total_costs, 5),
                'Median': np.percentile(total_costs, 50),
                '95th Percentile': np.percentile(total_costs, 95),
                'Mean': np.mean(total_costs),
                'Max': np.max(total_costs)
            }

        # Convert to DataFrame and display
        results_df = pd.DataFrame(simulation_results).T.round(2)
        results_df.index.name = 'Year'
        st.subheader("Simulation Summary")
        st.dataframe(results_df)

        # Export to CSV/Excel if enabled
        if export_results:
            os.makedirs(output_dir, exist_ok=True)
            results_df.to_csv(os.path.join(output_dir, "monte_carlo_simulation_summary.csv"))
            results_df.to_excel(os.path.join(output_dir, "monte_carlo_simulation_summary.xlsx"))
            st.success(f"Results exported to `{output_dir}`")

         # Plot results using Altair
        st.subheader("Cost Summary by Year")
        results_df_reset = results_df.reset_index().melt(id_vars='Year', var_name='Metric', value_name='Cost')
        chart = alt.Chart(results_df_reset).mark_line(point=True).encode(
            x='Year:O',
            y='Cost:Q',
            color='Metric:N',
            tooltip=['Year', 'Metric', 'Cost']
        ).properties(width=800, height=400)
        st.altair_chart(chart, use_container_width=True)

        # # Histogram for latest year
        # st.subheader("Distribution for Most Recent Year")
        # latest_year = int(results_df.index.max())
        # latest_index = list(combined_data.index).index(latest_year)
        # latest_events = int(combined_data.loc[latest_year, 'Annual Events'] * ((1 + incident_growth_rate) ** latest_index))
        # latest_avg = combined_data.loc[latest_year, 'Annual Average Cost']
        # latest_std = combined_data.loc[latest_year, 'Annual Std Dev'] * cost_volatility

        # sim_counts = np.random.poisson(lam=latest_events, size=n_simulations)
        # sim_costs = np.random.normal(loc=latest_avg, scale=latest_std, size=n_simulations)
        # sim_costs = np.clip(sim_costs, 0, None)
        # sim_totals = sim_counts * sim_costs

        # st.pyplot(plot_distribution(sim_totals, title=f"Simulation Histogram ({latest_year})"))
# Handle invalid file case
else:
    st.warning("Please upload a valid Excel file or enter a valid file path.")
