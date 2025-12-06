"""
Streamlit GUI wrapper for untitled26.py functionality.
Features:
- Fetch/scrape the Wikipedia table (or upload CSV)
- Clean numeric columns
- Show raw / cleaned / normalized data
- Interactive plots (Plotly + Seaborn/matplotlib)
- Train a regression model and show predictions/metrics
- Interactive profit prediction widget
- Download cleaned CSV

Run: streamlit run gui_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Companies Explorer", layout="wide")

# ---- Utils ----
def clean_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("$", "", regex=False)
    s = s.str.replace("—", "", regex=False)
    s = s.str.replace(r"\[.*?\]", "", regex=True)
    s = s.str.replace("N/A", "", regex=False)
    s = s.str.strip()
    return pd.to_numeric(s, errors="coerce")

@st.cache_data(show_spinner=False)
def fetch_wikipedia_table(url: str) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"class": "wikitable"})
    if table is None:
        raise ValueError("Could not find a wikitable on the page.")
    headers = [th.get_text(strip=True) for th in table.find("tr").find_all(["th","td"])]
    rows = [
        [td.get_text(strip=True) for td in tr.find_all(["td","th"])]
        for tr in table.find_all("tr")[1:]
        if len(tr.find_all(["td","th"])) == len(headers)
    ]
    return pd.DataFrame(rows, columns=headers)

# small profit estimator used in original script
segment_profit_margin = {
    "Technology": 0.25,
    "Finance": 0.20,
    "Healthcare": 0.18,
    "Energy": 0.12,
    "Retail": 0.08,
    "Automotive": 0.10,
    "Telecom": 0.15,
    "Other": 0.10
}
region_adjustment = {
    "North America": 1.05,
    "Europe": 0.95,
    "Asia": 1.10,
    "Middle East": 0.90,
    "South America": 0.85,
    "Africa": 0.80,
    "Other": 1.00
}

def predict_profit_simple(company_name, revenue, segment, location):
    seg = segment if segment in segment_profit_margin else "Other"
    loc = location if location in region_adjustment else "Other"
    base_margin = segment_profit_margin[seg]
    regional_factor = region_adjustment[loc]
    noise = np.random.uniform(-0.02, 0.02)
    profit_margin = (base_margin + noise) * regional_factor
    profit = revenue * profit_margin
    return profit, profit_margin

# ---- UI Layout ----
st.title("📊 Companies Explorer — interactive GUI")

with st.sidebar:
    st.header("Data Source")
    source = st.radio("Choose data source:", ("Fetch Wikipedia", "Upload CSV", "Use bundled example"))
    if source == "Fetch Wikipedia":
        url = st.text_input("Wikipedia URL:", "https://en.wikipedia.org/wiki/List_of_largest_companies_by_revenue")
        if st.button("Fetch data 📥"):
            try:
                df_raw = fetch_wikipedia_table(url)
                st.session_state['df_raw'] = df_raw
                st.success("Fetched table and stored in session")
            except Exception as e:
                st.error(f"Fetch failed: {e}")
    elif source == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is not None:
            df_raw = pd.read_csv(uploaded)
            st.session_state['df_raw'] = df_raw
            st.success("CSV uploaded and stored in session")
    else:
        st.write("Using an example derived from your script when available.")

st.sidebar.markdown("---")
st.sidebar.header("Actions")
if st.sidebar.button("Clean / Prepare Data ✨"):
    if 'df_raw' not in st.session_state:
        st.sidebar.error("No raw data available. Fetch or upload first.")
    else:
        df = st.session_state['df_raw'].copy()
        # identify numeric columns by common substrings
        numeric_cols = [c for c in df.columns if any(k in c.lower() for k in ["revenue","profit","employees"]) ]
        # Clean any numeric-like columns we detected
        for col in numeric_cols:
            df[col] = clean_numeric(df[col])
        for col in numeric_cols:
            if "profit" not in col.lower():
                df[col] = df[col].fillna(df[col].median())

        # Standardize column names so downstream code can rely on 'Revenue', 'Profit', 'Employees'
        col_map = {}
        seen = set()
        for c in df.columns:
            lc = c.lower()
            if "revenue" in lc and "Revenue" not in seen:
                col_map[c] = "Revenue"
                seen.add("Revenue")
            elif "profit" in lc and "Profit" not in seen:
                col_map[c] = "Profit"
                seen.add("Profit")
            elif ("employee" in lc or "employees" in lc) and "Employees" not in seen:
                col_map[c] = "Employees"
                seen.add("Employees")

        if col_map:
            df = df.rename(columns=col_map)

        # Re-ensure canonical numeric columns are numeric and filled
        for std in ["Revenue", "Profit", "Employees"]:
            if std in df.columns:
                df[std] = clean_numeric(df[std])
                if std != "Profit":
                    df[std] = df[std].fillna(df[std].median())

        # Derived metrics (only if the canonical columns exist)
        if "Employees" in df.columns and "Revenue" in df.columns:
            df["Revenue_per_employee"] = df["Revenue"] / df["Employees"].replace(0, pd.NA)
        if "Profit" in df.columns and "Revenue" in df.columns:
            df["Profit_margin"] = df["Profit"] / df["Revenue"]
        if "Profit" in df.columns:
            df["Profit_status"] = df["Profit"].apply(lambda x: "Profit" if x>0 else ("Loss" if x<0 else "Unknown"))
        st.session_state['df_clean'] = df
        st.sidebar.success("Data cleaned and stored as df_clean")

if st.sidebar.button("Normalize numeric columns 🔢"):
    if 'df_clean' not in st.session_state:
        st.sidebar.error("No cleaned data found. Run Clean / Prepare Data first.")
    else:
        dfc = st.session_state['df_clean'].copy()
        num_cols = [c for c in dfc.columns if pd.api.types.is_numeric_dtype(dfc[c])]
        if num_cols:
            scaler = MinMaxScaler()
            dfc[num_cols] = scaler.fit_transform(dfc[num_cols].fillna(0))
            st.session_state['df_norm'] = dfc
            st.sidebar.success("Normalized numeric columns and stored as df_norm")
        else:
            st.sidebar.warning("No numeric columns to normalize.")

st.sidebar.markdown("---")
st.sidebar.header("Model")
train_model = st.sidebar.button("Train Revenue Model 🤖")

# ---- Main content tabs ----
tabs = st.tabs(["Data", "Visuals", "Model", "Profit Predictor"])

# --- Data Tab ---
with tabs[0]:
    st.header("Data Viewer")
    if 'df_raw' in st.session_state:
        st.subheader("Raw Data")
        st.dataframe(st.session_state['df_raw'].head(50))
        st.download_button("Download Raw CSV", st.session_state['df_raw'].to_csv(index=False), file_name="companies_raw.csv")
    else:
        st.info("No raw data yet. Use the sidebar to fetch or upload data.")

    if 'df_clean' in st.session_state:
        st.subheader("Cleaned / Prepared Data")
        st.dataframe(st.session_state['df_clean'].head(80))
        st.download_button("Download Cleaned CSV", st.session_state['df_clean'].to_csv(index=False), file_name="companies_cleaned.csv")

    if 'df_norm' in st.session_state:
        st.subheader("Normalized Data")
        st.dataframe(st.session_state['df_norm'].head(80))

# --- Visuals Tab ---
with tabs[1]:
    st.header("Interactive Visualizations")
    dfv = None
    if 'df_clean' in st.session_state:
        dfv = st.session_state['df_clean']
    elif 'df_raw' in st.session_state:
        dfv = st.session_state['df_raw']
    else:
        st.info("No data to visualize — fetch or upload data in the sidebar.")

    if dfv is not None:
        # Debug info
        st.write("Available columns:", dfv.columns.tolist())
        
        # Find revenue-like column
        revenue_col = None
        for col in dfv.columns:
            if "revenue" in col.lower():
                revenue_col = col
                break
        
        st.subheader("Top Companies by Revenue")
        if revenue_col:
            st.write(f"Using revenue column: {revenue_col}")
            # Convert to numeric, handling common formats
            dfv[revenue_col] = pd.to_numeric(
                dfv[revenue_col].astype(str)
                .str.replace(",", "")
                .str.replace("$", "")
                .str.replace("₹", "")
                .str.replace("−", "-")  # Handle minus sign
                .str.replace("[", "", regex=False)
                .str.replace("]", "", regex=False)
                .str.strip(),
                errors="coerce"
            )
            st.write(f"First few revenue values after cleaning:", dfv[revenue_col].head().tolist())
            
            top_n = st.slider("Top N", 5, 30, 10)
            top_rev = dfv.nlargest(top_n, revenue_col)
            
            # attempt to find company name column as first non-numeric
            name_col = dfv.columns[0]
            st.write(f"Using company name column: {name_col}")
            
            try:
                fig = px.bar(
                    top_rev, 
                    x=revenue_col, 
                    y=name_col, 
                    orientation='h',
                    color=revenue_col,
                    labels={revenue_col: "Revenue"},
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")
                st.write("Revenue data sample:", top_rev[revenue_col].head())
                st.write("Name data sample:", top_rev[name_col].head())
        else:
            st.warning("No revenue column found. Looking for column name containing 'revenue' (case-insensitive)")

        st.subheader("Revenue vs Profit Scatter")
        # Find profit column
        profit_col = None
        for col in dfv.columns:
            if "profit" in col.lower():
                profit_col = col
                break
        
        if revenue_col and profit_col:
            st.write(f"Using columns: Revenue={revenue_col}, Profit={profit_col}")
            
            # Ensure profit is numeric
            dfv[profit_col] = pd.to_numeric(
                dfv[profit_col].astype(str)
                .str.replace(",", "")
                .str.replace("$", "")
                .str.replace("₹", "")
                .str.replace("−", "-")  # Handle minus sign
                .str.replace("[", "", regex=False)
                .str.replace("]", "", regex=False)
                .str.strip(),
                errors="coerce"
            )
            
            # Calculate profit margin after cleaning numeric columns
            dfv["Profit_margin"] = dfv[profit_col] / dfv[revenue_col]
            
            try:
                fig2 = px.scatter(
                    dfv, 
                    x=revenue_col, 
                    y=profit_col,
                    color="Profit_status" if "Profit_status" in dfv.columns else None,
                    hover_name=dfv.columns[0],
                    labels={
                        revenue_col: "Revenue",
                        profit_col: "Profit"
                    }
                )
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating scatter plot: {str(e)}")
                st.write("Data sample:")
                st.write(dfv[[revenue_col, profit_col]].head())

        st.subheader("Profit Margin Distribution")
        if revenue_col and profit_col:  # Changed condition to check if we can calculate margin
            # Show profit margin summary
            st.write("Profit margin summary:")
            margin_stats = dfv["Profit_margin"].describe()
            st.write(margin_stats)
            
            # Remove extreme outliers for better visualization
            margin_mean = margin_stats['mean']
            margin_std = margin_stats['std']
            clean_margins = dfv["Profit_margin"][
                (dfv["Profit_margin"] > margin_mean - 3 * margin_std) & 
                (dfv["Profit_margin"] < margin_mean + 3 * margin_std)
            ]
            
            try:
                fig3 = px.histogram(
                    clean_margins.to_frame(),  # Convert series to dataframe
                    x="Profit_margin",
                    nbins=30,
                    marginal="box",
                    title="Profit Margin Distribution (excluding extreme outliers)",
                    labels={"Profit_margin": "Profit Margin (Profit/Revenue)"}
                )
                st.plotly_chart(fig3, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating histogram: {str(e)}")
                st.write("Profit margin data sample:", clean_margins.head())
        
        # Add revenue category analysis
        st.subheader("Revenue Category Analysis")
        if revenue_col:
            # Create revenue categories
            def categorize_revenue(value, data):
                q1, q2 = data.quantile([0.33, 0.66])
                if value <= q1:
                    return "Low"
                elif value <= q2:
                    return "Medium"
                else:
                    return "High"
            
            dfv["Revenue_Category"] = dfv[revenue_col].apply(lambda x: categorize_revenue(x, dfv[revenue_col]))
            
            # 1. Revenue vs Profit with Revenue Categories
            if profit_col:
                fig4 = px.scatter(
                    dfv,
                    x=revenue_col,
                    y=profit_col,
                    color="Revenue_Category",
                    title="Revenue vs Profit by Revenue Category",
                    labels={revenue_col: "Revenue", profit_col: "Profit"},
                    category_orders={"Revenue_Category": ["Low", "Medium", "High"]},
                    hover_name=dfv.columns[0]
                )
                st.plotly_chart(fig4, use_container_width=True)
            
            # 2. Category Distribution Analysis
            fig5 = px.box(
                dfv,
                x="Revenue_Category",
                y=revenue_col,
                title="Revenue Distribution by Category",
                labels={"Revenue_Category": "Revenue Category", revenue_col: "Revenue"},
                category_orders={"Revenue_Category": ["Low", "Medium", "High"]}
            )
            st.plotly_chart(fig5, use_container_width=True)
            
            # 3. Category Counts Matrix
            category_counts = dfv["Revenue_Category"].value_counts().reset_index()
            category_counts.columns = ["Category", "Count"]
            
            fig6 = px.bar(
                category_counts,
                x="Category",
                y="Count",
                title="Number of Companies in Each Revenue Category",
                color="Category",
                text="Count",
                category_orders={"Category": ["Low", "Medium", "High"]},
            )
            fig6.update_traces(textposition='outside')
            st.plotly_chart(fig6, use_container_width=True)
            
            # Show summary statistics for each category
            st.subheader("Category Summary Statistics")
            summary_stats = dfv.groupby("Revenue_Category")[revenue_col].agg([
                "count", "mean", "median", "std"
            ]).round(2)
            st.dataframe(summary_stats)

# --- Model Tab ---
with tabs[2]:
    st.header("Revenue Prediction Model")
    if 'df_clean' not in st.session_state:
        st.info("Prepare data first (sidebar -> Clean / Prepare Data)")
    else:
        dfm = st.session_state['df_clean'].copy()
        # choose feature set
        st.write("Choose features to predict Revenue")
        features = []
        # allow user to pick available numeric columns
        numeric_cols = [c for c in dfm.columns if pd.api.types.is_numeric_dtype(dfm[c])]
        target = st.selectbox("Target column", options=[c for c in dfm.columns if pd.api.types.is_numeric_dtype(dfm[c])], index=0)
        feat_choices = st.multiselect("Numeric features", options=[c for c in numeric_cols if c!=target], default=[c for c in ["Employees"] if c in numeric_cols])

        # optional location encoding
        loc_col = None
        for c in dfm.columns:
            if any(x in c.lower() for x in ["location","headquarters","country"]):
                loc_col = c
                break
        if loc_col:
            use_loc = st.checkbox(f"Use location column: {loc_col}")
        else:
            use_loc = st.checkbox("Add dummy location encoded column (0)")

        if st.button("Train model ▶️"):
            X = dfm[feat_choices].copy()
            if use_loc and loc_col:
                le = LabelEncoder()
                X['Location_encoded'] = le.fit_transform(dfm[loc_col].astype(str))
            elif use_loc and not loc_col:
                X['Location_encoded'] = 0
            y = dfm[target]
            # simple split
            test_size = st.slider("Test set fraction", 0.1, 0.5, 0.25)
            X_train, X_test, y_train, y_test = train_test_split(X.fillna(0), y.fillna(0), test_size=test_size, random_state=42)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.metric("R²", f"{r2:.3f}")
            st.metric("MAE", f"{mae:.3f}")
            st.metric("RMSE", f"{rmse:.3f}")
            results = pd.DataFrame({"Actual":y_test.values, "Pred":y_pred}).reset_index(drop=True)
            st.subheader("First 10 predictions")
            st.dataframe(results.head(10))
            st.session_state['model_artifacts'] = {
                'model': model,
                'scaler': scaler,
                'features': list(X.columns),
                'le': le if (loc_col and use_loc) else None,
                'target': target
            }
            st.success("Model trained and saved to session_state['model_artifacts']")

        if 'model_artifacts' in st.session_state:
            st.subheader("Make a prediction")
            ma = st.session_state['model_artifacts']
            input_vals = {}
            for f in ma['features']:
                input_vals[f] = st.number_input(f, value=float(dfm[f].median()) if f in dfm else 0.0)
            if st.button("Predict now 🔮"):
                model = ma['model']
                scaler = ma['scaler']
                X_in = pd.DataFrame([input_vals])[ma['features']]
                Xs = scaler.transform(X_in.fillna(0))
                pred = model.predict(Xs)[0]
                st.write(f"Predicted {ma['target']}: {pred:.4f}")

# --- Profit Predictor Tab ---
with tabs[3]:
    st.header("Interactive Profit Predictor 🧮")
    
    # Get the data for F1 score calculation if available
    if 'df_clean' in st.session_state and 'Profit' in st.session_state['df_clean'].columns:
        df_eval = st.session_state['df_clean'].copy()
        
        # Calculate actual profit categories
        def categorize_profit(row):
            if pd.isna(row['Profit']):
                return "Unknown"
            elif row['Profit'] > 0:
                return "Profitable"
            elif row['Profit'] < 0:
                return "Loss"
            else:
                return "Break-even"
        
        df_eval['Actual_Category'] = df_eval.apply(categorize_profit, axis=1)
        
        # Calculate predicted categories using our simple model
        predictions = []
        for _, row in df_eval.iterrows():
            if pd.isna(row['Revenue']):
                predictions.append("Unknown")
            else:
                pred_profit, _ = predict_profit_simple(
                    row[df_eval.columns[0]],  # Company name
                    row['Revenue'],
                    'Other',  # Default segment
                    'Other'   # Default region
                )
                if pred_profit > 0:
                    predictions.append("Profitable")
                elif pred_profit < 0:
                    predictions.append("Loss")
                else:
                    predictions.append("Break-even")
        
        df_eval['Predicted_Category'] = predictions
        
        # Calculate metrics
        valid_mask = (df_eval['Actual_Category'] != "Unknown") & (df_eval['Predicted_Category'] != "Unknown")
        if valid_mask.any():
            f1 = f1_score(
                df_eval[valid_mask]['Actual_Category'],
                df_eval[valid_mask]['Predicted_Category'],
                average='weighted'
            )
            
            # Display F1 Score
            st.subheader("Model Performance")
            st.metric("F1 Score (Weighted)", f"{f1:.3f}")
            
            # Create confusion matrix
            confusion = pd.crosstab(
                df_eval[valid_mask]['Actual_Category'],
                df_eval[valid_mask]['Predicted_Category'],
                margins=True
            )
            
            # Display confusion matrix
            st.write("Confusion Matrix:")
            st.dataframe(confusion)
            
            # Visualize prediction accuracy
            fig = px.imshow(
                confusion.iloc[:-1, :-1],  # Remove totals
                labels=dict(x="Predicted", y="Actual", color="Count"),
                title="Profit Prediction Accuracy",
                color_continuous_scale="Blues"
            )
            fig.update_traces(text=confusion.iloc[:-1, :-1].values, texttemplate="%{z}")
            st.plotly_chart(fig, use_container_width=True)
    
    # Interactive predictor
    st.subheader("Make a Prediction")
    company_name = st.text_input("Company name", value="ACME Corp")
    revenue = st.number_input("Revenue (in same units as your data, e.g., billions)", value=1.0, step=0.1)
    seg = st.selectbox("Segment", options=list(segment_profit_margin.keys()), index=0)
    loc = st.selectbox("Location / Region", options=list(region_adjustment.keys()), index=0)
    if st.button("Estimate Profit 💡"):
        profit, profit_margin = predict_profit_simple(company_name, revenue, seg, loc)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estimated Profit", f"{profit:,.2f}")
        with col2:
            st.metric("Profit Margin", f"{profit_margin*100:.2f}%")
        
        # Show prediction category
        category = "Profitable" if profit > 0 else "Loss" if profit < 0 else "Break-even"
        st.write(f"Prediction Category: **{category}**")

# ---- Footer / tips ----
st.markdown("---")
st.caption("Tips: Use the sidebar to fetch data from Wikipedia or upload your CSV. Clean and normalize before training models or plotting.")

# allow user to save session cleaned data to a file
if 'df_clean' in st.session_state:
    st.sidebar.download_button("Download cleaned CSV", st.session_state['df_clean'].to_csv(index=False), file_name="companies_cleaned.csv")

