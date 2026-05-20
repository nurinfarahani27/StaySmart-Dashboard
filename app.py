import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import pickle      
import numpy as np

st.set_page_config(page_title="StaySmart Dashboard", layout="wide")
st.title("📊 StaySmart: Customer Loyalty Analytics Dashboard")

# Retrieve data from SQLite Database
def get_dashboard_data():
    conn = sqlite3.connect('telco_churn.db')
    df_clean = pd.read_sql_query("SELECT * FROM churn_table", conn)
    conn.close()
    return df_clean

df_filtered = get_dashboard_data()

@st.cache_resource
def load_ml_model():
    try:
        with open('churn_model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

model = load_ml_model()

# ==========================================
# SIDEBAR - REQUIRED FEATURES: FILTERS / SLICERS
# ==========================================
st.sidebar.header("🎛️ Dimensional Filters (Slice/Dice)")

gender_opt = df_filtered['gender'].unique().tolist()
selected_gender = st.sidebar.multiselect("Customer Gender:", gender_opt, default=gender_opt)

contract_opt = df_filtered['Contract'].unique().tolist()
selected_contract = st.sidebar.multiselect("Contract Type:", contract_opt, default=contract_opt)

internet_opt = df_filtered['InternetService'].unique().tolist()
selected_internet = st.sidebar.multiselect("Internet Service:", internet_opt, default=internet_opt)

# Filter data based on sidebar selection
df_viz = df_filtered[
    (df_filtered['gender'].isin(selected_gender)) &
    (df_filtered['Contract'].isin(selected_contract)) &
    (df_filtered['InternetService'].isin(selected_internet))
]

# ==========================================
# CREATING TABS FOR CLEAN INTERACTION
# ==========================================
tab_dashboard, tab_ml = st.tabs(["📊 BI Dashboard & OLAP", "🔮 StaySmart Predictor"])

# ------------------------------------------
# TAB 1: BI DASHBOARD & OLAP (Semua Section 1-4 duduk dalam ni)
# ------------------------------------------
with tab_dashboard:
    # ==========================================
    # SECTION 1: EXECUTIVE SUMMARY & KPI CARDS
    # ==========================================
    st.header("Section 1 — Executive Summary")
    st.subheader("📌 Key Performance Indicators (KPI Summary Cards)")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_cust = len(df_viz)
    churn_cust = len(df_viz[df_viz['Churn'] == 'Yes'])
    churn_rate = (churn_cust / total_cust) * 100 if total_cust > 0 else 0
    avg_bill = df_viz['MonthlyCharges'].mean() if total_cust > 0 else 0

    # Meets requirements: Customer count, total churn, growth/churn rate, average order value
    kpi1.metric(label="📊 Customer Count (Active)", value=f"{total_cust:,}")
    kpi2.metric(label="🚨 Total Churn", value=f"{churn_cust:,}")
    kpi3.metric(label="📈 Churn Rate (Growth Rate)", value=f"{churn_rate:.2f}%")
    kpi4.metric(label="💳 Average Order Value (Avg Bill)", value=f"${avg_bill:.2f}")

    st.markdown("---")

    # ==========================================
    # SECTION 2: TREND ANALYSIS
    # ==========================================
    st.header("Section 2 — Trend Analysis")
    col_trend1, col_trend2 = st.columns(2)

    with col_trend1:
        st.markdown("**📉 1. Trend Line Chart: Churn Pattern Over Time (Tenure Months)**")
        df_trend = df_viz.groupby(['tenure', 'Churn']).size().reset_index(name='Customer_Count')
        fig_line = px.line(df_trend, x="tenure", y="Customer_Count", color="Churn", markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

    with col_trend2:
        st.markdown("**📊 2. Bar Chart Comparison: Churn Comparison by Tenure Group**")
        # Translating tenure groups mapping visually for the chart order
        fig_bar = px.histogram(df_viz, x="Tenure_Group", color="Churn", barmode="group",
                               category_orders={"Tenure_Group": ["0-1 Tahun", "1-2 Tahun", "2-4 Tahun", "Lebih 4 Tahun"]})
        fig_bar.update_layout(xaxis_title="Tenure Group", yaxis_title="Count")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # SECTION 3: SEGMENTATION ANALYSIS & HEATMAP
    # ==========================================
    st.header("Section 3 — Segmentation Analysis")
    col_seg1, col_seg2 = st.columns(2)

    with col_seg1:
        st.markdown("**🍰 3. Pie / Proportion Chart: Payment Method Breakdown for Churned Customers**")
        fig_pie = px.pie(df_viz[df_viz['Churn'] == 'Yes'], names="PaymentMethod", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_seg2:
        st.markdown("**🔥 4. Heatmap Chart: Correlation Matrix of Charges & Tenure**")
        corr_matrix = df_viz[['tenure', 'MonthlyCharges', 'TotalCharges']].corr()
        fig_heat = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # SECTION 4: INSIGHTS & DRILL-DOWN INTERACTIVE CHART
    # ==========================================
    st.header("Section 4 — Insights")

    st.markdown("**🔍 5. Drill-down Interactive Chart: Inspect Internet Services by Contract Type**")
    drill_contract = st.selectbox("Select Contract Type for Drill-Down:", df_viz['Contract'].unique())
    df_drill = df_viz[df_viz['Contract'] == drill_contract]

    fig_drill = px.bar(df_drill, x="InternetService", color="Churn", title=f"Internet Service Breakdown for Contract: {drill_contract}", barmode="stack")
    st.plotly_chart(fig_drill, use_container_width=True)

    # BUSINESS INSIGHTS SECTION
    st.subheader("💡 Business Insights & Recommendations")
    st.info(
        "1. **High-Risk Segment:** Customers using **Fiber Optic** internet services on a **Month-to-month** contract exhibit the most aggressive churn behavior.\n"
        f"2. **Financial Impact:** The financial loss (Average Order Value) from each churned customer amounts to approximately **${avg_bill:.2f}** per month.\n"
        "3. **Strategic Recommendation:** Management should introduce discount incentives or loyalty programs to transition month-to-month subscribers into long-term contracts (1-2 Years) to secure predictable retention."
    )

# ------------------------------------------
# TAB 2: MACHINE LEARNING PREDICTOR
# ------------------------------------------
with tab_ml:
    st.header("🔮 StaySmart - Real-time Predictive Engine")
    st.markdown("Use this section as a consulting tool or SaaS product feature to predict the churn risk of new customers instantly.")

    if model is None:
        st.warning("⚠️ The model file `churn_model.pkl` was not found. Please ensure the trained model file exported from Google Colab is uploaded to this project folder.")
    else:
        col_ml_input, col_ml_result = st.columns(2)

        with col_ml_input:
            st.subheader("📥 Input New Customer Information")
            in_tenure = st.slider("Subscription Period (Tenure Months):", min_value=1, max_value=72, value=12)
            in_monthly = st.number_input("Monthly Charges ($):", min_value=10.0, max_value=150.0, value=65.0)
            in_total = st.number_input("Total Accumulated Charges ($):", min_value=10.0, max_value=8000.0, value=780.0)

            submit_prediction = st.button("🚀 Calculate Churn Risk")

        with col_ml_result:
            st.subheader("📊 Machine Learning Model Output")

            if submit_prediction:
                features = np.array([[in_tenure, in_monthly, in_total]])

                pred = model.predict(features)
                prob = model.predict_proba(features)[0][1]

                st.write("---")
                st.metric(label="Churn Probability", value=f"{prob * 100:.2f}%")

                if pred[0] == 1 or prob > 0.5:
                    st.error("🚨 **WARNING: This customer exhibits a HIGH risk of terminating services (Churn)!**")
                    st.markdown("""
                    **🎯 Automated Startup Recommendations:**
                    * Contact the customer via account managers within 48 hours.
                    * Offer an upgrade incentive to a 1-Year or 2-Year contract with a 15% discount promotional rate.
                    """)
                else:
                    st.success("✅ **SAFE STATUS: Customer is predicted to REMAIN loyal (Retain).**")
                    st.markdown("*The subscriber exhibits stable service behavior patterns matching historically retained profiles.*")
