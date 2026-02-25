import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD DATA AND MODEL
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/telco_cleaned_with_clusters.csv')
    return df

@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    return model

df = load_data()
model = load_model()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Customer Segments", "Churn Analysis", "CLV Analysis", "Predict Churn"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Info**")
st.sidebar.markdown(f"Total Customers: **{len(df):,}**")
st.sidebar.markdown(f"Churn Rate: **{df['Churn'].mean()*100:.1f}%**")
st.sidebar.markdown(f"Avg CLV: **${df['CLV'].mean():,.0f}**")

# ─────────────────────────────────────────
# SEGMENT LABELS
# ─────────────────────────────────────────
# Update this mapping based on YOUR cluster profile output
segment_map = {
    0: 'Champions',
    1: 'At Risk',
    2: 'Potential Loyalists',
    3: 'Dormant'
}
df['Segment'] = df['Cluster'].map(segment_map)

# ─────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────
if page == "Overview":
    st.title("📊 Customer Segmentation & Retention Dashboard")
    st.markdown("**Telco Customer Churn Analysis**")
    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Customers",
            value=f"{len(df):,}"
        )
    with col2:
        st.metric(
            label="Overall Churn Rate",
            value=f"{df['Churn'].mean()*100:.1f}%"
        )
    with col3:
        st.metric(
            label="Average CLV",
            value=f"${df['CLV'].mean():,.0f}"
        )
    with col4:
        at_risk = len(df[df['Segment'] == 'At Risk'])
        st.metric(
            label="At Risk Customers",
            value=f"{at_risk:,}"
        )

    st.markdown("---")

    # Two column layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Distribution by Segment")
        seg_counts = df['Segment'].value_counts()
        fig = px.pie(
            values=seg_counts.values,
            names=seg_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn Rate by Segment")
        churn_by_seg = df.groupby('Segment')['Churn'].mean().reset_index()
        churn_by_seg['Churn'] = (churn_by_seg['Churn'] * 100).round(1)
        churn_by_seg.columns = ['Segment', 'Churn Rate (%)']
        fig2 = px.bar(
            churn_by_seg,
            x='Segment',
            y='Churn Rate (%)',
            color='Segment',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 2 — CUSTOMER SEGMENTS
# ─────────────────────────────────────────
elif page == "Customer Segments":
    st.title("👥 Customer Segments")
    st.markdown("---")

    # Segment summary table
    st.subheader("Segment Profile Summary")
    summary = df.groupby('Segment').agg(
        Customer_Count=('Churn', 'count'),
        Avg_Tenure=('tenure', 'mean'),
        Avg_Monthly_Charges=('MonthlyCharges', 'mean'),
        Avg_CLV=('CLV', 'mean'),
        Churn_Rate=('Churn', 'mean')
    ).round(2)
    summary['Churn_Rate'] = (summary['Churn_Rate'] * 100).round(1)
    summary.columns = ['Count', 'Avg Tenure (months)',
                       'Avg Monthly Charges ($)', 'Avg CLV ($)', 'Churn Rate (%)']
    st.dataframe(summary, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tenure Distribution by Segment")
        fig = px.box(
            df,
            x='Segment',
            y='tenure',
            color='Segment',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Monthly Charges by Segment")
        fig2 = px.box(
            df,
            x='Segment',
            y='MonthlyCharges',
            color='Segment',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 3 — CHURN ANALYSIS
# ─────────────────────────────────────────
elif page == "Churn Analysis":
    st.title("⚠️ Churn Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn vs Retained")
        churn_counts = df['Churn'].value_counts()
        fig = px.pie(
            values=churn_counts.values,
            names=['Retained', 'Churned'],
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn by Tenure Group")
        df['Tenure_Group'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['0-12 months', '12-24 months', '24-48 months', '48-72 months']
        )
        churn_tenure = df.groupby('Tenure_Group')['Churn'].mean().reset_index()
        churn_tenure['Churn'] = (churn_tenure['Churn'] * 100).round(1)
        fig2 = px.bar(
            churn_tenure,
            x='Tenure_Group',
            y='Churn',
            labels={'Churn': 'Churn Rate (%)'},
            color='Churn',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Business Recommendations
    st.markdown("---")
    st.subheader("💡 Business Recommendations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.error("🚨 Who needs retention offers?")
        at_risk_clv = df[df['Segment'] == 'At Risk']['CLV'].mean()
        at_risk_count = len(df[df['Segment'] == 'At Risk'])
        st.markdown(f"""
        **Segment:** At Risk
        **Count:** {at_risk_count:,} customers
        **Avg CLV:** ${at_risk_clv:,.0f}
        **Action:** Immediate discount or contract upgrade offer
        """)

    with col2:
        st.success("🌟 Who gets early feature access?")
        champ_clv = df[df['Segment'] == 'Champions']['CLV'].mean()
        champ_count = len(df[df['Segment'] == 'Champions'])
        st.markdown(f"""
        **Segment:** Champions
        **Count:** {champ_count:,} customers
        **Avg CLV:** ${champ_clv:,.0f}
        **Action:** Loyalty rewards and beta feature access
        """)

    with col3:
        st.warning("⛔ Who should we stop spending on?")
        dormant_clv = df[df['Segment'] == 'Dormant']['CLV'].mean()
        dormant_count = len(df[df['Segment'] == 'Dormant'])
        st.markdown(f"""
        **Segment:** Dormant
        **Count:** {dormant_count:,} customers
        **Avg CLV:** ${dormant_clv:,.0f}
        **Action:** No retention spend — focus on root cause analysis
        """)

# ─────────────────────────────────────────
# PAGE 4 — CLV ANALYSIS
# ─────────────────────────────────────────
elif page == "CLV Analysis":
    st.title("💰 Customer Lifetime Value Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average CLV by Segment")
        clv_seg = df.groupby('Segment')['CLV'].mean().reset_index()
        fig = px.bar(
            clv_seg,
            x='Segment',
            y='CLV',
            color='Segment',
            labels={'CLV': 'Average CLV ($)'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Total Revenue at Risk by Segment")
        revenue_risk = df[df['Churn'] == 1].groupby('Segment')['CLV'].sum().reset_index()
        fig2 = px.bar(
            revenue_risk,
            x='Segment',
            y='CLV',
            color='Segment',
            labels={'CLV': 'Total CLV at Risk ($)'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig2, use_container_width=True)

    # CLV Distribution
    st.subheader("CLV Distribution Across All Customers")
    fig3 = px.histogram(
        df,
        x='CLV',
        color='Segment',
        nbins=50,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 5 — PREDICT CHURN
# ─────────────────────────────────────────
elif page == "Predict Churn":
    st.title("🔮 Predict Customer Churn")
    st.markdown("Enter a customer's details below to predict their churn probability.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account Info")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
        total_charges = st.number_input(
            "Total Charges ($)",
            value=float(monthly_charges * tenure)
        )
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])

    with col2:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", [0, 1])
        multiple_lines = st.selectbox("Multiple Lines", [0, 1])
        internet_service_fiber = st.selectbox("Internet: Fiber Optic", [0, 1])
        internet_service_no = st.selectbox("No Internet Service", [0, 1])
        online_security = st.selectbox("Online Security", [0, 1])
        online_backup = st.selectbox("Online Backup", [0, 1])

    with col3:
        st.subheader("Contract & Billing")
        contract_one_year = st.selectbox("Contract: One Year", [0, 1])
        contract_two_year = st.selectbox("Contract: Two Year", [0, 1])
        paperless_billing = st.selectbox("Paperless Billing", [0, 1])
        device_protection = st.selectbox("Device Protection", [0, 1])
        tech_support = st.selectbox("Tech Support", [0, 1])
        streaming_tv = st.selectbox("Streaming TV", [0, 1])

    st.markdown("---")

    # Build input matching your model's feature columns
    input_data = pd.DataFrame([{
        'gender': 1,
        'SeniorCitizen': senior_citizen,
        'Partner': 0,
        'Dependents': 0,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': 0,
        'PaperlessBilling': paperless_billing,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'InternetService_Fiber optic': internet_service_fiber,
        'InternetService_No': internet_service_no,
        'Contract_One year': contract_one_year,
        'Contract_Two year': contract_two_year,
        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 0,
        'PaymentMethod_Mailed check': 0
    }])

    # Ensure column order matches training data
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        probability = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if probability >= 0.7:
                st.error(f"🚨 HIGH CHURN RISK: {probability*100:.1f}%")
                st.markdown("**Recommended Action:** Immediate retention offer required.")
            elif probability >= 0.4:
                st.warning(f"⚠️ MEDIUM CHURN RISK: {probability*100:.1f}%")
                st.markdown("**Recommended Action:** Monitor closely, consider proactive outreach.")
            else:
                st.success(f"✅ LOW CHURN RISK: {probability*100:.1f}%")
                st.markdown("**Recommended Action:** Maintain engagement, consider upsell opportunities.")

        with col2:
            estimated_clv = monthly_charges * tenure
            st.metric("Estimated CLV", f"${estimated_clv:,.0f}")
            st.metric("Churn Probability", f"{probability*100:.1f}%")
            st.metric("Prediction", "Will Churn" if prediction == 1 else "Will Stay")