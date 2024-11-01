import streamlit as st
import numpy as np
import pandas as pd
import numpy_financial as npf
from datetime import datetime
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Comprehensive SaaS Financial Model", layout="wide")
st.title("SAFVR Business Model Financial Projection")

# Sidebar for global parameters
with st.sidebar:
    st.header("Company Parameters")
    
    st.subheader("Founding Details")
    founding_year = st.number_input("Founding Year", value=datetime.now().year, min_value=2000, max_value=datetime.now().year + 5)
    project_years = st.number_input("Projection Period (years)", value=5, min_value=1, max_value=20)
    
    st.subheader("Financial Parameters")
    discount_rate = st.number_input("Discount Rate (%)", value=10.0, min_value=0.0, max_value=100.0) / 100
    tax_rate = st.number_input("Corporate Tax Rate (%)", value=25.0, min_value=0.0, max_value=100.0) / 100

    # Toggle for Investment Metrics
    st.subheader("Metrics Display Options")
    show_investment_metrics = st.checkbox("Show Investment-Related Metrics", value=False)

# Main input parameters
st.header("Product & Pricing")

col1, col2 = st.columns(2)
with col1:
    st.subheader("VR Content Library")
    vr_subscription_price = st.number_input("Annual Subscription Price per User ($)", value=32000.0)
    vr_initial_users = st.number_input("Initial Number of VR Users", value=10, min_value=0)
    vr_discount_rate = st.number_input("Discount for 100+ Users (%)", value=10.0, min_value=0.0, max_value=100.0) / 100

with col2:
    st.subheader("AI Digital Assistant")
    ai_subscription_price = st.number_input("Monthly Subscription Price per User ($)", value=50.0)
    ai_initial_users = st.number_input("Initial Number of AI Users", value=50, min_value=0)
    ai_discount_rate = st.number_input("Discount for 100+ Users (%)", value=5.0, min_value=0.0, max_value=100.0) / 100

st.header("Cost Structure and Growth")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Operating Costs")
    fixed_operating_costs = st.number_input("Annual Fixed Operating Costs ($)", value=500000.0, min_value=0.0)
    variable_operating_costs = st.number_input("Variable Operating Costs per User per Year ($)", value=100.0, min_value=0.0)
    support_costs = st.number_input("Support Costs per User per Year ($)", value=50.0, min_value=0.0)
    rnd_expenses = st.number_input("Annual R&D Expenses ($)", value=200000.0, min_value=0.0)

    st.subheader("Growth Assumptions")
    user_growth_rate = st.number_input("Annual User Growth Rate (%)", value=20.0, min_value=0.0, max_value=100.0) / 100
    churn_rate = st.number_input("Annual Churn Rate (%)", value=5.0, min_value=0.0, max_value=100.0) / 100
    customer_lifetime = 1 / churn_rate if churn_rate > 0 else project_years

with col2:
    st.subheader("Customer Acquisition")
    customer_acquisition_cost = st.number_input("Customer Acquisition Cost (CAC) per User ($)", value=500.0, min_value=0.0)
    marketing_expenses = st.number_input("Annual Marketing Expenses ($)", value=100000.0, min_value=0.0)
    channel_partner_share = st.number_input("Channel Partner Revenue Share (%)", value=10.0, min_value=0.0, max_value=100.0) / 100

    # Only show investment inputs if the toggle is selected
    if show_investment_metrics:
        st.subheader("Investment Rounds")
        pre_seed_round = st.number_input("Pre-Seed Round Investment ($)", value=200000.0, min_value=0.0)
        pre_seed_valuation = st.number_input("Pre-Seed Pre-Money Valuation ($)", value=1000000.0, min_value=0.0)
        seed_round = st.number_input("Seed Round Investment ($)", value=500000.0, min_value=0.0)
        seed_valuation = st.number_input("Seed Pre-Money Valuation ($)", value=3000000.0, min_value=0.0)
        series_a_round = st.number_input("Series A Investment ($)", value=2000000.0, min_value=0.0)
        series_a_valuation = st.number_input("Series A Pre-Money Valuation ($)", value=10000000.0, min_value=0.0)
        revenue_multiple = st.number_input("Revenue Multiple for Valuation", value=10.0, min_value=0.0)

    st.subheader("Expansion Revenue")
    expansion_rate = st.number_input("Expansion Revenue Rate (% of existing users)", value=10.0, min_value=0.0, max_value=100.0) / 100
    expansion_revenue_per_user = st.number_input("Expansion Revenue per User per Year ($)", value=1000.0, min_value=0.0)

def calculate_cap_table():
    # Initialize cap table
    cap_table = pd.DataFrame({
        'Round': ['Founders'],
        'Investment': [0],
        'Pre-Money Valuation': [0],
        'Post-Money Valuation': [0],
        'Equity (%)': [100.0]
    })
    
    # Pre-Seed Round
    pre_seed_post_money = pre_seed_valuation + pre_seed_round
    pre_seed_equity = (pre_seed_round / pre_seed_post_money) * 100
    founders_equity_post_pre_seed = cap_table.loc[0, 'Equity (%)'] * (pre_seed_valuation / pre_seed_post_money)
    
    cap_table = pd.concat([cap_table, pd.DataFrame({
        'Round': ['Pre-Seed Investors'],
        'Investment': [pre_seed_round],
        'Pre-Money Valuation': [pre_seed_valuation],
        'Post-Money Valuation': [pre_seed_post_money],
        'Equity (%)': [pre_seed_equity]
    })], ignore_index=True)
    
    cap_table.loc[0, 'Equity (%)'] = founders_equity_post_pre_seed
    
    # Seed Round
    seed_post_money = seed_valuation + seed_round
    seed_equity = (seed_round / seed_post_money) * 100
    founders_equity_post_seed = cap_table.loc[0, 'Equity (%)'] * (seed_valuation / seed_post_money)
    pre_seed_equity_post_seed = cap_table.loc[1, 'Equity (%)'] * (seed_valuation / seed_post_money)
    
    cap_table = pd.concat([cap_table, pd.DataFrame({
        'Round': ['Seed Investors'],
        'Investment': [seed_round],
        'Pre-Money Valuation': [seed_valuation],
        'Post-Money Valuation': [seed_post_money],
        'Equity (%)': [seed_equity]
    })], ignore_index=True)
    
    cap_table.loc[0, 'Equity (%)'] = founders_equity_post_seed
    cap_table.loc[1, 'Equity (%)'] = pre_seed_equity_post_seed
    
    # Series A Round
    series_a_post_money = series_a_valuation + series_a_round
    series_a_equity = (series_a_round / series_a_post_money) * 100
    founders_equity_post_series_a = cap_table.loc[0, 'Equity (%)'] * (series_a_valuation / series_a_post_money)
    pre_seed_equity_post_series_a = cap_table.loc[1, 'Equity (%)'] * (series_a_valuation / series_a_post_money)
    seed_equity_post_series_a = cap_table.loc[2, 'Equity (%)'] * (series_a_valuation / series_a_post_money)
    
    cap_table = pd.concat([cap_table, pd.DataFrame({
        'Round': ['Series A Investors'],
        'Investment': [series_a_round],
        'Pre-Money Valuation': [series_a_valuation],
        'Post-Money Valuation': [series_a_post_money],
        'Equity (%)': [series_a_equity]
    })], ignore_index=True)
    
    cap_table.loc[0, 'Equity (%)'] = founders_equity_post_series_a
    cap_table.loc[1, 'Equity (%)'] = pre_seed_equity_post_series_a
    cap_table.loc[2, 'Equity (%)'] = seed_equity_post_series_a
    
    return cap_table

def calculate_metrics():
    years = np.arange(1, project_years + 1)
    
    # User projections
    vr_users = [vr_initial_users]
    ai_users = [ai_initial_users]
    for _ in years[1:]:
        vr_users.append(vr_users[-1] * (1 + user_growth_rate))
        ai_users.append(ai_users[-1] * (1 + user_growth_rate))
    
    vr_users = np.array(vr_users)
    ai_users = np.array(ai_users)
    
    # Apply churn
    vr_users = vr_users * ((1 - churn_rate) ** (years - 1))
    ai_users = ai_users * ((1 - churn_rate) ** (years - 1))
    
    # Apply discounts
    vr_discount = np.where(vr_users >= 100, vr_discount_rate, 0)
    ai_discount = np.where(ai_users >= 100, ai_discount_rate, 0)
    
    vr_price = vr_subscription_price * (1 - vr_discount)
    ai_price = ai_subscription_price * 12 * (1 - ai_discount)
    
    # Revenue calculations
    vr_revenue = vr_users * vr_price
    ai_revenue = ai_users * ai_price
    total_revenue = vr_revenue + ai_revenue
    
    # Expansion revenue
    total_users = vr_users + ai_users
    expansion_revenue = total_users * expansion_rate * expansion_revenue_per_user
    total_revenue += expansion_revenue
    
    # Monthly and Annual Recurring Revenue
    mrr = total_revenue / 12
    arr = mrr * 12
    
    # Costs calculations
    variable_costs = (variable_operating_costs + support_costs) * total_users
    marketing_total = np.full(project_years, marketing_expenses)
    rnd_total = np.full(project_years, rnd_expenses)
    fixed_costs = np.full(project_years, fixed_operating_costs)
    channel_partner_costs = total_revenue * channel_partner_share
    cac_total = customer_acquisition_cost * (vr_users + ai_users)
    
    total_costs = variable_costs + fixed_costs + marketing_total + rnd_total + channel_partner_costs + cac_total
    gross_profit = total_revenue - variable_costs - channel_partner_costs
    ebitda = gross_profit - fixed_costs - marketing_total - rnd_total - cac_total
    net_income = ebitda * (1 - tax_rate)
    
    # Cash flows
    net_cash_flow = net_income
    cumulative_cash_flow = np.cumsum(net_cash_flow)
    
    # NPV and IRR
    initial_investment = -(fixed_operating_costs + rnd_expenses + marketing_expenses)
    cash_flows = np.concatenate(([initial_investment], net_cash_flow))
    npv = npf.npv(discount_rate, cash_flows)
    irr = npf.irr(cash_flows)
    
    # LTV and CAC
    avg_revenue_per_user = total_revenue / total_users
    ltv = avg_revenue_per_user * customer_lifetime * (1 - channel_partner_share)
    ltv_cac_ratio = ltv / customer_acquisition_cost if customer_acquisition_cost > 0 else np.nan
    
    # Additional Metrics
    gross_margin = (gross_profit / total_revenue) * 100
    revenue_growth_rate = np.insert((total_revenue[1:] - total_revenue[:-1]) / total_revenue[:-1], 0, np.nan) * 100
    customer_retention_rate = (1 - churn_rate) * 100
    burn_rate = -net_cash_flow
    months_of_runway = cumulative_cash_flow / burn_rate
    
    # CAC Payback Period
    cac_payback_period = customer_acquisition_cost / (avg_revenue_per_user * (1 - channel_partner_share)) if avg_revenue_per_user.any() > 0 else np.nan
    
    # Prepare DataFrame
    df = pd.DataFrame({
        'Year': years,
        'VR Users': vr_users,
        'AI Users': ai_users,
        'Total Users': total_users,
        'VR Revenue': vr_revenue,
        'AI Revenue': ai_revenue,
        'Expansion Revenue': expansion_revenue,
        'Total Revenue': total_revenue,
        'Revenue Growth Rate (%)': revenue_growth_rate,
        'Gross Profit': gross_profit,
        'Gross Margin (%)': gross_margin,
        'EBITDA': ebitda,
        'Net Income': net_income,
        'Net Cash Flow': net_cash_flow,
        'Cumulative Cash Flow': cumulative_cash_flow,
        'MRR': mrr,
        'ARR': arr,
        'Burn Rate': burn_rate,
        'Months of Runway': months_of_runway,
        'CAC Payback Period': cac_payback_period
    })
    
    return {
        'df': df,
        'npv': npv,
        'irr': irr,
        'ltv': ltv.mean(),
        'ltv_cac_ratio': ltv_cac_ratio.mean() if isinstance(ltv_cac_ratio, np.ndarray) else ltv_cac_ratio,
        'avg_revenue_per_user': avg_revenue_per_user
    }

# Calculate metrics
metrics = calculate_metrics()
df = metrics['df']

# Display results
st.header("Financial Projections")

# Key Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("NPV", f"${metrics['npv']:,.2f}")
    irr_display = f"{metrics['irr']*100:.2f}%" if not np.isnan(metrics['irr']) else "N/A"
    st.metric("IRR", irr_display)
    st.metric("Final Year Total Users", f"{df['Total Users'].iloc[-1]:,.0f}")

with col2:
    st.metric("Final Year ARR", f"${df['ARR'].iloc[-1]:,.2f}")
    st.metric("LTV/CAC Ratio", f"{metrics['ltv_cac_ratio']:.2f}")
    st.metric("Average LTV per User", f"${metrics['ltv']:,.2f}")

with col3:
    st.metric("Final Year Net Income", f"${df['Net Income'].iloc[-1]:,.2f}")
    st.metric("Gross Margin (Final Year)", f"{df['Gross Margin (%)'].iloc[-1]:.2f}%")
    st.metric("Customer Retention Rate", f"{(1 - churn_rate)*100:.2f}%")

# Visualizations
st.subheader("Revenue & ARR Projection")
chart_data = df[['Year', 'Total Revenue', 'ARR']]
st.line_chart(chart_data.set_index('Year'))

st.subheader("Gross Margin Over Time")
gross_margin_chart = df[['Year', 'Gross Margin (%)']]
st.line_chart(gross_margin_chart.set_index('Year'))

# Full metrics tables
st.subheader("Financial Statements")
st.dataframe(df[['Year', 'Total Revenue', 'Gross Profit', 'EBITDA', 'Net Income', 'Net Cash Flow', 'Cumulative Cash Flow']].style.format("${:,.2f}"))

st.subheader("User Metrics")
st.dataframe(df[['Year', 'Total Users', 'VR Users', 'AI Users', 'MRR', 'ARR']].style.format("${:,.2f}"))

# Only show investment metrics if the toggle is selected
if show_investment_metrics:
    # Cap Table Visualization
    st.header("Cap Table and Dilution")
    st.subheader("Cap Table After Series A")
    cap_table = calculate_cap_table()
    cap_table_display = cap_table[['Round', 'Investment', 'Equity (%)']]
    st.dataframe(cap_table_display.style.format({'Investment': '${:,.2f}', 'Equity (%)': '{:.2f}%'}))

    # Enhanced pie chart visualization
    pie_data = cap_table[['Round', 'Equity (%)']]
    pie_data = pie_data.rename(columns={'Equity (%)': 'Equity'})

    st.subheader("Equity Distribution") 
    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(pie_data['Equity'], 
                                      labels=pie_data['Round'],
                                      autopct='%1.1f%%',
                                      textprops={'fontsize': 12})
    plt.setp(autotexts, size=10, weight="bold")
    plt.setp(texts, size=12)
    ax.set_title("Equity Distribution After Series A", pad=20, size=14)
    st.pyplot(fig)
    plt.close()

# Add more comprehensive business model metrics
st.header("Advanced Business Model Metrics")

st.subheader("Churn and Retention Metrics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Churn Rate", f"{churn_rate*100:.2f}%")
    st.metric("Customer Lifetime", f"{customer_lifetime:.2f} years")

with col2:
    st.metric("Customer Retention Rate", f"{(1 - churn_rate)*100:.2f}%")
    st.metric("Average Revenue per User (ARPU)", f"${metrics['avg_revenue_per_user'].mean():,.2f}")

st.subheader("Growth Metrics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Revenue Growth Rate (Final Year)", f"{df['Revenue Growth Rate (%)'].iloc[-1]:.2f}%")
    st.metric("User Growth Rate (Annual)", f"{user_growth_rate*100:.2f}%")

with col2:
    st.metric("Monthly Recurring Revenue (MRR) Final Year", f"${df['MRR'].iloc[-1]:,.2f}")
    st.metric("Annual Recurring Revenue (ARR) Final Year", f"${df['ARR'].iloc[-1]:,.2f}")

st.subheader("Efficiency Metrics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Gross Margin (Final Year)", f"{df['Gross Margin (%)'].iloc[-1]:.2f}%")
    st.metric("Burn Rate (Final Year)", f"${df['Burn Rate'].iloc[-1]:,.2f}")

with col2:
    st.metric("Months of Runway (Final Year)", f"{df['Months of Runway'].iloc[-1]:,.2f}")
    cac_payback_period = df['CAC Payback Period'].iloc[-1]
    cac_payback_display = f"{cac_payback_period:.2f} years" if not np.isnan(cac_payback_period) else "N/A"
    st.metric("CAC Payback Period", cac_payback_display)
