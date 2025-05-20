import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ---------- Data Preprocessing ----------
@st.cache_data
def load_and_process_data():
    df = pd.read_csv("/Users/Div/Desktop/tourism_dataset_csv.csv")

    # Convert 'Accommodation_Available' to binary
    df['Accommodation_Available'] = df['Accommodation_Available'].map({'Yes': 1, 'No': 0})

    # Compute Revenue per Visitor
    df['Revenue_per_visitor'] = df['Revenue'] / df['Visitors']

    # Compute CPS (Category Potential Score)
    grouped = df.groupby(['Country', 'Category']).agg({
        'Revenue_per_visitor': 'mean',
        'Rating': 'mean',
        'Accommodation_Available': 'mean'
    }).reset_index()

    for col in ['Revenue_per_visitor', 'Rating', 'Accommodation_Available']:
        grouped[col] = (grouped[col] - grouped[col].min()) / (grouped[col].max() - grouped[col].min())

    grouped['CPS'] = grouped[['Revenue_per_visitor', 'Rating', 'Accommodation_Available']].mean(axis=1)
    df = df.merge(grouped[['Country', 'Category', 'CPS']], on=['Country', 'Category'], how='left')

    # Compute TDI (Tourism Diversity Index)
    def calculate_tdi(data):
        tdi_scores = {}
        for country in data['Country'].unique():
            subset = data[data['Country'] == country]
            proportions = subset['Category'].value_counts(normalize=True)
            si_squared = (proportions**2).sum()
            tdi_scores[country] = 1 - si_squared
        return pd.DataFrame.from_dict(tdi_scores, orient='index', columns=['TDI'])

    tdi_df = calculate_tdi(df).reset_index().rename(columns={'index': 'Country'})
    df = df.merge(tdi_df, on='Country', how='left')

    return df

df = load_and_process_data()

# ---------- UI Layout ----------
st.title("🌍 Tourism Insights Dashboard")

# Sidebar filters
st.sidebar.header("Filter Options")
countries = st.sidebar.multiselect("Select Country", df["Country"].unique(), default=df["Country"].unique())
categories = st.sidebar.multiselect("Select Category", df["Category"].unique(), default=df["Category"].unique())

filtered_df = df[(df["Country"].isin(countries)) & (df["Category"].isin(categories))]

# ---------- Section 1: Overview ----------
st.subheader("1. Overview Statistics")
st.write("**Total Records:**", filtered_df.shape[0])
st.write("**Average Revenue:**", round(filtered_df["Revenue"].mean(), 2))
st.write("**Average Rating:**", round(filtered_df["Rating"].mean(), 2))

# ---------- Section 2: CPS Histogram ----------
if "CPS" in filtered_df.columns and not filtered_df["CPS"].isnull().all():
    st.subheader("2. Distribution of Category Potential Score (CPS)")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["CPS"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)
else:
    st.warning("CPS data not available for current filter selection.")

# ---------- Section 3: Heatmap ----------
if set(["Country", "Category", "CPS"]).issubset(filtered_df.columns) and not filtered_df.empty:
    st.subheader("3. CPS Heatmap by Country and Category")
    pivot_data = filtered_df.pivot_table(index="Country", columns="Category", values="CPS", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, ax=ax)
    st.pyplot(fig)

# ---------- Section 4: TDI Map ----------
if set(["Country", "TDI"]).issubset(df.columns):
    st.subheader("4. Tourism Diversity Index (TDI) Map")
    tdi_map = df[["Country", "TDI"]].drop_duplicates()
    fig = px.choropleth(
        tdi_map,
        locations="Country",
        locationmode="country names",
        color="TDI",
        hover_name="Country",
        color_continuous_scale="YlGnBu",
        title="TDI by Country"
    )
    st.plotly_chart(fig)