import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="World Population Analysis Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin: 2rem 0 1rem 0;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('world_population.csv')
        return df
    except FileNotFoundError:
        st.error("Please upload the world_population.csv file to use this dashboard")
        return None

def create_pie_chart(df):
    df_clean = df.copy()
    df_clean.rename(columns={'2022 Population': 'Population_2022', 'Country/Territory': 'Country'}, inplace=True)
    top10 = df_clean.sort_values(by='Population_2022', ascending=False).head(10)
    top10['Percentage'] = top10['Population_2022'] / top10['Population_2022'].sum() * 100
    fig = px.pie(top10, values='Percentage', names='Country', title='Top 10 Most Populous Countries (2022)', color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=600)
    return fig

def create_density_plot(df):
    df_clean = df.copy()
    df_clean["log_pop_share"] = np.log10(df_clean["World Population Percentage"])
    fig = go.Figure()
    continents = df_clean['Continent'].unique()
    colors = px.colors.qualitative.Pastel1
    for i, continent in enumerate(continents):
        continent_data = df_clean[df_clean['Continent'] == continent]
        fig.add_trace(go.Violin(
            x=continent_data['log_pop_share'],
            name=continent,
            side='positive',
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)],
            opacity=0.6
        ))
    fig.update_layout(title="Density of World Population Share by Continent", xaxis_title="World Population Share (%) (log scale)", yaxis_title="Density", height=500)
    return fig

def create_bubble_chart(df):
    df_clean = df.copy()
    df_clean = df_clean[["Country/Territory", "Area (km²)", "2022 Population", "Continent"]]
    df_clean.columns = ["Country", "LandArea", "Population", "Continent"]
    df_clean["Population"] = pd.to_numeric(df_clean["Population"], errors='coerce')
    df_clean["LandArea"] = pd.to_numeric(df_clean["LandArea"], errors='coerce')
    df_clean = df_clean.dropna()
    fig = px.scatter(df_clean, x="LandArea", y="Population", color="Continent", size="Population", hover_name="Country", log_x=True, log_y=True, title="Relationship Between Population and Land Area")
    fig.update_layout(xaxis_title="Land Area (km²) - Log Scale", yaxis_title="Population - Log Scale", height=600)
    return fig

def create_growth_boxplot(df):
    continent_order = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    fig = px.box(df, x='Continent', y='Growth Rate', category_orders={'Continent': continent_order}, color='Continent', title="Population Growth Rate Distribution by Continent")
    fig.update_layout(xaxis_title="Continent", yaxis_title="Annual Growth Rate (%)", height=500, showlegend=False)
    return fig

def create_continent_share_bar(df):
    continent_share = df.groupby("Continent")["World Population Percentage"].sum().reset_index()
    continent_share.rename(columns={"World Population Percentage": "Total_World_Pop_Share"}, inplace=True)
    fig = px.bar(continent_share, x='Continent', y='Total_World_Pop_Share', title="World Population Share by Continent (2022)", color='Continent', text='Total_World_Pop_Share')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title="Continent", yaxis_title="World Population Share (%)", height=500, showlegend=False)
    return fig

def create_population_trends(df):
    countries = ["Brazil", "China", "India", "Russia", "United States"]
    df_filtered = df[df["Country/Territory"].isin(countries)]
    year_cols = [col for col in df_filtered.columns if "Population" in col and any(year in col for year in ['1970', '1980', '1990', '2000', '2010', '2015', '2020', '2022'])]
    df_melted = df_filtered.melt(id_vars=["Country/Territory"], value_vars=year_cols, var_name="Year", value_name="Population")
    df_melted.rename(columns={"Country/Territory": "Country"}, inplace=True)
    df_melted["Year"] = df_melted["Year"].str.extract(r"(\d{4})").astype(int)
    df_melted["Population"] = pd.to_numeric(df_melted["Population"], errors='coerce')
    df_melted = df_melted.dropna()
    df_melted.sort_values(["Country", "Year"], inplace=True)
    fig = px.line(df_melted, x="Year", y="Population", color="Country", title="Population Trends Over Time (Major Countries)", markers=True)
    fig.update_layout(xaxis_title="Year", yaxis_title="Population", height=600)
    return fig

def main():
    st.markdown('<h1 class="main-header">🌍 World Population Analysis Dashboard</h1>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Upload World Population CSV", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Data loaded successfully!")
        selected_plot = st.sidebar.selectbox("Choose Visualization:", [
            "🥧 Top 10 Most Populous Countries",
            "📈 Population Share Density by Continent",
            "🫧 Population vs Land Area",
            "📦 Growth Rate Distribution",
            "📊 Population Share by Continent",
            "📉 Population Trends Over Time"
        ])

        if selected_plot == "🥧 Top 10 Most Populous Countries":
            st.markdown('<h2 class="section-header">Top 10 Most Populous Countries (2022)</h2>', unsafe_allow_html=True)
            fig = create_pie_chart(df)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_plot == "📈 Population Share Density by Continent":
            st.markdown('<h2 class="section-header">Population Share Density by Continent</h2>', unsafe_allow_html=True)
            fig = create_density_plot(df)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_plot == "🫧 Population vs Land Area":
            st.markdown('<h2 class="section-header">Population vs Land Area</h2>', unsafe_allow_html=True)
            fig = create_bubble_chart(df)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_plot == "📦 Growth Rate Distribution":
            st.markdown('<h2 class="section-header">Population Growth Rate Distribution by Continent</h2>', unsafe_allow_html=True)
            fig = create_growth_boxplot(df)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_plot == "📊 Population Share by Continent":
            st.markdown('<h2 class="section-header">World Population Share by Continent</h2>', unsafe_allow_html=True)
            fig = create_continent_share_bar(df)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_plot == "📉 Population Trends Over Time":
            st.markdown('<h2 class="section-header">Population Trends Over Time</h2>', unsafe_allow_html=True)
            fig = create_population_trends(df)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
