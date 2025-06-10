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
    """Load and preprocess the world population dataset"""
    try:
        # You'll need to upload your CSV file to Streamlit
        df = pd.read_csv('world_population.csv')
        return df
    except FileNotFoundError:
        # Create sample data if file not found
        st.error("Please upload the world_population.csv file to use this dashboard")
        return None

def create_pie_chart(df):
    """Create pie chart for top 10 most populous countries"""
    df_clean = df.copy()
    df_clean.rename(columns={'2022 Population': 'Population_2022', 'Country/Territory': 'Country'}, inplace=True)
    
    top10 = df_clean.sort_values(by='Population_2022', ascending=False).head(10)
    top10['Percentage'] = top10['Population_2022'] / top10['Population_2022'].sum() * 100
    
    fig = px.pie(
        top10, 
        values='Percentage', 
        names='Country',
        title='Top 10 Most Populous Countries (2022)',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=600)
    return fig

def create_density_plot(df):
    """Create density plot for world population share by continent"""
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
    
    fig.update_layout(
        title="Density of World Population Share by Continent",
        xaxis_title="World Population Share (%) (log scale)",
        yaxis_title="Density",
        height=500
    )
    return fig

def create_bubble_chart(df):
    """Create bubble chart for population vs land area"""
    df_clean = df.copy()
    df_clean = df_clean[["Country/Territory", "Area (km²)", "2022 Population", "Continent"]]
    df_clean.columns = ["Country", "LandArea", "Population", "Continent"]
    
    # Clean data
    df_clean["Population"] = pd.to_numeric(df_clean["Population"], errors='coerce')
    df_clean["LandArea"] = pd.to_numeric(df_clean["LandArea"], errors='coerce')
    df_clean = df_clean.dropna()
    
    fig = px.scatter(
        df_clean,
        x="LandArea",
        y="Population",
        color="Continent",
        size="Population",
        hover_name="Country",
        log_x=True,
        log_y=True,
        title="Relationship Between Population and Land Area"
    )
    
    fig.update_layout(
        xaxis_title="Land Area (km²) - Log Scale",
        yaxis_title="Population - Log Scale",
        height=600
    )
    return fig

def create_growth_boxplot(df):
    """Create boxplot for population growth rate by continent"""
    continent_order = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    
    fig = px.box(
        df,
        x='Continent',
        y='Growth Rate',
        category_orders={'Continent': continent_order},
        color='Continent',
        title="Population Growth Rate Distribution by Continent"
    )
    
    fig.update_layout(
        xaxis_title="Continent",
        yaxis_title="Annual Growth Rate (%)",
        height=500,
        showlegend=False
    )
    return fig

def create_continent_share_bar(df):
    """Create bar chart for world population share by continent"""
    continent_share = df.groupby("Continent")["World Population Percentage"].sum().reset_index()
    continent_share.rename(columns={"World Population Percentage": "Total_World_Pop_Share"}, inplace=True)
    
    fig = px.bar(
        continent_share,
        x='Continent',
        y='Total_World_Pop_Share',
        title="World Population Share by Continent (2022)",
        color='Continent',
        text='Total_World_Pop_Share'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        xaxis_title="Continent",
        yaxis_title="World Population Share (%)",
        height=500,
        showlegend=False
    )
    return fig

def create_growth_density_scatter(df):
    """Create scatter plot for growth rate vs population density"""
    fig = px.scatter(
        df,
        x='Growth Rate',
        y='Density (per km²)',
        color='Continent',
        hover_name='Country/Territory',
        title="Growth Rate vs. Population Density",
        trendline="ols"
    )
    
    fig.update_layout(
        xaxis_title="Population Growth Rate (%)",
        yaxis_title="Population Density (people/km²)",
        height=500
    )
    return fig

def create_hexbin_plot(df):
    """Create hexbin plot for land area vs population density"""
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=['Area (km²)', 'Density (per km²)'])
    
    fig = px.density_heatmap(
        df_clean,
        x='Area (km²)',
        y='Density (per km²)',
        title="Land Area vs. Population Density (Heatmap)",
        log_x=True,
        nbinsx=40,
        nbinsy=40
    )
    
    fig.update_layout(
        xaxis_title="Land Area (km²) - Log Scale",
        yaxis_title="Population Density (people/km²)",
        height=500
    )
    return fig

def create_population_trends(df):
    """Create line chart for population trends by major countries"""
    countries = ["Brazil", "China", "India", "Russia", "United States"]
    df_filtered = df[df["Country/Territory"].isin(countries)]
    
    year_cols = [col for col in df_filtered.columns if "Population" in col and any(year in col for year in ['1970', '1980', '1990', '2000', '2010', '2015', '2020', '2022'])]
    
    df_melted = df_filtered.melt(
        id_vars=["Country/Territory"], 
        value_vars=year_cols,
        var_name="Year", 
        value_name="Population"
    )
    
    df_melted.rename(columns={"Country/Territory": "Country"}, inplace=True)
    df_melted["Year"] = df_melted["Year"].str.extract(r"(\d{4})").astype(int)
    df_melted["Population"] = pd.to_numeric(df_melted["Population"], errors='coerce')
    df_melted = df_melted.dropna()
    df_melted.sort_values(["Country", "Year"], inplace=True)
    
    fig = px.line(
        df_melted,
        x="Year",
        y="Population",
        color="Country",
        title="Population Trends Over Time (Major Countries)",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Population",
        height=600
    )
    return fig

def create_violin_plot(df):
    """Create violin plot for growth rate distribution by continent"""
    fig = px.violin(
        df,
        x="Continent",
        y="Growth Rate",
        color="Continent",
        title="Growth Rate Distribution by Continent (Violin Plot)"
    )
    
    fig.update_layout(
        xaxis_title="Continent",
        yaxis_title="Growth Rate",
        height=500,
        showlegend=False
    )
    return fig

def main():
    # Title
    st.markdown('<h1 class="main-header">🌍 World Population Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    ## Introduction
    
    The world's population is constantly changing due to factors such as birth rates, death rates, migration, and urbanization. 
    While some regions face rapid growth, others are dealing with aging populations, reshaping global social and economic dynamics. 
    These demographic changes bring both challenges and opportunities, making population analysis vital for sustainable development and effective policy-making.
    
    This dashboard explores the 11 most populous countries, each with over 100 million people, which significantly influence global demographic trends.
    Using interactive visualizations, we present 10 different analyses that reveal important patterns and enhance understanding of current demographic shifts.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Select a visualization to explore:")
    
    plot_options = [
        "📊 Dataset Overview",
        "🥧 Top 10 Most Populous Countries",
        "📈 Population Share Density by Continent", 
        "🫧 Population vs Land Area",
        "📦 Growth Rate Distribution",
        "📊 Population Share by Continent",
        "🎯 Growth Rate vs Density",
        "🔥 Land Area vs Density Heatmap",
        "📉 Population Trends Over Time",
        "🎻 Growth Rate Distribution (Violin)"
    ]
    
    selected_plot = st.sidebar.selectbox("Choose Visualization:", plot_options)
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload World Population CSV", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Data loaded successfully!")
        
        # Dataset overview
        if selected_plot == "📊 Dataset Overview":
            st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Countries", len(df))
            with col2:
                st.metric("Continents", df['Continent'].nunique())
            with col3:
                st.metric("Total Population 2022", f"{df['2022 Population'].sum():,.0f}")
            with col4:
                st.metric("Avg Growth Rate", f"{df['Growth Rate'].mean():.3f}%")
            
            st.subheader("Dataset Information")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write("**Author:** Sourav Banerjee - Senior Data Scientist at Launchpad (India)")
            st.write("**Source:** https://www.kaggle.com/datasets/iamsouravbanerjee/world-population-dataset")
            
            st.subheader("Column Descriptions")
            column_descriptions = {
                "Rank": "Rank by Population",
                "CCA3": "3 Digit Country/Territories Code",
                "Country/Territory": "Name of the Country/Territory",
                "Capital": "Name of the Capital",
                "Continent": "Name of the Continent",
                "2022 Population": "Population in 2022",
                "2020 Population": "Population in 2020",
                "2015 Population": "Population in 2015",
                "2010 Population": "Population in 2010",
                "2000 Population": "Population in 2000",
                "1990 Population": "Population in 1990",
                "1980 Population": "Population in 1980",
                "1970 Population": "Population in 1970",
                "Area (km²)": "Area size in square kilometers",
                "Density (per km²)": "Population Density per square kilometer",
                "Growth Rate": "Population Growth Rate",
                "World Population Percentage": "Population percentage of world total"
            }
            
            for col, desc in column_descriptions.items():
                if col in df.columns:
                    st.write(f"**{col}:** {desc}")
            
            st.subheader("Sample Data")
            st.dataframe(df.head(10))
            
            st.subheader("Statistical Summary")
            st.dataframe(df.describe())
        
        # Individual visualizations
        elif selected_plot == "🥧 Top 10 Most Populous Countries":
            st.markdown('<h2 class="section-header">Top 10 Most Populous Countries (2022)</h2>', unsafe_allow_html=True)
            fig = create_pie_chart(df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>Key Insights:</h4>
            <ul>
                <li>China and India dominate with over 62% of the top 10 countries' total population</li>
                <li>Asia represents 5 out of 10 countries in this ranking</li>
                <li>The United States is the only North American country in the top 5</li>
                <li>Population concentration shows significant global inequality in distribution</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif selected_plot == "📈 Population Share Density by Continent":
            st.markdown('<h2 class="section-header">Population Share Density by Continent</h2>', unsafe_allow_html=True)
            fig = create_density_plot(df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>Key Insights:</h4>
            <ul>
                <li>Asia shows the widest distribution, indicating both very large and small countries</li>
                <li>Most countries have relatively small population shares globally</li>
                <li>Oceania countries consistently have very small population shares</li>
                <li>The logarithmic scale reveals the extreme range in country sizes</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif selected_plot == "🫧 Population vs Land Area":
            st.markdown('<h2 class="section-header">Population vs Land Area Relationship</h2>', unsafe_allow_html=True)
            fig = create_bubble_chart(df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>Key Insights:</h4>
            <ul>
                <li>Positive correlation between land area and population, but with significant variation</li>
                <li>Some small countries have exceptionally high populations (high density)</li>
                <li>Some large countries have relatively low populations (low density)</li>
                <li>Geographic and economic factors influence this relationship significantly</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif selected_plot == "📦 Growth Rate Distribution":
            st.markdown('<h2 class="section-header">Population Growth Rate Distribution by Continent</h2>', unsafe_allow_html=True)
            fig = create_growth_boxplot(df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>Key Insights:</h4>
            <ul>
                <li>Africa has the highest median growth rate (~1.024%)</li>
                <li>Europe shows the lowest growth rates, often below 1%</li>
                <li>Asia displays wide variation due to diverse economic conditions</li>
                <li>North and South America show moderate, stable growth patterns</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif selected_plot == "📊 Population Share by Continent":
            st.markdown('<h2 class="section-header">World Population Share by Continent</h2>', unsafe_allow_html=True)
            fig = create_continent_share_bar(df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>Key Insights:</h4>
            <ul>
                <li>Asia dominates with ~58.7% of world population</li>
                <li>Africa is second with ~17.8%, showing rapid growth potential</li>
                <li>Europe accounts for ~9.6% but is experiencing declining growth</li>
                <li>Oceania has less than 0.6% of world population</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif selected_plot == "🎯 Growth Rate vs Density":
            st.markdown('<h2 class="section-header">Growth Rate vs Population Density</h2>', unsafe_allow_html=True)
            fig = create_growth_density_scatter(df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>Key Insights:</h4>
            <ul>
                <li>The relationship between growth rate and density varies by continent</li>
                <li>Some high-density areas show lower growth rates</li>
                <li>Outliers indicate countries with unique demographic situations</li>
                <li>The trend line helps identify general patterns across regions</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif selected_plot == "🔥 Land Area vs Density Heatmap":
            st.markdown('<h2 class="section-header">Land Area vs Population Density Heatmap</h2>', unsafe_allow_html=True)
            fig = create_hexbin_plot(df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>Key Insights:</h4>
            <ul>
                <li>Most countries cluster in specific land area and density ranges</li>
                <li>The heatmap reveals data concentration zones</li>
                <li>Logarithmic scaling helps visualize extreme values</li>
                <li>Useful for identifying typical patterns in global demographics</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif selected_plot == "📉 Population Trends Over Time":
            st.markdown('<h2 class="section-header">Population Trends Over Time</h2>', unsafe_allow_html=True)
            fig = create_population_trends(df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>Key Insights:</h4>
            <ul>
                <li>China and India show dramatic population growth over decades</li>
                <li>The US demonstrates steady, consistent growth</li>
                <li>Russia shows population decline in recent years</li>
                <li>Brazil exhibits strong growth but at a moderating pace</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif selected_plot == "🎻 Growth Rate Distribution (Violin)":
            st.markdown('<h2 class="section-header">Growth Rate Distribution by Continental Regions</h2>', unsafe_allow_html=True)
            fig = create_violin_plot(df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>Key Insights:</h4>
            <ul>
                <li>Violin plots show the full distribution shape for each continent</li>
                <li>Africa displays the most variation in growth rates</li>
                <li>Europe shows a narrow distribution around low growth rates</li>
                <li>Each continent has distinct demographic patterns and outliers</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.warning("Please upload the world_population.csv file to begin the analysis.")
        st.info("You can download the dataset from: https://www.kaggle.com/datasets/iamsouravbanerjee/world-population-dataset")
        
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>World Population Analysis Dashboard | Built with Streamlit 🚀</p>
        <p>Data Source: Kaggle - World Population Dataset by Sourav Banerjee</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


