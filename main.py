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
    .stSelectbox > div > div > select {
        color: #1f77b4;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample world population dataset"""
    # Sample data - replace with your actual CSV loading
    # df = pd.read_csv('world_population.csv')
    
    # For demo purposes, creating sample data
    np.random.seed(42)
    countries = ['China', 'India', 'United States', 'Indonesia', 'Pakistan', 
                'Brazil', 'Nigeria', 'Bangladesh', 'Russia', 'Mexico',
                'Japan', 'Ethiopia', 'Philippines', 'Egypt', 'Vietnam']
    continents = ['Asia', 'Asia', 'North America', 'Asia', 'Asia',
                 'South America', 'Africa', 'Asia', 'Europe', 'North America',
                 'Asia', 'Africa', 'Asia', 'Africa', 'Asia']
    
    data = {
        'Country/Territory': countries,
        'Continent': continents,
        '2022 Population': [1425887337, 1417173173, 338289857, 275501339, 235824862,
                           215313498, 218541212, 171186372, 144713314, 127504125,
                           125124989, 123379924, 111046913, 110990103, 97338583],
        '2020 Population': [1439323776, 1380004385, 331002651, 273523615, 220892340,
                           212559417, 206139589, 164689383, 145934462, 128932753,
                           126476461, 114963588, 109581078, 102334404, 97338579],
        '2015 Population': [1397715000, 1311050527, 321418820, 258705000, 199085847,
                           207847528, 182201962, 161356039, 146267288, 127017224,
                           127094745, 109224559, 100699395, 91508084, 93447601],
        '2010 Population': [1348191368, 1240613620, 309321666, 244016173, 184404791,
                           201103330, 158503197, 152518015, 142958164, 114255555,
                           128105431, 94100756, 94013200, 84474427, 87967653],
        '2000 Population': [1290550765, 1053898818, 282338631, 214072421, 154369924,
                           176319621, 122851984, 129194327, 146844839, 100569305,
                           126843424, 65970761, 81159644, 68831314, 78773873],
        '1990 Population': [1176883672, 873277798, 253339662, 184346117, 115414069,
                           152227494, 97552411, 111455185, 148394216, 86154887,
                           123537399, 50974718, 68100000, 57214482, 68809344],
        '1980 Population': [1000072000, 696783517, 231664058, 150958653, 84493700,
                           122958132, 73698098, 92085055, 139390000, 69655120,
                           117902068, 38113879, 48317000, 44928591, 54053703],
        '1970 Population': [822534450, 555189649, 209896021, 115124000, 60671000,
                           96369875, 56473000, 70066000, 130079210, 52775000,
                           104921688, 29966000, 37540000, 35959000, 43407000],
        'Area (km²)': [9596960, 3287263, 9833517, 1904569, 881913, 8514877,
                      923768, 147570, 17098242, 1964375, 377975, 1104300,
                      300000, 1001449, 331212],
        'Density (per km²)': [148.6, 431.1, 34.4, 144.7, 267.4, 25.3,
                             236.5, 1159.7, 8.5, 64.9, 331.2, 111.8,
                             370.2, 110.8, 294.1],
        'Growth Rate': [0.39, 0.68, 0.59, 0.87, 1.93, 0.65, 2.41, 0.98,
                       -0.43, -0.18, -0.53, 2.57, 1.69, 1.99, 0.49],
        'World Population Percentage': [17.72, 17.65, 4.21, 3.43, 2.93, 2.68, 2.72,
                                       2.13, 1.80, 1.58, 1.56, 1.54, 1.38, 1.38, 1.21]
    }
    
    df = pd.DataFrame(data)
    return df

def create_pie_chart(df, top_n=10):
    """Create pie chart for top N most populous countries"""
    df_clean = df.copy()
    df_clean.rename(columns={'2022 Population': 'Population_2022', 'Country/Territory': 'Country'}, inplace=True)
    
    top_countries = df_clean.sort_values(by='Population_2022', ascending=False).head(top_n)
    top_countries['Percentage'] = top_countries['Population_2022'] / top_countries['Population_2022'].sum() * 100
    
    fig = px.pie(
        top_countries, 
        values='Percentage', 
        names='Country',
        title=f'Top {top_n} Most Populous Countries (2022)',
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.3  # Donut chart style
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=600, showlegend=True)
    return fig

def create_continent_share_bar(df, show_values=True):
    """Create bar chart for world population share by continent"""
    continent_share = df.groupby("Continent")["World Population Percentage"].sum().reset_index()
    continent_share.rename(columns={"World Population Percentage": "Total_World_Pop_Share"}, inplace=True)
    continent_share = continent_share.sort_values('Total_World_Pop_Share', ascending=False)
    
    fig = px.bar(
        continent_share,
        x='Continent',
        y='Total_World_Pop_Share',
        title="World Population Share by Continent (2022)",
        color='Continent',
        text='Total_World_Pop_Share' if show_values else None,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    if show_values:
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    
    fig.update_layout(
        xaxis_title="Continent",
        yaxis_title="World Population Share (%)",
        height=500,
        showlegend=False
    )
    return fig

def create_density_plot(df, continents_selected=None):
    """Create density plot for world population share by continent"""
    df_clean = df.copy()
    
    if continents_selected:
        df_clean = df_clean[df_clean['Continent'].isin(continents_selected)]
    
    df_clean["log_pop_share"] = np.log10(df_clean["World Population Percentage"])
    
    fig = go.Figure()
    
    continents = df_clean['Continent'].unique()
    colors = px.colors.qualitative.Set2
    
    for i, continent in enumerate(continents):
        continent_data = df_clean[df_clean['Continent'] == continent]
        
        fig.add_trace(go.Violin(
            x=continent_data['log_pop_share'],
            name=continent,
            side='positive',
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)],
            opacity=0.7
        ))
    
    fig.update_layout(
        title="Population Share Density by Continent (Log Scale)",
        xaxis_title="World Population Share (%) - Log Scale",
        yaxis_title="Density",
        height=500
    )
    return fig

def create_growth_boxplot(df, continents_selected=None):
    """Create boxplot for population growth rate by continent"""
    df_filtered = df.copy()
    
    if continents_selected:
        df_filtered = df_filtered[df_filtered['Continent'].isin(continents_selected)]
    
    continent_order = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    available_continents = [c for c in continent_order if c in df_filtered['Continent'].unique()]
    
    fig = px.box(
        df_filtered,
        x='Continent',
        y='Growth Rate',
        category_orders={'Continent': available_continents},
        color='Continent',
        title="Population Growth Rate Distribution by Continent",
        points="all"  # Show all data points
    )
    
    fig.update_layout(
        xaxis_title="Continent",
        yaxis_title="Annual Growth Rate (%)",
        height=500,
        showlegend=False
    )
    return fig

def create_population_trends(df, countries_selected=None, year_range=None):
    """Create line chart for population trends by selected countries"""
    if not countries_selected:
        countries_selected = ["China", "India", "United States", "Brazil", "Russia"]
    
    df_filtered = df[df["Country/Territory"].isin(countries_selected)]
    
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
    
    if year_range:
        df_melted = df_melted[(df_melted["Year"] >= year_range[0]) & (df_melted["Year"] <= year_range[1])]
    
    df_melted.sort_values(["Country", "Year"], inplace=True)
    
    fig = px.line(
        df_melted,
        x="Year",
        y="Population",
        color="Country",
        title="Population Trends Over Time",
        markers=True,
        line_shape='spline'
    )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Population",
        height=600,
        hovermode='x unified'
    )
    return fig

def main():
    # Load data
    df = load_sample_data()
    
    # Title
    st.markdown('<h1 class="main-header">🌍 Interactive World Population Dashboard</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    ## 📊 Welcome to the Interactive Population Explorer
    
    Explore global population dynamics through interactive visualizations. Use the controls in the sidebar to customize your analysis 
    and discover insights about demographic trends, growth patterns, and regional distributions.
    
    **Available Visualizations:**
    - 🥧 Top Most Populous Countries
    - 📊 Population Share by Continent  
    - 📈 Population Share Density Analysis
    - 📦 Growth Rate Distribution
    - 📉 Population Trends Over Time
    """)
    
    # Sidebar for interactive controls
    st.sidebar.title("🎛️ Interactive Controls")
    st.sidebar.markdown("---")
    
    # Visualization selector
    st.sidebar.subheader("📈 Select Visualization")
    plot_options = [
        "🥧 Top Most Populous Countries",
        "📊 Population Share by Continent",
        "📈 Population Share Density by Continent", 
        "📦 Growth Rate Distribution",
        "📉 Population Trends Over Time"
    ]
    
    selected_plot = st.sidebar.selectbox("Choose Visualization:", plot_options)
    
    st.sidebar.markdown("---")
    
    # Dataset overview in sidebar
    st.sidebar.subheader("📋 Dataset Overview")
    st.sidebar.metric("Total Countries", len(df))
    st.sidebar.metric("Continents", df['Continent'].nunique()) 
    st.sidebar.metric("Total Population 2022", f"{df['2022 Population'].sum():,.0f}")
    st.sidebar.metric("Avg Growth Rate", f"{df['Growth Rate'].mean():.2f}%")
    
    st.sidebar.markdown("---")
    
    # Interactive widgets based on selected plot
    if selected_plot == "🥧 Top Most Populous Countries":
        st.markdown('<h2 class="section-header">🥧 Top Most Populous Countries</h2>', unsafe_allow_html=True)
        
        # Interactive controls
        col1, col2 = st.columns([1, 3])
        with col1:
            top_n = st.slider("Number of countries to show:", min_value=5, max_value=15, value=10, step=1)
        
        fig = create_pie_chart(df, top_n)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        if st.checkbox("Show data table"):
            top_countries = df.nlargest(top_n, '2022 Population')[['Country/Territory', '2022 Population', 'Continent', 'World Population Percentage']]
            st.dataframe(top_countries, use_container_width=True)
    
    elif selected_plot == "📊 Population Share by Continent":
        st.markdown('<h2 class="section-header">📊 Population Share by Continent</h2>', unsafe_allow_html=True)
        
        # Interactive controls
        show_values = st.checkbox("Show percentage values on bars", value=True)
        
        fig = create_continent_share_bar(df, show_values)
        st.plotly_chart(fig, use_container_width=True)
        
        # Continental breakdown
        if st.checkbox("Show continental breakdown"):
            continent_data = df.groupby("Continent").agg({
                'Country/Territory': 'count',
                '2022 Population': 'sum',
                'World Population Percentage': 'sum',
                'Growth Rate': 'mean'
            }).round(2)
            continent_data.columns = ['Countries', 'Total Population', 'World Share (%)', 'Avg Growth Rate (%)']
            st.dataframe(continent_data, use_container_width=True)
    
    elif selected_plot == "📈 Population Share Density by Continent":
        st.markdown('<h2 class="section-header">📈 Population Share Density by Continent</h2>', unsafe_allow_html=True)
        
        # Interactive controls
        available_continents = df['Continent'].unique().tolist()
        selected_continents = st.multiselect(
            "Select continents to analyze:",
            available_continents,
            default=available_continents
        )
        
        if selected_continents:
            fig = create_density_plot(df, selected_continents)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            if st.checkbox("Show statistical summary"):
                stats = df[df['Continent'].isin(selected_continents)].groupby('Continent')['World Population Percentage'].describe()
                st.dataframe(stats, use_container_width=True)
        else:
            st.warning("Please select at least one continent to display the analysis.")
    
    elif selected_plot == "📦 Growth Rate Distribution":
        st.markdown('<h2 class="section-header">📦 Growth Rate Distribution</h2>', unsafe_allow_html=True)
        
        # Interactive controls
        available_continents = df['Continent'].unique().tolist()
        selected_continents = st.multiselect(
            "Select continents for comparison:",
            available_continents,
            default=available_continents
        )
        
        if selected_continents:
            fig = create_growth_boxplot(df, selected_continents)
            st.plotly_chart(fig, use_container_width=True)
            
            # Growth rate insights
            col1, col2 = st.columns(2)
            with col1:
                if st.checkbox("Show highest growth countries"):
                    top_growth = df.nlargest(5, 'Growth Rate')[['Country/Territory', 'Continent', 'Growth Rate']]
                    st.write("**Top 5 Fastest Growing:**")
                    st.dataframe(top_growth, use_container_width=True)
            
            with col2:
                if st.checkbox("Show lowest growth countries"):
                    low_growth = df.nsmallest(5, 'Growth Rate')[['Country/Territory', 'Continent', 'Growth Rate']]
                    st.write("**Top 5 Slowest Growing:**")
                    st.dataframe(low_growth, use_container_width=True)
        else:
            st.warning("Please select at least one continent to display the analysis.")
    
    elif selected_plot == "📉 Population Trends Over Time":
        st.markdown('<h2 class="section-header">📉 Population Trends Over Time</h2>', unsafe_allow_html=True)
        
        # Interactive controls
        available_countries = df['Country/Territory'].unique().tolist()
        default_countries = ["China", "India", "United States", "Brazil", "Russia"]
        available_defaults = [c for c in default_countries if c in available_countries]
        
        selected_countries = st.multiselect(
            "Select countries to compare:",
            available_countries,
            default=available_defaults if available_defaults else available_countries[:5]
        )
        
        # Year range selector
        year_range = st.slider(
            "Select year range:",
            min_value=1970,
            max_value=2022,
            value=(1990, 2022),
            step=10
        )
        
        if selected_countries:
            fig = create_population_trends(df, selected_countries, year_range)
            st.plotly_chart(fig, use_container_width=True)
            
            # Population comparison table
            if st.checkbox("Show population comparison table"):
                trend_data = df[df['Country/Territory'].isin(selected_countries)]
                year_cols = ['Country/Territory'] + [col for col in df.columns if 'Population' in col and any(str(year) in col for year in range(year_range[0], year_range[1]+1, 10))]
                comparison_table = trend_data[year_cols].set_index('Country/Territory')
                st.dataframe(comparison_table, use_container_width=True)
        else:
            st.warning("Please select at least one country to display the trends.")
    
    # Key insights section
    st.markdown("---")
    st.markdown("## 🔍 Key Global Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🌏 Population Leaders**
        - Asia dominates global population
        - China & India account for 35%+ of world population
        - Rapid urbanization in developing nations
        """)
    
    with col2:
        st.markdown("""
        **📈 Growth Patterns**
        - Africa shows highest growth rates
        - Europe faces population decline
        - Economic development affects growth
        """)
    
    with col3:
        st.markdown("""
        **🔮 Future Trends**
        - India likely to surpass China soon
        - African population expected to double
        - Aging populations in developed countries
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌍 Interactive World Population Dashboard | Built with Streamlit & Plotly</p>
        <p>📊 Explore demographic trends and make data-driven discoveries</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()