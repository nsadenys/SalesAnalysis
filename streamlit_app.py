# # # Import python packages
import streamlit as st
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import plotly.graph_objs as go


clt = pd.read_csv("real_estate_charlotte_2000_2024.csv")
clt.head()
#from snowflake.snowpark.context 
#import get_active_session

# Get the current credentials
#session = get_active_session()

st.set_page_config(page_title="Global Sale Products SuperStoreOrder Analysis",
                   page_icon=":sale:",
                   layout = "wide", initial_sidebar_state="expanded")

# Initialized the current page in the sessionstate if it doesn't exist
if "current_page" not in st.session_state:
    st.session_state.current_page = "overview"

def switch_page(page: str):
    st.session_state.current_page = page

# Sidebar
st.sidebar.subheader("Product Dashboard Page")

overview_button = st.sidebar.button(
    "Overview", on_click=switch_page, args=["overview"]
)

data_button = st.sidebar.button(
    "Market Trend", on_click=switch_page, args=["market_trend"]
)

data_button = st.sidebar.button(
    "What to Buy?", on_click=switch_page, args=["what_to_buy"]
)

data_button = st.sidebar.button(
    "Where to Buy?", on_click=switch_page, args=["where_to_buy"]
)

st.sidebar.subheader("Please filter here: ")
house_type = st.sidebar.selectbox(
    "",
    options=["ALL"] + list(clt["descbuildingtype"].unique()),  # Include ALL option
    index=0  # Default selection is "ALL"
)

# KPI Function
def KPI(clt, house_type):
    # Function to calculate and display KPIs based on selected house type.
    
    # Parameters:
    # - clt (DataFrame): The real estate dataframe containing the data.
    # - house_type (str): The selected house type ('ALL' or a specific house type).

    clt["dateofsale"] = pd.to_datetime(clt["dateofsale"])
    clt["yearofsale"] = clt["dateofsale"].dt.year

    # Filter data based on house type selection
    if house_type == "ALL":
        clt_selection = clt  # Use all house types if "ALL" is selected
    else:
        clt_selection = clt[clt["descbuildingtype"] == house_type]

    # Function to calculate median prices and growth rate for a list of house types
    def calculate_kpis_for_house_types(house_type, data):
        # If all house types are selected, calculate the metrics for all of them combined
        if house_type == "ALL":
            # Calculate for all `descbuildingtype` combined
            median_price_2003 = data[data["yearofsale"] == 2003]["price"].median()
            median_price_2023 = data[data["yearofsale"] == 2023]["price"].median()
            growth_rate = ((median_price_2023 - median_price_2003) / median_price_2003) * 100

            return {
                "2003 Median Price": f"${median_price_2003:.0f}",
                "2023 Median Price": f"${median_price_2023:.0f}",
                "Growth Rate": f"{growth_rate:.2f}%"
            }
        
        # If specific house type is selected, calculate for that house type
        clt_house = data[data["descbuildingtype"] == house_type]
        
        # Median price in 2003
        median_price_2003 = clt_house[clt_house["yearofsale"] == 2003]["price"].median()
        median_price_2003_str = f"${median_price_2003:.0f}"
        
        # Median price in 2023
        median_price_2023 = clt_house[clt_house["yearofsale"] == 2023]["price"].median()
        median_price_2023_str = f"${median_price_2023:.0f}"
        
        # Growth rate from 2003 to 2023
        growth_rate = ((median_price_2023 - median_price_2003) / median_price_2003) * 100
        growth_rate_str = f"{growth_rate:.2f}%"
        
        # Store the results for this house type
        return {
            "2003 Median Price": median_price_2003_str,
            "2023 Median Price": median_price_2023_str,
            "Growth Rate": growth_rate_str
        }

    # Calculate KPIs based on house type selection
    kpi_results = calculate_kpis_for_house_types(house_type, clt_selection)

    # Define the CSS for red text, font size 13, and centered alignment
    kpi_style = """
        <style>
            .kpi {
                color: #1E3A5F;
                font-size: 14px;
                font-weight: bold;
                text-align: center;
            }
        </style>
    """

    # Display the KPIs
    st.markdown(kpi_style, unsafe_allow_html=True)

    if house_type == "ALL":
        # If all house types are selected, display the combined result
        left_column, middle_column, right_column = st.columns(3)
        with left_column:
            st.markdown(f'<div class="kpi">2003:</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi">{kpi_results["2003 Median Price"]}</div>', unsafe_allow_html=True)
        with middle_column:
            st.markdown(f'<div class="kpi">2023:</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi">{kpi_results["2023 Median Price"]}</div>', unsafe_allow_html=True)
        with right_column:
            st.markdown(f'<div class="kpi">Growth Rate in 20 Years:</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi">{kpi_results["Growth Rate"]}</div>', unsafe_allow_html=True)
    else:
        # Otherwise, display the result for the selected house type
        left_column, middle_column, right_column = st.columns(3)
        with left_column:
            st.markdown(f'<div class="kpi">2003:</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi">{kpi_results["2003 Median Price"]}</div>', unsafe_allow_html=True)
        with middle_column:
            st.markdown(f'<div class="kpi">2023:</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi">{kpi_results["2023 Median Price"]}</div>', unsafe_allow_html=True)
        with right_column:
            st.markdown(f'<div class="kpi">Growth Rate in 20 Years:</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi">{kpi_results["Growth Rate"]}</div>', unsafe_allow_html=True)

# The Overview page fuction
def overview():
    st.markdown("<h1 style='text-align: center; color: red;'> Housing Market Trend Analysis in Charlotte Region</h1>", unsafe_allow_html=True)
    #st.subheader('Overview: This Analysis is to help you understand the housing market trend in Charlotte region in the last 20 years so that you can decide what to buy and where to buy the right kind of properties that will gain the most value in the future.')
    st.markdown("<h3 style='text-align: left; color: #1E3A5F;'>Overview: This Analysis is to help you understand the housing market trend in Charlotte region in the last 20 years so that you can decide what to buy and where to buy the right kind of properties that will gain the most value in the future.</h3>", unsafe_allow_html=True)
    st.image("charlotteLogo.JPG")

# The Market Trend page function
def market_trend():
    st.markdown("<h1 style='text-align: center; color: red;'> Housing Market Trend Analysis in Charlotte Region</h1>", unsafe_allow_html=True) 
    st.markdown("<h3 style='text-align: center; color: #1E3A5F;'>The summary shows the increase of median house price from 2003 to 2023</h3>", unsafe_allow_html=True)    

    # Call the KPI function
    KPI(clt, house_type)

    # Filter the DataFrame based on the selected building type
    clt_selection = clt.query("descbuildingtype == @house_type")
    left_column,  right_column = st.columns(2)
    with left_column:
        # Plotting a 20 years trend line chart:
        # Filter data based on house type selection
        if house_type == "ALL":
            clt_selection = clt  # Use all house types if "ALL" is selected
        else:
            clt_selection = clt[clt["descbuildingtype"] == house_type]

        if house_type == "ALL":
            median_prices_by_year_buildingtype = clt.groupby(["yearofsale", "descbuildingtype"])["price"].median().unstack()
        else:
            median_prices_by_year_buildingtype = clt_selection.groupby(["yearofsale", "descbuildingtype"])["price"].median().unstack()

        # Group data by year and building type and calculate median price
        if house_type == "ALL":
            median_prices_by_year_buildingtype = clt.groupby(["yearofsale", "descbuildingtype"])["price"].median().unstack()
        else:
            median_prices_by_year_buildingtype = clt_selection.groupby(["yearofsale", "descbuildingtype"])["price"].median().unstack()

        # Create the figure for the trend chart
        chart_20_year_trend = plt.figure(figsize=(10, 6))

        # Define the colors for each building type
        colors = ['#1E3A5F', 'orange', 'steelblue']  # Add more colors if needed

        # Define a function to format the price axis as an integer (in thousands with 'k' suffix)
        def format_price(x, pos):
            return f'${int(x / 1000)}k'  # Format as integer in $Xk

        # Plot each building type with the corresponding color
        for idx, buildingtype in enumerate(median_prices_by_year_buildingtype.columns):
            plt.plot(
                median_prices_by_year_buildingtype.index,
                median_prices_by_year_buildingtype[buildingtype],
                marker="o",
                label=buildingtype,
                color=colors[idx]  # Assign the color from the list
            )

            # For "ALL", display the median price for 2024 at the end of each line
            if house_type == "ALL":
                # Get the median price for 2024 (last year)
                median_price_2024 = median_prices_by_year_buildingtype.loc[2024, buildingtype]
                
                # Display the median price for 2024 at the end of each line
                plt.text(
                    2024, median_price_2024, f"${int(median_price_2024 // 1000)}k", 
                    color=colors[idx], 
                    verticalalignment='bottom', horizontalalignment='center', 
                    fontsize=10, fontweight='bold'
                )

        # Add title and labels
        plt.title("20-Year House Price Trend")
        plt.xlabel("Year of Sale")
        plt.ylabel("Median Price")
        plt.legend(title="House Type")
        plt.grid(True)

        # Display the plot in Streamlit
        st.pyplot(chart_20_year_trend)

    with right_column:
        # Plotting a 20 years trend bar chart: 
        
        # Filter the data for the years 2003, 2013, 2018, and 2024
        filtered_years = [2003, 2013, 2018, 2024]
        filtered_data = clt[clt["yearofsale"].isin(filtered_years)]

        # Define custom colors for each bedroom count
        colors = ['#1E3A5F', 'orange', 'steelblue', 'lightsteelblue']  # For the years 2003, 2013, 2018, and 2024
    
        # Filter data based on the selected house type
        if house_type == "ALL":
            clt_selection = filtered_data  # Use all house types if "ALL" is selected
        else:
            clt_selection = filtered_data[filtered_data["descbuildingtype"] == house_type]

        # Group by year and building type and calculate the median price
        median_prices = clt_selection.groupby(["descbuildingtype", "yearofsale"])["price"].median().unstack(level=1)

        # Calculate move-in cost (as a proportion of the median price)
        movein_cost = (median_prices * 0.2) + (median_prices * 0.08 * 0.035)

        # Plotting the side-by-side bar chart
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot median prices as bars for each house type and year
        median_prices.plot(kind="bar", ax=ax, width=0.8, color=colors)

        # Add title and labels
        plt.title("House Price in 20 Years", fontsize=16)
        plt.xlabel("House Type", fontsize=14)
        plt.ylabel("Median Price ($)", fontsize=14)
        plt.xticks(rotation=0)  # Set x-axis labels to be horizontal

        # Add a legend to differentiate house types
        plt.legend(title="Year of Sale", title_fontsize='13', loc='upper left', bbox_to_anchor=(1, 1))

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

# What to Buy Page function:
def what_to_buy():
    st.markdown("<h1 style='text-align: center; color: red;'> Housing Market Trend Analysis in Charlotte Region</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #1E3A5F;'>The summary shows the increase of median house price from 2003 to 2023</h3>", unsafe_allow_html=True)
    
    # Call the KPI function
    KPI(clt, house_type)

    left_column,  right_column = st.columns(2)

    # Plotting Number of Bedrooms vs Price Comparison chart
    with left_column:
        # Convert 'dateofsale' to datetime and extract the year
        clt["dateofsale"] = pd.to_datetime(clt["dateofsale"])
        clt["yearofsale"] = clt["dateofsale"].dt.year

        # Handle missing values for 'bedrooms' and 'price'
        clt["bedrooms"] = clt["bedrooms"].fillna(0).astype(int)
        clt["price"] = clt["price"].fillna(0).astype(int)

        # Filter data for selected years
        filtered_data = clt[clt["yearofsale"].isin([2003, 2008, 2013, 2018, 2023])]

        # Filter bedrooms between 2 and 5
        filtered_data = filtered_data[
            (filtered_data["bedrooms"] >= 2) & (filtered_data["bedrooms"] <= 5)
        ]

        # Group by bedrooms and year of sale, then calculate median price
        median_prices = (
            filtered_data.groupby(["yearofsale", "bedrooms"])["price"]
            .median()
            .unstack()
            .astype(int)
        )

        # Reset index for plotting purposes
        median_prices = median_prices.reset_index()

        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define the number of bars and the width of each bar
        bar_width = 0.15
        years = median_prices["yearofsale"]

        # Create the bar positions (one bar for each bedroom count)
        positions = range(len(years))

        # Define custom colors for each bedroom count
        colors = ['#1E3A5F', 'orange', 'steelblue', 'lightsteelblue']  # Color for 2, 3, 4, and 5 bedrooms

        # Plot the bars for each bedroom count (overlapping bars)
        for i, bedroom_count in enumerate([2, 3, 4, 5]):
            ax.bar(
                [pos + i * bar_width for pos in positions],  # Shift the positions for overlap
                median_prices[bedroom_count],  # Median price for each bedroom count
                width=bar_width,
                label=f'{bedroom_count} Bedrooms',
                color=colors[i]  # Use custom colors
            )

        # Set the x-axis labels and other chart properties
        ax.set_xlabel('Year')
        ax.set_ylabel('Median Price ($)')
        ax.set_title('Number of Bedrooms vs Price Comparison', fontsize=14, color='#1E3A5F')
        ax.set_xticks([pos + 1.5 * bar_width for pos in positions])  # Adjust x-ticks for overlap
        ax.set_xticklabels(years)
        ax.legend()

        # Show the plot
        plt.tight_layout()
        st.pyplot(fig)

    # Plotting House Size vs Price chart
    with right_column:
        # Handle missing values for 'bedrooms', 'price', and 'heatedarea'
        clt["heatedarea"] = clt["heatedarea"].fillna(0).astype(float)
        clt["price"] = clt["price"].fillna(0).astype(float)

        # Filter the data based on the selected building type and the year range (2004-2024)
        if house_type == "ALL":
            filtered_data = clt[(clt['yearofsale'].between(2004, 2024))]
        else:
            filtered_data = clt[(clt['yearofsale'].between(2004, 2024)) & 
                                (clt['descbuildingtype'] == house_type)]

        # Group by year and calculate median price
        median_prices = filtered_data.groupby(["yearofsale"])["price"].median()

        # Group by year and calculate median heated area
        median_area = filtered_data.groupby(["yearofsale"])["heatedarea"].median()

        # Create a dual-axis chart
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot the median heated area as a bar chart on the left y-axis
        ax1.bar(median_area.index, median_area, color='orange', alpha=0.6, label='Size in SQFT')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Size in SQFT', color='black')  # Set y-axis label color to black
        ax1.tick_params(axis='y', labelcolor='black')

        # Create a second y-axis to plot the median price as a line chart
        ax2 = ax1.twinx()
        ax2.plot(median_prices.index, median_prices, color='#1E3A5F', marker='o', label='Median Price')
        ax2.set_ylabel('Price ($)', color='black')  # Set y-axis label color to black
        ax2.tick_params(axis='y', labelcolor='black')

        # Add a title to the plot
        plt.title(f'House Size vs Price ({house_type} - 2004 to 2024)', fontsize=14, color='#1E3A5F')

        # Show the legend
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Show the plot
        plt.tight_layout()
        st.pyplot(fig)

def where_to_buy():
    st.markdown("<h1 style='text-align: center; color: red;'> Housing Market Trend Analysis in Charlotte Region</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #1E3A5F;'>The summary shows the increase of median house price from 2003 to 2023</h3>", unsafe_allow_html=True)

    # Call the KPI function
    KPI(clt, house_type)
    
    clt['dateofsale'] = pd.to_datetime(clt['dateofsale'])
    clt['yearofsale'] = clt['dateofsale'].dt.year

    # Create two columns for the layout (rows in Streamlit)
    col1, col2 = st.columns(2)

    # Plotting the House Growth by Region chart
    with col1:

        # Filter data based on house type selection
        if house_type == "ALL":
            clt_selection = clt  # Use all house types if "ALL" is selected
        else:
            clt_selection = clt[clt["descbuildingtype"] == house_type]
            
        # Filter data for the years 2003 and 2023
        filtered_data = clt_selection[clt_selection['yearofsale'].isin([2003, 2023])]

        # Group by city and year, then calculate median price
        median_prices = filtered_data.groupby(['city', 'yearofsale'])['price'].median().unstack()

        # Calculate the difference between the 2023 median price and 2003 median price
        median_prices['price_difference'] = median_prices[2023] - median_prices[2003]

        # Exclude rows where the price difference is NaN
        median_price_diff_by_city = median_prices.dropna(subset=['price_difference']).astype(int)

        # Calculate the growth rate: (Median Price in 2023 - Median Price in 2003) / Median Price in 2003 * 100
        median_price_diff_by_city['growth_rate'] = ((median_price_diff_by_city[2023] - median_price_diff_by_city[2003]) / median_price_diff_by_city[2003]) * 100

        # Convert the growth rate to a readable format (e.g., rounded to 2 decimal places)
        median_price_diff_by_city['growth_rate'] = median_price_diff_by_city['growth_rate'].round(2)

        # Format the price difference for readability (e.g., $Xk)
        median_price_diff_by_city['price_difference'] = median_price_diff_by_city['price_difference'].apply(lambda x: f"${x // 1000}k")

        # Prepare the data for the plot
        plot_data = median_price_diff_by_city[['price_difference', 'growth_rate']].reset_index()

        # Create the horizontal bar chart using Plotly Express
        fig = px.bar(
            plot_data, 
            x='price_difference', 
            y='city', 
            orientation='h', 
            title=f"House Price Growth by Regions (2003-2023) for {house_type} Homes",
            hover_data={'growth_rate': True, 'price_difference': True},  # Show growth rate and price difference on hover
            text='price_difference',  # Add the price difference text on the bars
            color_discrete_sequence=['steelblue']
        )

        # Update layout for better presentation
        fig.update_layout(
            title={'x': 0.5},  # Center title
            xaxis_title="Price Difference ($)",
            yaxis_title="City",
            xaxis=dict(tickformat="$.0f"),  # Format x-axis to show price in dollars
            coloraxis_colorbar_title="Growth Rate (%)",  # Add a color bar for growth rate
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

    # Plotting the Home Sold by Year vs Price Chart
    with col2:
        # Convert 'dateofsale' to datetime and extract the year
        clt["dateofsale"] = pd.to_datetime(clt["dateofsale"])
        clt["yearofsale"] = clt["dateofsale"].dt.year

        # Filter the data based on the selected building type and the year range (2004-2024)
        if house_type == "ALL":
            filtered_data = clt[(clt['yearofsale'].between(2004, 2024))]
        else:
            filtered_data = clt[(clt['yearofsale'].between(2004, 2024)) & 
                                (clt['descbuildingtype'] == house_type)]

        # Calculate median price by year
        median_prices = filtered_data.groupby(["yearofsale"])["price"].median()

        # Calculate number of homes sold by year
        home_sold = filtered_data.groupby(["yearofsale"])["pid"].count()

        # Define a function to format the price axis as an integer (in thousands with 'k' suffix)
        def format_price(x, pos):
            return f'${int(x / 1000)}k'  # Format as integer in $Xk

        # Plot the results
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot median prices on the right y-axis (navy blue line)
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Price ($)", color="black")
        ax1.plot(
            median_prices.index,
            median_prices.values,
            color="#1E3A5F",
            label="Price",
            marker="o",
        )

        # Format the price axis (right y-axis) to display in thousands with 'k' (as an integer)
        ax1.yaxis.set_major_formatter(FuncFormatter(format_price))
        ax1.tick_params(axis="y", labelcolor="black")

        # Create a second y-axis for the number of homes sold (orange line)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Number of Homes Sold", color="black")
        ax2.plot(
            home_sold.index,
            home_sold.values,
            color="orange",
            label="Number of Homes Sold",
            marker="x",
        )
        ax2.tick_params(axis="y", labelcolor="black")

        # Format the x-axis labels as integers (years)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Add a title and show the plot
        plt.title(f"{house_type} Home Sold: by Year vs Price")
        fig.tight_layout()
        st.pyplot(fig)

fn_map = {
    "overview": overview,
    "market_trend": market_trend,
    "what_to_buy": what_to_buy,
    "where_to_buy": where_to_buy
}

main_window = st.container()

main_workflow = fn_map.get(st.session_state.current_page, overview)

main_workflow()






# # # # Import python packages
# import streamlit as st
# import altair as alt
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter
# import plotly.express as px
# import plotly.graph_objs as go


# clt = pd.read_csv("real_estate_charlotte_2000_2024.csv")
# clt.head()
# #from snowflake.snowpark.context 
# #import get_active_session

# # Get the current credentials
# #session = get_active_session()

# st.set_page_config(page_title="Charlotte Housing Market Trend Analysis",
#                    page_icon=":house:",
#                    layout = "wide", initial_sidebar_state="expanded")

# # Initialized the current page in the sessionstate if it doesn't exist
# if "current_page" not in st.session_state:
#     st.session_state.current_page = "overview"

# def switch_page(page: str):
#     st.session_state.current_page = page

# # Sidebar
# st.sidebar.subheader("Navigation Page")

# overview_button = st.sidebar.button(
#     "Overview", on_click=switch_page, args=["overview"]
# )

# data_button = st.sidebar.button(
#     "Market Trend", on_click=switch_page, args=["market_trend"]
# )

# data_button = st.sidebar.button(
#     "What to Buy?", on_click=switch_page, args=["what_to_buy"]
# )

# data_button = st.sidebar.button(
#     "Where to Buy?", on_click=switch_page, args=["where_to_buy"]
# )

# st.sidebar.subheader("Please filter here: ")
# house_type = st.sidebar.selectbox(
#     "",
#     options=["ALL"] + list(clt["descbuildingtype"].unique()),  # Include ALL option
#     index=0  # Default selection is "ALL"
# )

# # KPI Function
# def KPI(clt, house_type):
#     # Function to calculate and display KPIs based on selected house type.
    
#     # Parameters:
#     # - clt (DataFrame): The real estate dataframe containing the data.
#     # - house_type (str): The selected house type ('ALL' or a specific house type).

#     clt["dateofsale"] = pd.to_datetime(clt["dateofsale"])
#     clt["yearofsale"] = clt["dateofsale"].dt.year

#     # Filter data based on house type selection
#     if house_type == "ALL":
#         clt_selection = clt  # Use all house types if "ALL" is selected
#     else:
#         clt_selection = clt[clt["descbuildingtype"] == house_type]

#     # Function to calculate median prices and growth rate for a list of house types
#     def calculate_kpis_for_house_types(house_type, data):
#         # If all house types are selected, calculate the metrics for all of them combined
#         if house_type == "ALL":
#             # Calculate for all `descbuildingtype` combined
#             median_price_2003 = data[data["yearofsale"] == 2003]["price"].median()
#             median_price_2023 = data[data["yearofsale"] == 2023]["price"].median()
#             growth_rate = ((median_price_2023 - median_price_2003) / median_price_2003) * 100

#             return {
#                 "2003 Median Price": f"${median_price_2003:.0f}",
#                 "2023 Median Price": f"${median_price_2023:.0f}",
#                 "Growth Rate": f"{growth_rate:.2f}%"
#             }
        
#         # If specific house type is selected, calculate for that house type
#         clt_house = data[data["descbuildingtype"] == house_type]
        
#         # Median price in 2003
#         median_price_2003 = clt_house[clt_house["yearofsale"] == 2003]["price"].median()
#         median_price_2003_str = f"${median_price_2003:.0f}"
        
#         # Median price in 2023
#         median_price_2023 = clt_house[clt_house["yearofsale"] == 2023]["price"].median()
#         median_price_2023_str = f"${median_price_2023:.0f}"
        
#         # Growth rate from 2003 to 2023
#         growth_rate = ((median_price_2023 - median_price_2003) / median_price_2003) * 100
#         growth_rate_str = f"{growth_rate:.2f}%"
        
#         # Store the results for this house type
#         return {
#             "2003 Median Price": median_price_2003_str,
#             "2023 Median Price": median_price_2023_str,
#             "Growth Rate": growth_rate_str
#         }

#     # Calculate KPIs based on house type selection
#     kpi_results = calculate_kpis_for_house_types(house_type, clt_selection)

#     # Define the CSS for red text, font size 13, and centered alignment
#     kpi_style = """
#         <style>
#             .kpi {
#                 color: #1E3A5F;
#                 font-size: 14px;
#                 font-weight: bold;
#                 text-align: center;
#             }
#         </style>
#     """

#     # Display the KPIs
#     st.markdown(kpi_style, unsafe_allow_html=True)

#     if house_type == "ALL":
#         # If all house types are selected, display the combined result
#         left_column, middle_column, right_column = st.columns(3)
#         with left_column:
#             st.markdown(f'<div class="kpi">2003:</div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="kpi">{kpi_results["2003 Median Price"]}</div>', unsafe_allow_html=True)
#         with middle_column:
#             st.markdown(f'<div class="kpi">2023:</div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="kpi">{kpi_results["2023 Median Price"]}</div>', unsafe_allow_html=True)
#         with right_column:
#             st.markdown(f'<div class="kpi">Growth Rate in 20 Years:</div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="kpi">{kpi_results["Growth Rate"]}</div>', unsafe_allow_html=True)
#     else:
#         # Otherwise, display the result for the selected house type
#         left_column, middle_column, right_column = st.columns(3)
#         with left_column:
#             st.markdown(f'<div class="kpi">2003:</div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="kpi">{kpi_results["2003 Median Price"]}</div>', unsafe_allow_html=True)
#         with middle_column:
#             st.markdown(f'<div class="kpi">2023:</div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="kpi">{kpi_results["2023 Median Price"]}</div>', unsafe_allow_html=True)
#         with right_column:
#             st.markdown(f'<div class="kpi">Growth Rate in 20 Years:</div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="kpi">{kpi_results["Growth Rate"]}</div>', unsafe_allow_html=True)

# # The Overview page fuction
# def overview():
#     st.markdown("<h1 style='text-align: center; color: red;'> Housing Market Trend Analysis in Charlotte Region</h1>", unsafe_allow_html=True)
#     #st.subheader('Overview: This Analysis is to help you understand the housing market trend in Charlotte region in the last 20 years so that you can decide what to buy and where to buy the right kind of properties that will gain the most value in the future.')
#     st.markdown("<h3 style='text-align: left; color: #1E3A5F;'>Overview: This Analysis is to help you understand the housing market trend in Charlotte region in the last 20 years so that you can decide what to buy and where to buy the right kind of properties that will gain the most value in the future.</h3>", unsafe_allow_html=True)
#     st.image("charlotteLogo.JPG")

# # The Market Trend page function
# def market_trend():
#     st.markdown("<h1 style='text-align: center; color: red;'> Housing Market Trend Analysis in Charlotte Region</h1>", unsafe_allow_html=True) 
#     st.markdown("<h3 style='text-align: center; color: #1E3A5F;'>The summary shows the increase of median house price from 2003 to 2023</h3>", unsafe_allow_html=True)    

#     # Call the KPI function
#     KPI(clt, house_type)

#     # Filter the DataFrame based on the selected building type
#     clt_selection = clt.query("descbuildingtype == @house_type")
#     left_column,  right_column = st.columns(2)
#     with left_column:
#         # Plotting a 20 years trend line chart:
#         # Filter data based on house type selection
#         if house_type == "ALL":
#             clt_selection = clt  # Use all house types if "ALL" is selected
#         else:
#             clt_selection = clt[clt["descbuildingtype"] == house_type]

#         if house_type == "ALL":
#             median_prices_by_year_buildingtype = clt.groupby(["yearofsale", "descbuildingtype"])["price"].median().unstack()
#         else:
#             median_prices_by_year_buildingtype = clt_selection.groupby(["yearofsale", "descbuildingtype"])["price"].median().unstack()

#         # Group data by year and building type and calculate median price
#         if house_type == "ALL":
#             median_prices_by_year_buildingtype = clt.groupby(["yearofsale", "descbuildingtype"])["price"].median().unstack()
#         else:
#             median_prices_by_year_buildingtype = clt_selection.groupby(["yearofsale", "descbuildingtype"])["price"].median().unstack()

#         # Create the figure for the trend chart
#         chart_20_year_trend = plt.figure(figsize=(10, 6))

#         # Define the colors for each building type
#         colors = ['#1E3A5F', 'orange', 'steelblue']  # Add more colors if needed

#         # Define a function to format the price axis as an integer (in thousands with 'k' suffix)
#         def format_price(x, pos):
#             return f'${int(x / 1000)}k'  # Format as integer in $Xk

#         # Plot each building type with the corresponding color
#         for idx, buildingtype in enumerate(median_prices_by_year_buildingtype.columns):
#             plt.plot(
#                 median_prices_by_year_buildingtype.index,
#                 median_prices_by_year_buildingtype[buildingtype],
#                 marker="o",
#                 label=buildingtype,
#                 color=colors[idx]  # Assign the color from the list
#             )

#             # For "ALL", display the median price for 2024 at the end of each line
#             if house_type == "ALL":
#                 # Get the median price for 2024 (last year)
#                 median_price_2024 = median_prices_by_year_buildingtype.loc[2024, buildingtype]
                
#                 # Display the median price for 2024 at the end of each line
#                 plt.text(
#                     2024, median_price_2024, f"${int(median_price_2024 // 1000)}k", 
#                     color=colors[idx], 
#                     verticalalignment='bottom', horizontalalignment='center', 
#                     fontsize=10, fontweight='bold'
#                 )

#         # Add title and labels
#         plt.title("20-Year House Price Trend")
#         plt.xlabel("Year of Sale")
#         plt.ylabel("Median Price")
#         plt.legend(title="House Type")
#         plt.grid(True)

#         # Display the plot in Streamlit
#         st.pyplot(chart_20_year_trend)

#     with right_column:
#         # Plotting a 20 years trend bar chart: 
        
#         # Filter the data for the years 2003, 2013, 2018, and 2024
#         filtered_years = [2003, 2013, 2018, 2024]
#         filtered_data = clt[clt["yearofsale"].isin(filtered_years)]

#         # Define custom colors for each bedroom count
#         colors = ['#1E3A5F', 'orange', 'steelblue', 'lightsteelblue']  # For the years 2003, 2013, 2018, and 2024
    
#         # Filter data based on the selected house type
#         if house_type == "ALL":
#             clt_selection = filtered_data  # Use all house types if "ALL" is selected
#         else:
#             clt_selection = filtered_data[filtered_data["descbuildingtype"] == house_type]

#         # Group by year and building type and calculate the median price
#         median_prices = clt_selection.groupby(["descbuildingtype", "yearofsale"])["price"].median().unstack(level=1)

#         # Calculate move-in cost (as a proportion of the median price)
#         movein_cost = (median_prices * 0.2) + (median_prices * 0.08 * 0.035)

#         # Plotting the side-by-side bar chart
#         fig, ax = plt.subplots(figsize=(12, 8))

#         # Plot median prices as bars for each house type and year
#         median_prices.plot(kind="bar", ax=ax, width=0.8, color=colors)

#         # Add title and labels
#         plt.title("House Price in 20 Years", fontsize=16)
#         plt.xlabel("House Type", fontsize=14)
#         plt.ylabel("Median Price ($)", fontsize=14)
#         plt.xticks(rotation=0)  # Set x-axis labels to be horizontal

#         # Add a legend to differentiate house types
#         plt.legend(title="Year of Sale", title_fontsize='13', loc='upper left', bbox_to_anchor=(1, 1))

#         # Adjust layout to prevent overlapping
#         plt.tight_layout()

#         # Display the plot in Streamlit
#         st.pyplot(fig)

# # What to Buy Page function:
# def what_to_buy():
#     st.markdown("<h1 style='text-align: center; color: red;'> Housing Market Trend Analysis in Charlotte Region</h1>", unsafe_allow_html=True)
#     st.markdown("<h3 style='text-align: center; color: #1E3A5F;'>The summary shows the increase of median house price from 2003 to 2023</h3>", unsafe_allow_html=True)
    
#     # Call the KPI function
#     KPI(clt, house_type)

#     left_column,  right_column = st.columns(2)

#     # Plotting Number of Bedrooms vs Price Comparison chart
#     with left_column:
#         # Convert 'dateofsale' to datetime and extract the year
#         clt["dateofsale"] = pd.to_datetime(clt["dateofsale"])
#         clt["yearofsale"] = clt["dateofsale"].dt.year

#         # Handle missing values for 'bedrooms' and 'price'
#         clt["bedrooms"] = clt["bedrooms"].fillna(0).astype(int)
#         clt["price"] = clt["price"].fillna(0).astype(int)

#         # Filter data for selected years
#         filtered_data = clt[clt["yearofsale"].isin([2003, 2008, 2013, 2018, 2023])]

#         # Filter bedrooms between 2 and 5
#         filtered_data = filtered_data[
#             (filtered_data["bedrooms"] >= 2) & (filtered_data["bedrooms"] <= 5)
#         ]

#         # Group by bedrooms and year of sale, then calculate median price
#         median_prices = (
#             filtered_data.groupby(["yearofsale", "bedrooms"])["price"]
#             .median()
#             .unstack()
#             .astype(int)
#         )

#         # Reset index for plotting purposes
#         median_prices = median_prices.reset_index()

#         # Create a figure and axis object
#         fig, ax = plt.subplots(figsize=(10, 6))

#         # Define the number of bars and the width of each bar
#         bar_width = 0.15
#         years = median_prices["yearofsale"]

#         # Create the bar positions (one bar for each bedroom count)
#         positions = range(len(years))

#         # Define custom colors for each bedroom count
#         colors = ['#1E3A5F', 'orange', 'steelblue', 'lightsteelblue']  # Color for 2, 3, 4, and 5 bedrooms

#         # Plot the bars for each bedroom count (overlapping bars)
#         for i, bedroom_count in enumerate([2, 3, 4, 5]):
#             ax.bar(
#                 [pos + i * bar_width for pos in positions],  # Shift the positions for overlap
#                 median_prices[bedroom_count],  # Median price for each bedroom count
#                 width=bar_width,
#                 label=f'{bedroom_count} Bedrooms',
#                 color=colors[i]  # Use custom colors
#             )

#         # Set the x-axis labels and other chart properties
#         ax.set_xlabel('Year')
#         ax.set_ylabel('Median Price ($)')
#         ax.set_title('Number of Bedrooms vs Price Comparison', fontsize=14, color='#1E3A5F')
#         ax.set_xticks([pos + 1.5 * bar_width for pos in positions])  # Adjust x-ticks for overlap
#         ax.set_xticklabels(years)
#         ax.legend()

#         # Show the plot
#         plt.tight_layout()
#         st.pyplot(fig)

#     # Plotting House Size vs Price chart
#     with right_column:
#         # Handle missing values for 'bedrooms', 'price', and 'heatedarea'
#         clt["heatedarea"] = clt["heatedarea"].fillna(0).astype(float)
#         clt["price"] = clt["price"].fillna(0).astype(float)

#         # Filter the data based on the selected building type and the year range (2004-2024)
#         if house_type == "ALL":
#             filtered_data = clt[(clt['yearofsale'].between(2004, 2024))]
#         else:
#             filtered_data = clt[(clt['yearofsale'].between(2004, 2024)) & 
#                                 (clt['descbuildingtype'] == house_type)]

#         # Group by year and calculate median price
#         median_prices = filtered_data.groupby(["yearofsale"])["price"].median()

#         # Group by year and calculate median heated area
#         median_area = filtered_data.groupby(["yearofsale"])["heatedarea"].median()

#         # Create a dual-axis chart
#         fig, ax1 = plt.subplots(figsize=(10, 6))

#         # Plot the median heated area as a bar chart on the left y-axis
#         ax1.bar(median_area.index, median_area, color='orange', alpha=0.6, label='Size in SQFT')
#         ax1.set_xlabel('Year')
#         ax1.set_ylabel('Size in SQFT', color='black')  # Set y-axis label color to black
#         ax1.tick_params(axis='y', labelcolor='black')

#         # Create a second y-axis to plot the median price as a line chart
#         ax2 = ax1.twinx()
#         ax2.plot(median_prices.index, median_prices, color='#1E3A5F', marker='o', label='Median Price')
#         ax2.set_ylabel('Price ($)', color='black')  # Set y-axis label color to black
#         ax2.tick_params(axis='y', labelcolor='black')

#         # Add a title to the plot
#         plt.title(f'House Size vs Price ({house_type} - 2004 to 2024)', fontsize=14, color='#1E3A5F')

#         # Show the legend
#         ax1.legend(loc='upper left')
#         ax2.legend(loc='upper right')

#         # Show the plot
#         plt.tight_layout()
#         st.pyplot(fig)

# def where_to_buy():
#     st.markdown("<h1 style='text-align: center; color: red;'> Housing Market Trend Analysis in Charlotte Region</h1>", unsafe_allow_html=True)
#     st.markdown("<h3 style='text-align: center; color: #1E3A5F;'>The summary shows the increase of median house price from 2003 to 2023</h3>", unsafe_allow_html=True)

#     # Call the KPI function
#     KPI(clt, house_type)
    
#     clt['dateofsale'] = pd.to_datetime(clt['dateofsale'])
#     clt['yearofsale'] = clt['dateofsale'].dt.year

#     # Create two columns for the layout (rows in Streamlit)
#     col1, col2 = st.columns(2)

#     # Plotting the House Growth by Region chart
#     with col1:

#         # Filter data based on house type selection
#         if house_type == "ALL":
#             clt_selection = clt  # Use all house types if "ALL" is selected
#         else:
#             clt_selection = clt[clt["descbuildingtype"] == house_type]
            
#         # Filter data for the years 2003 and 2023
#         filtered_data = clt_selection[clt_selection['yearofsale'].isin([2003, 2023])]

#         # Group by city and year, then calculate median price
#         median_prices = filtered_data.groupby(['city', 'yearofsale'])['price'].median().unstack()

#         # Calculate the difference between the 2023 median price and 2003 median price
#         median_prices['price_difference'] = median_prices[2023] - median_prices[2003]

#         # Exclude rows where the price difference is NaN
#         median_price_diff_by_city = median_prices.dropna(subset=['price_difference']).astype(int)

#         # Calculate the growth rate: (Median Price in 2023 - Median Price in 2003) / Median Price in 2003 * 100
#         median_price_diff_by_city['growth_rate'] = ((median_price_diff_by_city[2023] - median_price_diff_by_city[2003]) / median_price_diff_by_city[2003]) * 100

#         # Convert the growth rate to a readable format (e.g., rounded to 2 decimal places)
#         median_price_diff_by_city['growth_rate'] = median_price_diff_by_city['growth_rate'].round(2)

#         # Format the price difference for readability (e.g., $Xk)
#         median_price_diff_by_city['price_difference'] = median_price_diff_by_city['price_difference'].apply(lambda x: f"${x // 1000}k")

#         # Prepare the data for the plot
#         plot_data = median_price_diff_by_city[['price_difference', 'growth_rate']].reset_index()

#         # Create the horizontal bar chart using Plotly Express
#         fig = px.bar(
#             plot_data, 
#             x='price_difference', 
#             y='city', 
#             orientation='h', 
#             title=f"House Price Growth by Regions (2003-2023) for {house_type} Homes",
#             hover_data={'growth_rate': True, 'price_difference': True},  # Show growth rate and price difference on hover
#             text='price_difference',  # Add the price difference text on the bars
#             color_discrete_sequence=['steelblue']
#         )

#         # Update layout for better presentation
#         fig.update_layout(
#             title={'x': 0.5},  # Center title
#             xaxis_title="Price Difference ($)",
#             yaxis_title="City",
#             xaxis=dict(tickformat="$.0f"),  # Format x-axis to show price in dollars
#             coloraxis_colorbar_title="Growth Rate (%)",  # Add a color bar for growth rate
#         )

#         # Display the plot in Streamlit
#         st.plotly_chart(fig)

#     # Plotting the Home Sold by Year vs Price Chart
#     with col2:
#         # Convert 'dateofsale' to datetime and extract the year
#         clt["dateofsale"] = pd.to_datetime(clt["dateofsale"])
#         clt["yearofsale"] = clt["dateofsale"].dt.year

#         # Filter the data based on the selected building type and the year range (2004-2024)
#         if house_type == "ALL":
#             filtered_data = clt[(clt['yearofsale'].between(2004, 2024))]
#         else:
#             filtered_data = clt[(clt['yearofsale'].between(2004, 2024)) & 
#                                 (clt['descbuildingtype'] == house_type)]

#         # Calculate median price by year
#         median_prices = filtered_data.groupby(["yearofsale"])["price"].median()

#         # Calculate number of homes sold by year
#         home_sold = filtered_data.groupby(["yearofsale"])["pid"].count()

#         # Define a function to format the price axis as an integer (in thousands with 'k' suffix)
#         def format_price(x, pos):
#             return f'${int(x / 1000)}k'  # Format as integer in $Xk

#         # Plot the results
#         fig, ax1 = plt.subplots(figsize=(10, 6))

#         # Plot median prices on the right y-axis (navy blue line)
#         ax1.set_xlabel("Year")
#         ax1.set_ylabel("Price ($)", color="black")
#         ax1.plot(
#             median_prices.index,
#             median_prices.values,
#             color="#1E3A5F",
#             label="Price",
#             marker="o",
#         )

#         # Format the price axis (right y-axis) to display in thousands with 'k' (as an integer)
#         ax1.yaxis.set_major_formatter(FuncFormatter(format_price))
#         ax1.tick_params(axis="y", labelcolor="black")

#         # Create a second y-axis for the number of homes sold (orange line)
#         ax2 = ax1.twinx()
#         ax2.set_ylabel("Number of Homes Sold", color="black")
#         ax2.plot(
#             home_sold.index,
#             home_sold.values,
#             color="orange",
#             label="Number of Homes Sold",
#             marker="x",
#         )
#         ax2.tick_params(axis="y", labelcolor="black")

#         # Format the x-axis labels as integers (years)
#         ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

#         # Add a title and show the plot
#         plt.title(f"{house_type} Home Sold: by Year vs Price")
#         fig.tight_layout()
#         st.pyplot(fig)

# fn_map = {
#     "overview": overview,
#     "market_trend": market_trend,
#     "what_to_buy": what_to_buy,
#     "where_to_buy": where_to_buy
# }

# main_window = st.container()

# main_workflow = fn_map.get(st.session_state.current_page, overview)

# main_workflow()