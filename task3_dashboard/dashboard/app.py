import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import warnings
warnings.filterwarnings('ignore')

# Load Data
df = pd.read_csv("../data/Sample - Superstore.csv", encoding="latin1")
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month_name()

app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("📊 Superstore Sales Dashboard",
            style={"textAlign":"center","color":"#2c3e50",
                   "fontFamily":"Arial","marginBottom":"20px"}),

    # Filters Row
    html.Div([
        html.Div([
            html.Label("Select Category:"),
            dcc.Dropdown(
                id="category-filter",
                options=[{"label":"All","value":"All"}] + 
                        [{"label":c,"value":c} for c in df["Category"].unique()],
                value="All", clearable=False
            )
        ], style={"width":"30%","display":"inline-block","padding":"10px"}),

        html.Div([
            html.Label("Select Region:"),
            dcc.Dropdown(
                id="region-filter",
                options=[{"label":"All","value":"All"}] + 
                        [{"label":r,"value":r} for r in df["Region"].unique()],
                value="All", clearable=False
            )
        ], style={"width":"30%","display":"inline-block","padding":"10px"}),

        html.Div([
            html.Label("Select Year:"),
            dcc.Dropdown(
                id="year-filter",
                options=[{"label":"All","value":"All"}] + 
                        [{"label":y,"value":y} for y in sorted(df["Year"].unique())],
                value="All", clearable=False
            )
        ], style={"width":"30%","display":"inline-block","padding":"10px"}),
    ]),

    # KPI Cards
    html.Div(id="kpi-cards", style={"display":"flex","justifyContent":"space-around",
                                     "margin":"20px 0"}),

    # Charts Row 1
    html.Div([
        dcc.Graph(id="sales-trend", style={"width":"60%","display":"inline-block"}),
        dcc.Graph(id="category-pie", style={"width":"40%","display":"inline-block"}),
    ]),

    # Charts Row 2
    html.Div([
        dcc.Graph(id="region-bar", style={"width":"50%","display":"inline-block"}),
        dcc.Graph(id="profit-scatter", style={"width":"50%","display":"inline-block"}),
    ]),

    # Charts Row 3
    html.Div([
        dcc.Graph(id="top-products", style={"width":"50%","display":"inline-block"}),
        dcc.Graph(id="segment-bar", style={"width":"50%","display":"inline-block"}),
    ]),

], style={"backgroundColor":"#f8f9fa","minHeight":"100vh","padding":"20px"})


@app.callback(
    [Output("kpi-cards","children"),
     Output("sales-trend","figure"),
     Output("category-pie","figure"),
     Output("region-bar","figure"),
     Output("profit-scatter","figure"),
     Output("top-products","figure"),
     Output("segment-bar","figure")],
    [Input("category-filter","value"),
     Input("region-filter","value"),
     Input("year-filter","value")]
)
def update_dashboard(category, region, year):
    filtered = df.copy()
    if category != "All":
        filtered = filtered[filtered["Category"] == category]
    if region != "All":
        filtered = filtered[filtered["Region"] == region]
    if year != "All":
        filtered = filtered[filtered["Year"] == year]

    # KPI Cards
    total_sales = filtered["Sales"].sum()
    total_profit = filtered["Profit"].sum()
    total_orders = filtered["Order ID"].nunique()
    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0

    kpi_style = {"backgroundColor":"white","padding":"20px","borderRadius":"10px",
                 "textAlign":"center","boxShadow":"2px 2px 5px rgba(0,0,0,0.1)",
                 "width":"22%"}

    kpis = [
        html.Div([html.H3(f"${total_sales:,.0f}"),
                  html.P("Total Sales",style={"color":"gray"})], style=kpi_style),
        html.Div([html.H3(f"${total_profit:,.0f}",
                          style={"color":"green" if total_profit>0 else "red"}),
                  html.P("Total Profit",style={"color":"gray"})], style=kpi_style),
        html.Div([html.H3(f"{total_orders:,}"),
                  html.P("Total Orders",style={"color":"gray"})], style=kpi_style),
        html.Div([html.H3(f"{profit_margin:.1f}%",
                          style={"color":"green" if profit_margin>0 else "red"}),
                  html.P("Profit Margin",style={"color":"gray"})], style=kpi_style),
    ]

    # Sales Trend
    trend = filtered.groupby(filtered["Order Date"].dt.to_period("M")).agg(
        {"Sales":"sum"}).reset_index()
    trend["Order Date"] = trend["Order Date"].astype(str)
    fig_trend = px.line(trend, x="Order Date", y="Sales",
                        title="Monthly Sales Trend",
                        template="plotly_white")

    # Category Pie
    cat_sales = filtered.groupby("Category")["Sales"].sum().reset_index()
    fig_pie = px.pie(cat_sales, values="Sales", names="Category",
                     title="Sales by Category", hole=0.4)

    # Region Bar
    reg_data = filtered.groupby("Region").agg(
        {"Sales":"sum","Profit":"sum"}).reset_index()
    fig_region = px.bar(reg_data, x="Region", y=["Sales","Profit"],
                        barmode="group", title="Sales & Profit by Region",
                        template="plotly_white")

    # Profit Scatter
    fig_scatter = px.scatter(filtered, x="Sales", y="Profit",
                             color="Category", size="Quantity",
                             title="Sales vs Profit",
                             template="plotly_white", opacity=0.6)

    # Top Products
    top_prod = filtered.groupby("Product Name")["Sales"].sum() \
        .sort_values(ascending=False).head(10).reset_index()
    fig_products = px.bar(top_prod, x="Sales", y="Product Name",
                          orientation="h", title="Top 10 Products by Sales",
                          template="plotly_white")

    # Segment Bar
    seg_data = filtered.groupby("Segment").agg(
        {"Sales":"sum","Profit":"sum"}).reset_index()
    fig_segment = px.bar(seg_data, x="Segment", y=["Sales","Profit"],
                         barmode="group", title="Sales & Profit by Segment",
                         template="plotly_white")

    return kpis, fig_trend, fig_pie, fig_region, fig_scatter, fig_products, fig_segment


if __name__ == "__main__":
    app.run(debug=True)