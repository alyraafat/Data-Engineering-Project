from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
import argparse

def create_dashboard(dataset_path: str):
    df = pd.read_parquet(dataset_path)

    app = Dash()
    app.title = "Fintech Dashboard"  

    app.layout = html.Div(children=[
        # Title and subtitle
        html.H1("Fintech Dashboard", style={"textAlign": "center"}),
        html.H2("Aly Raafat AbdelFattah Aly, 52-1008", style={"textAlign": "center"}),  

    
        # 1st graph
        html.H3("1. Distribution of loan amounts across different grades:"),
        dcc.Graph(id="grade-loan-graph"), 


        # 2nd graph
        html.H3("2. Loan Amount vs Annual Income Across States:"),
        dcc.Dropdown(
            id="state-filter",
            options=[{"label": "All", "value": "all"}] + [{"label": state, "value": state} for state in df['state'].unique()],
            value="all",
            placeholder="Select a state",
        ),
        dcc.Graph(id="loan-income-graph"),
        

        # 3rd graph
        html.H3("3. Trend of loan issuance over the months, filtered by year:"),
        dcc.Dropdown(
            id="year-filter",
            options=[{"label": year, "value": year} for year in np.sort(df['issue_date_cleaned'].dt.year.unique())],
            value=df['issue_date_cleaned'].dt.year.min(),
            placeholder="Select a year",
        ),
        dcc.RadioItems(
            id="aggregation-method",
            options=[
                {"label": "Number of Loans/Month", "value": "count"},
                {"label": "Total Number of Loans/Month", "value": "sum"}
            ],
            value="count",  
            inline=True, 
        ),
        dcc.Graph(id="loan-trend-graph"),
        

        # 4th graph
        html.H3("4. States with the highest average loan amount:"),
        # html.Div([
        #     html.Label("Enter the number of top states to display:"),
        #     dcc.Input(
        #         id="top-states-input",
        #         type="number",
        #         value=5,  
        #         min=1,
        #         max=len(df["state"].unique()),   
        #         placeholder="Enter a number",
        #     ),
        # ]),
        dcc.Graph(id="average-loan-graph"),


        # 5th graph
        html.H3("5. Percentage distribution of loan grades:"),
        dcc.Graph(id="grade-distribution-graph"),
    ])

    callbacks(app, df)

    app.run_server(debug=True, host='0.0.0.0', port=9051, threaded=False)


def callbacks(app: Dash, df: pd.DataFrame):
    plot_q1(app, df)
    plot_q2(app, df)
    plot_q3(app, df)
    # plot_q4(app, df)    
    plot_q4_part2(app, df)
    plot_q5(app, df)

def plot_q1(app: Dash, df: pd.DataFrame):
    @app.callback(
        Output("grade-loan-graph", "figure"),
        Input("grade-loan-graph", "id")
    )
    def update_grade_loan_graph(_):
        df_copy = df.copy()
        df_copy = df.sort_values("letter_grade", ascending=True)
        fig = px.box(
            df_copy,
            x="letter_grade",  
            y="loan_amount",  
            title="Distribution of Loan Amounts Across Grades",
            labels={"letter_grade": "Grade", "loan_amount": "Loan Amount"},
            color="letter_grade",  
            template="plotly_white", 
        )
        fig.update_layout(showlegend=True)  
        return fig
    
def plot_q2(app: Dash, df: pd.DataFrame):
    @app.callback(
        Output("loan-income-graph", "figure"),
        Input("state-filter", "value")
    )
    def update_loan_income_graph(selected_state: str):
        if selected_state != "all":
            filtered_df = df[df["state"] == selected_state]
        else:
            filtered_df = df

        fig = px.scatter(
            filtered_df,
            x="annual_inc",  
            y="loan_amount",
            color="loan_status", 
            title="Loan Amount vs Annual Income Across States",
            labels={"annual_inc": "Annual Income", "loan_amount": "Loan Amount", "loan_status": "Loan Status"},
            template="plotly_white",  
            hover_data=["state"], 
        )
        fig.update_layout(
            legend_title_text="Loan Status",  
            xaxis_title="Annual Income",  
            yaxis_title="Loan Amount",  
        )
        return fig
    
def plot_q3(app: Dash, df: pd.DataFrame):
    @app.callback(
        Output("loan-trend-graph", "figure"),
        [
            Input("year-filter", "value"),
            Input("aggregation-method", "value")
        ]
    )
    def update_loan_trend_graph(selected_year: int, aggregation_method: str):
        filtered_df = df[df["issue_date_cleaned"].dt.year == selected_year]
        if aggregation_method == "count":
            y_axis_label = "Number of Loans"
        else:
            y_axis_label = "Total Loan Amount"

        monthly_trend = (
            filtered_df.groupby(filtered_df["issue_date_cleaned"].dt.month)
            .agg(loan_trend=("loan_amount", aggregation_method))
            .reset_index()
            .rename(columns={"issue_date_cleaned": "month"})
        )
        fig = px.line(
            monthly_trend,
            x="month",
            y="loan_trend", 
            title=f"Loan Issuance Trend for {selected_year}",
            labels={"month": "Month", "loan_trend": y_axis_label},
            markers=True,
            template="plotly_white", 
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(1, 13)),  
            ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        )

        return fig
    
def plot_q4(app: Dash, df: pd.DataFrame):
    @app.callback(
        Output("average-loan-graph", "figure"),
        Input("top-states-input", "value")
    )
    def update_average_loan_graph(top_n: int):
        state_avg_loan = (
            df.groupby("state")
            .agg(average_loan_amount=("loan_amount", "mean"))
            .sort_values("average_loan_amount", ascending=False)
            .reset_index()
        )
        if top_n is None or top_n == 0:
            top_n = len(state_avg_loan)

        state_avg_loan = state_avg_loan.head(top_n)
        fig = px.bar(
            state_avg_loan,
            x="state",
            y="average_loan_amount",
            title=f"Top {top_n} States with the Highest Average Loan Amount",
            labels={"state": "State", "average_loan_amount": "Average Loan Amount"},
            color="state",  
            template="plotly_white",  
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=0)  
        return fig

def plot_q4_part2(app: Dash, df: pd.DataFrame):
    @app.callback(
        Output("average-loan-graph", "figure"),
        Input("average-loan-graph", "id") 
    )
    def update_average_loan_graph(_):
        state_avg_loan = (
            df.groupby("state")
            .agg(average_loan_amount=("loan_amount", "mean"))
            .reset_index()
        )

        fig = px.choropleth(
            state_avg_loan,
            locations="state",  
            locationmode="USA-states",  
            color="average_loan_amount",  
            color_continuous_scale="Viridis", 
            scope="usa", 
            labels={"average_loan_amount": "Avg Loan Amount"},
            title="Choropleth Map of Average Loan Amount by State",
        )
        fig.update_layout(
            geo=dict(bgcolor="rgba(0,0,0,0)"), 
        )
        return fig
    
def plot_q5(app: Dash, df: pd.DataFrame):
    @app.callback(
        Output("grade-distribution-graph", "figure"),
        Input("grade-distribution-graph", "id")
    )
    def update_grade_distribution_graph(_):
        grade_distribution = (
            (df["letter_grade"].value_counts(normalize=True) * 100)
            .reset_index()
            .sort_values('letter_grade',ascending=True)
        )
        fig = px.histogram(
            grade_distribution,
            x="letter_grade",
            y="proportion", 
            title="Percentage Distribution of Loan Grades",
            labels={"letter_grade": "Loan Grade", "count": "Count"},
            color="letter_grade", 
            template="plotly_white",  
        )
        fig.update_layout(
            xaxis_title="Loan Grade",
            yaxis_title="Percentage (%)",
            showlegend=False, 
        )
        return fig
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fintech dashboard")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    args = parser.parse_args()

    create_dashboard(dataset_path=args.dataset_path)