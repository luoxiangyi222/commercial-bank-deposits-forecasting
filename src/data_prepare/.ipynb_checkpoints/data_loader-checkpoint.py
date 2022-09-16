import pandas as pd

def load_data(filename, insample_end_year):
    
    original_df = pd.read_csv(filename)

    df = original_df.copy()

    ###### drop some useless columns
    df = df.drop(
        columns=[
            "domestic_debt",  # remove, use total natinal debt
            "external_debt",  # remove, use total natinal debt
            "Compound SORA - 1 month",  # remove since lack of data
            "Compound SORA - 3 month",  # remove since lack of data
            "Compound SORA - 6 month",  # remove since lack of data
            "average_MNE_per_employee_overall_economy",  # remove since lack of data
            "Resident Unemployment Rate",  # remove, use total unemployment rate
            "Citizen Unemployment Rate",  # remove, use total unemployment rate
            "GDP_year_on_year_growth_rate_current_price",  # remove, use in-chain data
            "household_net_worth",  # remove, consider as irrelevant
            "gross_fixed_capital_formation_current_price",  # remove, use in chain
        ]
    )

    df = df.rename(
        columns={
            "lag3q_Age Dependency Ratio: Residents Aged Under 15 Years And 65 Years Per Hundred Residents Aged 15-64 Years (Number)": "lag3q_Age Dependency Ratio (15-64 Years)"
        }
    )

    # add quarter information
    # TODO: one hot encoding for Quarter
    df["Year"] = df["Data Series"].apply(lambda s: int(s.split(" ")[0]))
    df["Qrt"] = df["Data Series"].apply(lambda s: int(s.split(" ")[1][0]))
    
    df=df.dropna().reset_index(drop=True)
    
    #### 3-year backtesting

    # insample: ~2017 Q4
    # outsample: 2018 Q1 ~

    insample_df = df[df["Year"] <= insample_end_year]
    outsample_df = df[df["Year"] > insample_end_year]

    return (insample_df, outsample_df, df)
