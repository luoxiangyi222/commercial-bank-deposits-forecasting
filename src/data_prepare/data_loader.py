import pandas as pd
import numpy as np


def load_data(filename, insample_end_year: int):
    original_df = pd.read_csv(filename)

    df = original_df.copy()

    # Drop some useless columns
    df = df.drop(
        columns=[
            "domestic_debt",  # remove, use total national debt
            "external_debt",  # remove, use total national debt
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

    # rename long name into shorter name
    df = df.rename(
        columns={
            "lag3q_Age Dependency Ratio: Residents Aged Under 15 Years And 65 Years Per Hundred Residents Aged 15-64 "
            "Years (Number)": "lag3q_Age Dependency Ratio (15-64 Years) "
        }
    )

    # Add year and quarter number
    df["Year"] = df["Data Series"].apply(lambda s: int(s.split(" ")[0]))
    df["Qrt"] = df["Data Series"].apply(lambda s: int(s.split(" ")[1][0]))
    df["Qrt"] = ( # Cycling encoding for quarter number
        df["Qrt"].apply(lambda x: np.sin(x * (1 / 4) * (2 * np.pi))).astype("int32")
    )

    # Add

    full_df = df.dropna().reset_index(drop=True)
    full_df = full_df[full_df['Year'] <= 2020]

    #### 3-year backtesting

    # in sample: ~2017 Q4
    # out sample: 2018 Q1 ~ 2020 Q4

    insample_df = full_df[full_df["Year"] <= insample_end_year]
    outsample_df = full_df[(full_df["Year"] > insample_end_year)]

    return insample_df, outsample_df, full_df

