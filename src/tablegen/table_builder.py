# src/tablegen/table_builder.py

import pandas as pd
from typing import List
from .row_schema import SpringProgramRow

def build_dataframe_from_rows(rows: List[SpringProgramRow]) -> pd.DataFrame:
    return pd.DataFrame([row.dict() for row in rows])

def save_table_as_csv(rows: List[SpringProgramRow], path: str):
    df = build_dataframe_from_rows(rows)
    df.to_csv(path, index=False)
    print(f"✅ Table saved to {path}")
