"""Data loading and initial preprocessing utilities for GTD project."""

from typing import Optional
import pandas as pd


def load_gtd_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load GTD dataset (Excel or CSV) and report basic info.

    Parameters
    ----------
    filepath : str
        Path to the dataset file (Excel or CSV).

    Returns
    -------
    df : pd.DataFrame or None
        Loaded dataframe or None on error.
    """
    print("\n" + "=" * 60)
    print("BLOCK: DATA LOADING")
    print("=" * 60)

    try:
        if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)

        print("Dataset loaded successfully!")
        print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def initial_data_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Print a brief data profile and return missing-value summary.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded dataset.

    Returns
    -------
    missing_df : pd.DataFrame
        DataFrame with columns: Missing_Count and Missing_Percentage.
    """
    print("\nINITIAL DATA PROFILE")
    print("-" * 80)

    print(f"Dataset Period: {df['iyear'].min()} - {df['iyear'].max()}")
    print(f"Total Incidents: {len(df):,}")
    print(f"Total Features: {len(df.columns)}")

    missing = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': (missing / len(df)) * 100
    }).sort_values('Missing_Percentage', ascending=False)

    print(missing_df[missing_df['Missing_Count'] > 0].head(15))
    print("\nData types distribution:")
    print(df.dtypes.value_counts())

    return missing_df
