import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_splits_for_output(df):
    """
    Format split data to match the expected CSV output format.

    Parameters:
    - df: DataFrame with raw split data

    Returns:
    - DataFrame formatted like activity_20738496042_splits.csv
    """
    logger.info(f"Formatting {len(df)} splits for output")

    # Create a copy to avoid modifying original
    df_formatted = df.copy()

    # Format Time as HH:MM:SS
    df_formatted['Time'] = df_formatted['time_seconds'].apply(lambda x:
        f"{int(x // 3600):02d}:{int((x % 3600) // 60):02d}:{int(x % 60):02d}"
    )

    # Format Distance as "XXXm"
    df_formatted['Distance'] = df_formatted['distance_meters'].apply(lambda x: f"{int(x)}m")

    # Format Avg Pace as MM:SS
    df_formatted['Avg Pace'] = df_formatted['avg_pace_seconds'].apply(lambda x:
        f"{int(x // 60)}:{int(x % 60):02d}"
    )

    # Format Avg HR (keep as integer or None)
    df_formatted['Avg HR'] = df_formatted['avg_hr'].apply(lambda x: int(x) if pd.notna(x) else None)

    # Format Total Ascent as "XXm"
    df_formatted['Total Ascent'] = df_formatted['total_ascent_meters'].apply(lambda x: f"{int(x)}m")

    # Format Total Descent as "XXm"
    df_formatted['Total Descent'] = df_formatted['total_descent_meters'].apply(lambda x: f"{int(x)}m")

    # Rename lap column
    df_formatted = df_formatted.rename(columns={'lap': 'Lap'})

    # Select only the columns in the expected format
    output_columns = ['Lap', 'Time', 'Distance', 'Avg Pace', 'Avg HR', 'Total Ascent', 'Total Descent']
    df_output = df_formatted[output_columns]

    logger.info(f"Formatting complete: {len(df_output.columns)} columns")

    return df_output


def save_processed_data(activities_dict, output_dir="data/processed"):
    """
    Save processed activity data to CSV files.

    Parameters:
    - activities_dict: Dictionary mapping activity IDs to DataFrames
    - output_dir: Directory to save processed CSV files

    Returns:
    - List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path.absolute()}")

    saved_files = []

    for activity_id, df in activities_dict.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing activity: {activity_id}")
        logger.info(f"{'='*60}")

        # Format splits to match expected CSV format
        df_processed = format_splits_for_output(df)

        # Save to CSV with _splits suffix to match expected format
        output_file = output_path / f"{activity_id}_splits.csv"
        df_processed.to_csv(output_file, index=False)
        saved_files.append(output_file)

        logger.info(f"Saved processed data to: {output_file}")
        logger.info(f"Shape: {df_processed.shape}")
        logger.info(f"Columns: {list(df_processed.columns)}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Saved {len(saved_files)} processed files to {output_dir}")
    logger.info(f"{'='*60}\n")

    return saved_files


def process_activities(activities_dict, output_dir="data/processed"):
    """
    Main function to process activities and save results.

    Parameters:
    - activities_dict: Dictionary mapping activity IDs to DataFrames with raw split data
    - output_dir: Directory to save processed CSV files

    Returns:
    - Dictionary mapping activity IDs to processed DataFrames
    """
    logger.info("Starting feature engineering pipeline")

    if not activities_dict:
        logger.warning("No activities to process")
        return {}

    logger.info(f"Processing {len(activities_dict)} activities")

    # Save processed data
    save_processed_data(activities_dict, output_dir)

    # Load and return processed data
    processed_activities = {}
    for activity_id in activities_dict.keys():
        file_path = Path(output_dir) / f"{activity_id}_splits.csv"
        processed_activities[activity_id] = pd.read_csv(file_path)

    logger.info("Feature engineering pipeline complete")

    return processed_activities


if __name__ == "__main__":
    # Test with sample data
    logger.info("Running feature engineering test")

    # Create sample data
    sample_df = pd.DataFrame({
        'lap': range(1, 6),
        'time_seconds': [300, 310, 305, 320, 315],
        'distance_meters': [1000, 1000, 1000, 1000, 1000],
        'avg_pace_seconds': [300, 310, 305, 320, 315],
        'avg_hr': [150, 155, 152, 160, 158],
        'total_ascent_meters': [10, 15, 5, 20, 8],
        'total_descent_meters': [8, 10, 12, 5, 15]
    })

    processed = format_splits_for_output(sample_df)
    logger.info(f"\nSample processed data:\n{processed.head()}")
