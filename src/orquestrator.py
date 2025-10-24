import logging
from pathlib import Path
from extract_features import extract_features_from_gpx_files
from feature_engineering import process_activities

# Configure logging for orchestrator
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def run_pipeline(raw_data_dir="data/raw", processed_data_dir="data/processed",
                 split_distance=1000, elevation_threshold=2.0):
    """
    Run the complete data processing pipeline.

    Pipeline steps:
    1. Extract features from GPX files
    2. Engineer features and save to CSV

    Parameters:
    - raw_data_dir: Directory containing raw GPX files
    - processed_data_dir: Directory to save processed CSV files
    - split_distance: Distance for each split in meters (default: 1000m)
    - elevation_threshold: Minimum accumulated elevation change to count (default: 2.0m)

    Returns:
    - Dictionary of processed activities
    """
    logger.info("\n" + "="*80)
    logger.info("STARTING SMART PEAK SPLITS PIPELINE")
    logger.info("="*80)

    # Validate input directory
    raw_path = Path(raw_data_dir)
    if not raw_path.exists():
        logger.error(f"Raw data directory does not exist: {raw_data_dir}")
        raise FileNotFoundError(f"Directory not found: {raw_data_dir}")

    # Step 1: Extract features from GPX files
    logger.info("\n" + "-"*80)
    logger.info("STEP 1: EXTRACTING FEATURES FROM GPX FILES")
    logger.info("-"*80)

    try:
        activities = extract_features_from_gpx_files(
            raw_data_dir=raw_data_dir,
            split_distance=split_distance,
            elevation_threshold=elevation_threshold
        )

        if not activities:
            logger.warning("No activities extracted. Pipeline terminated.")
            return {}

        logger.info(f" Successfully extracted features from {len(activities)} activities")

    except Exception as e:
        logger.error(f" Feature extraction failed: {str(e)}")
        raise

    # Step 2: Feature engineering and save to CSV
    logger.info("\n" + "-"*80)
    logger.info("STEP 2: ENGINEERING FEATURES AND SAVING TO CSV")
    logger.info("-"*80)

    try:
        processed_activities = process_activities(
            activities_dict=activities,
            output_dir=processed_data_dir
        )

        logger.info(f" Successfully processed and saved {len(processed_activities)} activities")

    except Exception as e:
        logger.error(f" Feature engineering failed: {str(e)}")
        raise

    # Pipeline summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*80)

    for activity_id, df in processed_activities.items():
        logger.info(f"\nActivity: {activity_id}")
        logger.info(f"  - Splits: {len(df)}")

        # Parse distance from "XXXm" format
        total_distance = df['Distance'].apply(lambda x: float(x.replace('m', ''))).sum()
        logger.info(f"  - Total distance: {total_distance:.0f}m")

        # Parse time from "HH:MM:SS" format
        def parse_time(time_str):
            parts = time_str.split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

        total_time = df['Time'].apply(parse_time).sum()
        logger.info(f"  - Total time: {total_time:.0f}s ({total_time/3600:.2f}h)")

        # Parse elevation from "XXm" format
        total_ascent = df['Total Ascent'].apply(lambda x: float(x.replace('m', ''))).sum()
        total_descent = df['Total Descent'].apply(lambda x: float(x.replace('m', ''))).sum()
        logger.info(f"  - Total elevation gain: {total_ascent:.0f}m")
        logger.info(f"  - Total elevation loss: {total_descent:.0f}m")

        # Parse pace from "MM:SS" format
        def parse_pace(pace_str):
            parts = pace_str.split(':')
            return int(parts[0]) + int(parts[1]) / 60

        avg_pace = df['Avg Pace'].apply(parse_pace).mean()
        logger.info(f"  - Average pace: {avg_pace:.2f} min/km")

        if df['Avg HR'].notna().any():
            logger.info(f"  - Average HR: {df['Avg HR'].mean():.0f} bpm")

    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80 + "\n")

    return processed_activities


if __name__ == "__main__":
    logger.info("Starting pipeline execution from orchestrator")

    try:
        # Run the pipeline with default parameters
        processed_data = run_pipeline(
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            split_distance=1000,
            elevation_threshold=2.0
        )

        logger.info(f"\nPipeline executed successfully!")
        logger.info(f"Processed {len(processed_data)} activities")
        logger.info(f"Results saved to: data/processed/")
        logger.info(f"Log saved to: pipeline.log")

    except Exception as e:
        logger.error(f"\nPipeline failed with error: {str(e)}")
        raise
