import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth using the Haversine formula.
    Returns distance in meters.
    """
    R = 6371000  # Earth's radius in meters

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance


def calculate_elevation_gain_loss(elevations, threshold=2.0):
    """
    Calculate elevation gain and loss using Garmin-style algorithm.
    Accumulates elevation changes and only counts when threshold is exceeded.

    Parameters:
    - elevations: List of elevation values in meters
    - threshold: Minimum accumulated change to count (default: 2.0m)

    Returns:
    - (total_ascent, total_descent) in meters
    """
    if len(elevations) < 2:
        return 0, 0

    total_ascent = 0
    total_descent = 0
    accumulated_gain = 0
    accumulated_loss = 0

    for i in range(len(elevations) - 1):
        diff = elevations[i+1] - elevations[i]

        if diff > 0:
            # Going up
            accumulated_gain += diff
            accumulated_loss = 0  # Reset descent accumulator

            # If we've accumulated enough gain, count it
            if accumulated_gain >= threshold:
                total_ascent += accumulated_gain
                accumulated_gain = 0
        else:
            # Going down
            accumulated_loss += abs(diff)
            accumulated_gain = 0  # Reset ascent accumulator

            # If we've accumulated enough loss, count it
            if accumulated_loss >= threshold:
                total_descent += accumulated_loss
                accumulated_loss = 0

    # Add any remaining accumulated values
    if accumulated_gain >= threshold:
        total_ascent += accumulated_gain
    if accumulated_loss >= threshold:
        total_descent += accumulated_loss

    return total_ascent, total_descent


def parse_gpx_to_splits(gpx_file_path, split_distance=1000, elevation_threshold=2.0):
    """
    Parse GPX file and generate 1km split data.

    Parameters:
    - gpx_file_path: Path to the GPX file
    - split_distance: Distance for each split in meters (default: 1000m = 1km)
    - elevation_threshold: Minimum accumulated elevation change to count (default: 2.0m)

    Returns:
    - DataFrame with split data
    """
    logger.info(f"Parsing GPX file: {gpx_file_path}")

    # Parse XML
    tree = ET.parse(gpx_file_path)
    root = tree.getroot()

    # Define namespaces
    ns = {
        'gpx': 'http://www.topografix.com/GPX/1/1',
        'ns3': 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1'
    }

    # Extract all track points
    logger.info("Extracting track points from GPX file")
    track_points = []
    for trkpt in root.findall('.//gpx:trkpt', ns):
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))

        ele_elem = trkpt.find('gpx:ele', ns)
        ele = float(ele_elem.text) if ele_elem is not None else 0

        time_elem = trkpt.find('gpx:time', ns)
        time = datetime.fromisoformat(time_elem.text.replace('Z', '+00:00')) if time_elem is not None else None

        # Extract heart rate
        hr_elem = trkpt.find('.//ns3:hr', ns)
        hr = int(hr_elem.text) if hr_elem is not None else None

        track_points.append({
            'lat': lat,
            'lon': lon,
            'ele': ele,
            'time': time,
            'hr': hr
        })

    logger.info(f"Extracted {len(track_points)} track points")

    # Calculate splits
    logger.info(f"Calculating splits with distance={split_distance}m, elevation_threshold={elevation_threshold}m")
    splits = []
    current_split = {
        'lap': 1,
        'distance': 0,
        'start_time': track_points[0]['time'],
        'hr_values': [],
        'elevations': []
    }

    total_distance = 0
    prev_point = track_points[0]

    for point in track_points[1:]:
        # Calculate distance from previous point
        dist = haversine_distance(
            prev_point['lat'], prev_point['lon'],
            point['lat'], point['lon']
        )

        current_split['distance'] += dist
        total_distance += dist

        # Add HR and elevation data
        if point['hr'] is not None:
            current_split['hr_values'].append(point['hr'])
        current_split['elevations'].append(point['ele'])

        # Check if we've completed a split
        if current_split['distance'] >= split_distance:
            # Calculate split metrics
            end_time = point['time']
            duration = (end_time - current_split['start_time']).total_seconds()

            # Calculate average pace (min/km)
            avg_pace_seconds = duration / (current_split['distance'] / 1000)

            # Calculate average HR
            avg_hr = int(sum(current_split['hr_values']) / len(current_split['hr_values'])) if current_split['hr_values'] else None

            # Calculate elevation gain/loss using Garmin-style algorithm
            total_ascent, total_descent = calculate_elevation_gain_loss(current_split['elevations'], elevation_threshold)

            splits.append({
                'lap': current_split['lap'],
                'time_seconds': duration,
                'distance_meters': current_split['distance'],
                'avg_pace_seconds': avg_pace_seconds,
                'avg_hr': avg_hr,
                'total_ascent_meters': total_ascent,
                'total_descent_meters': total_descent
            })

            logger.debug(f"Completed lap {current_split['lap']}: {current_split['distance']:.0f}m, {duration:.1f}s")

            # Start new split
            current_split = {
                'lap': current_split['lap'] + 1,
                'distance': 0,
                'start_time': point['time'],
                'hr_values': [],
                'elevations': [point['ele']]
            }

        prev_point = point

    # Handle remaining distance (last incomplete split)
    if current_split['distance'] > 0:
        end_time = track_points[-1]['time']
        duration = (end_time - current_split['start_time']).total_seconds()

        avg_pace_seconds = duration / (current_split['distance'] / 1000) if current_split['distance'] > 0 else 0
        avg_hr = int(sum(current_split['hr_values']) / len(current_split['hr_values'])) if current_split['hr_values'] else None
        total_ascent, total_descent = calculate_elevation_gain_loss(current_split['elevations'], elevation_threshold)

        splits.append({
            'lap': current_split['lap'],
            'time_seconds': duration,
            'distance_meters': current_split['distance'],
            'avg_pace_seconds': avg_pace_seconds,
            'avg_hr': avg_hr,
            'total_ascent_meters': total_ascent,
            'total_descent_meters': total_descent
        })

        logger.debug(f"Completed final lap {current_split['lap']}: {current_split['distance']:.0f}m, {duration:.1f}s")

    df = pd.DataFrame(splits)
    logger.info(f"Generated {len(df)} splits from GPX file")
    logger.info(f"Total distance: {total_distance:.0f}m")
    logger.info(f"Total elevation gain: {df['total_ascent_meters'].sum():.0f}m")
    logger.info(f"Total elevation loss: {df['total_descent_meters'].sum():.0f}m")

    return df


def extract_features_from_gpx_files(raw_data_dir="data/raw", split_distance=1000, elevation_threshold=2.0):
    """
    Extract features from all GPX files in the raw data directory.

    Parameters:
    - raw_data_dir: Directory containing GPX files
    - split_distance: Distance for each split in meters (default: 1000m = 1km)
    - elevation_threshold: Minimum accumulated elevation change to count (default: 2.0m)

    Returns:
    - Dictionary mapping activity IDs to DataFrames with split data
    """
    raw_path = Path(raw_data_dir)

    if not raw_path.exists():
        logger.error(f"Raw data directory does not exist: {raw_data_dir}")
        raise FileNotFoundError(f"Directory not found: {raw_data_dir}")

    gpx_files = list(raw_path.glob("*.gpx"))
    logger.info(f"Found {len(gpx_files)} GPX files in {raw_data_dir}")

    if not gpx_files:
        logger.warning(f"No GPX files found in {raw_data_dir}")
        return {}

    activities = {}
    for gpx_file in gpx_files:
        activity_id = gpx_file.stem  # filename without extension
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing activity: {activity_id}")
        logger.info(f"{'='*60}")

        try:
            df = parse_gpx_to_splits(str(gpx_file), split_distance, elevation_threshold)
            activities[activity_id] = df
            logger.info(f"Successfully processed {activity_id}")
        except Exception as e:
            logger.error(f"Error processing {activity_id}: {str(e)}")
            continue

    logger.info(f"\n{'='*60}")
    logger.info(f"Feature extraction complete: {len(activities)}/{len(gpx_files)} activities processed successfully")
    logger.info(f"{'='*60}\n")

    return activities


if __name__ == "__main__":
    # Test the extraction
    activities = extract_features_from_gpx_files()

    if activities:
        # Display summary for first activity
        first_activity = list(activities.keys())[0]
        logger.info(f"\nSample data for {first_activity}:")
        logger.info(f"\n{activities[first_activity].head()}")
