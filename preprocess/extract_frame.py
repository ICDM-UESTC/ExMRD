import os
import sys
import subprocess
import glob
import shutil
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_video_duration(video_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get video duration {video_path}: {e}")
        return None
    except ValueError:
        logging.error(f"Failed to parse video duration {video_path}")
        return None

def get_video_framerate(video_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, check=True
        )
        framerate = result.stdout.strip().split('/')
        return float(framerate[0]) / float(framerate[1])
    except (subprocess.CalledProcessError, ValueError, ZeroDivisionError) as e:
        logging.warning(f"Failed to get video framerate {video_path}, using default value 30: {e}")
        return 30

def extract_frames(video_path, output_folder, num_frames):
    duration = get_video_duration(video_path)
    if duration is None or duration <= 0:
        logging.error(f"Invalid video file or duration: {video_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)

    if os.path.exists(video_output_folder):
        existing_frames = glob.glob(os.path.join(video_output_folder, "frame_*.jpg"))
        if len(existing_frames) == num_frames:
            logging.info(f"Skipping video {video_name}: Already have {num_frames} frames")
            return
        shutil.rmtree(video_output_folder)
    
    os.makedirs(video_output_folder, exist_ok=True)

    interval = duration / num_frames
    timestamps = [i * interval for i in range(num_frames)]
    framerate = get_video_framerate(video_path)

    for i, ts in enumerate(timestamps):
        output_file = os.path.join(video_output_folder, f"frame_{i:03d}.jpg")
        try:
            subprocess.run(
                ['ffmpeg', '-loglevel', 'error', '-ss', str(ts), '-i', video_path, '-frames:v', '1', '-q:v', '2', output_file],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Error processing frame {i} of video {video_name}: {e}")
            logging.error(f"ffmpeg error: {e.stderr}")

    created_frames = glob.glob(os.path.join(video_output_folder, "frame_*.jpg"))
    if len(created_frames) != num_frames:
        logging.warning(f"Warning: Video {video_name} only created {len(created_frames)} frames instead of expected {num_frames} frames")

def process_dataset(dataset_name, num_frames):
    input_folder = f'data/{dataset_name}/videos'
    output_folder = f'data/{dataset_name}/frames_{num_frames}'

    if not os.path.exists(input_folder):
        logging.error(f"Error: Input folder does not exist: {input_folder}")
        return False

    os.makedirs(output_folder, exist_ok=True)

    video_paths = glob.glob(os.path.join(input_folder, '*.mp4'))
    
    if not video_paths:
        logging.warning(f"No videos found in {input_folder}")
        return False
    
    logging.info(f"Processing {len(video_paths)} videos from {dataset_name}")
    for video_path in tqdm(video_paths, desc=f"Processing {dataset_name}"):
        extract_frames(video_path, output_folder, num_frames)
    
    return True

def main():
    num_frames = 16
    datasets = ["FakeSV", "FakeTT", "FVC"]
    
    processed_count = 0
    for dataset in datasets:
        if process_dataset(dataset, num_frames):
            processed_count += 1
    
    if processed_count == 0:
        logging.error("No datasets were processed successfully")
        sys.exit(1)
    else:
        logging.info(f"Successfully processed {processed_count}/{len(datasets)} datasets")

if __name__ == "__main__":
    main()