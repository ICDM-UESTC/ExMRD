import os
import subprocess
from tqdm import tqdm

def convert_mp4_to_wav(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all mp4 files
    mp4_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    
    # Store names of files that couldn't be converted
    unconverted_files = []
    
    # Create progress bar using tqdm
    for filename in tqdm(mp4_files, desc="Conversion progress"):
        input_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + '.wav'
        output_path = os.path.join(output_folder, output_filename)
        
        # Check if already converted
        if os.path.exists(output_path):
            continue
        
        # Build FFmpeg command
        ffmpeg_command = [
            'ffmpeg',
            '-i', input_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-loglevel', 'error',  # Reduce FFmpeg output
            output_path
        ]
        
        try:
            # Execute FFmpeg command
            subprocess.run(ffmpeg_command, check=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f'\nError converting {filename}: {e.stderr.decode()}')
            unconverted_files.append(filename)

    # Print names of files that couldn't be converted
    if unconverted_files:
        print("\nThe following files could not be converted:")
        for file in unconverted_files:
            print(file)
    else:
        print("\nAll files have been successfully converted!")

# Iterate through datasets: FakeSV, FakeTT, and FVC
datasets = ["FakeSV", "FakeTT", "FVC"]

for dataset in datasets:
    print(f"\nProcessing {dataset} dataset...")
    input_folder = f'data/{dataset}/videos'
    output_folder = f'data/{dataset}/audios'
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder for {dataset} does not exist. Skipping...")
        continue
        
    convert_mp4_to_wav(input_folder, output_folder)
    print(f"{dataset} conversion complete!")

print("\nAll datasets processed!")