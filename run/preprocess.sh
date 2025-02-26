#!/bin/bash

# preprocess for video
python preprocess/video_to_wav.py       # convert videos to wavs

python preprocess/extract_frame.py      # extract frames from videos

python preprocess/wav_to_transcript.py  # convert wavs to transcripts

python preprocess/frames_to_quad_4.py    # convert frames to quads

python preprocess/video_to_ocr_en.py     # extract ocr from videos (for
# other dataset, you can fetch ocr from its data file)

python preprocess/make_video_feature.py  # extract video features