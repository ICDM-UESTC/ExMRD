#!/bin/bash

# Run textual refining
python preprocess/run_textual_refining.py

# Run visual refining
python preprocess/run_visual_refining.py

# Run retrieving
python preprocess/run_retrieving.py

# Run reasoning
python preprocess/run_reasoning.py