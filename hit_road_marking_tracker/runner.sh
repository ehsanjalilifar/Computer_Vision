#!/bin/bash

directory="input_files/Videos/Stream_2"

echo "$directory"

if [ ! -d "$directory" ]; then
    echo "The provided directory does not exist."
    exit 1
fi

for file in "$directory"/*.mkv; do
    if [ -f "$file" ]; then
        # Remove the .mkv extension
        filename=$(basename "$file")
        filename_without_extension="${filename%.mkv}"
        echo "Processing file: $filename"
        python car_hit_line_detection.py --model "yolov8m-seg.pt" --source "$filename" --lanes "NTTA_Entry_Round1_lanes" --markings "NTTA_Entry_Round1_lines" --rois "NTTA_Entry_Round1_ROIs" --zoom "NTTA_Entry_Round1_zoom" --skipframes 1 | tee logs/"$filename_without_extension".txt
    fi
done