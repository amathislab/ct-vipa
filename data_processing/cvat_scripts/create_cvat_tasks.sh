#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <video_folder> <annotation_folder> <share_basepath> <project_id> <credentials> <cvat_org>"
    exit 1
fi

# Store command line arguments
video_folder="$1"
annotation_folder="$2"
share_basepath="$3"
project_id="$4"
credentials="$5"
cvat_org="$6"

# Check if folders exist
if [ ! -d "$video_folder" ] || [ ! -d "$annotation_folder" ] || [ ! -d "$share_basepath" ]; then
    echo "Error: One of the directories do not exist"
    exit 1
fi

# Count total number of mp4 files
total_files=$(find "$video_folder" -type f | wc -l)
current=0
echo "Total files to upload to cvat: $total_files"

# Get existing task names for this project
tasks_json=$(cvat-cli --server-port 8080 --auth $credentials task ls --json)
task_names=$(echo "$tasks_json" | jq -r --arg pid "$project_id" '.[] | select(.project_id == ($pid|tonumber)) | .name')

# Find all mp4 files recursively and loop through them
find "$video_folder" -type f | while read video_path; do
    # Update progress
    ((current++))
    progress=$((current * 100 / total_files))
    printf "\rProgress: [%-50s] %d%%" $(printf "#%.0s" $(seq 1 $((progress/2)))) $progress

    # Extract filename without extension
    rel_path=$(realpath --relative-to="$video_folder" "$video_path")
    filename=${rel_path%.*}
    share_video_rel_path=$(realpath --relative-to="$share_basepath" "$video_path")
    # Check if task already exists
    if echo "$task_names" | grep -q "^${filename##*/}$"; then
        echo -e "\nSkipping ${filename##*/} - task already exists"
        continue
    fi
    # Construct the corresponding xml path
    xml_path="$annotation_folder/${filename}.xml"
    echo $xml_path
    
    # Check if corresponding xml file exists
    if [ ! -f "$xml_path" ]; then
        echo -e "\nWarning: No matching XML file found for $filename"
        continue
    fi
    
    # Your processing code goes here
    cvat-cli --server-port 8080 --auth $credentials --organization $cvat_org \
        task create "${filename##*/}" share \
        resources $share_video_rel_path \
        --project_id $project_id \
        --annotation_path $xml_path \
        --use_cache \
        --annotation_format "CVAT 1.1"
done