#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_cvat_xml> <output_folder> <project_id> <credentials>"
    exit 1
fi

# Store command line arguments
input_cvat_xml="$1"
output_folder="$2"
project_id="$3"
credentials="$4"

# Check if folders exist
if [ ! -d "$input_cvat_xml" ]; then
    echo "Error: input_cvat_xml folder do not exist"
    exit 1
fi

tasks_json=$(cvat-cli --server-port 8080 --auth $credentials task ls --json)

echo "$tasks_json" | jq -c '.[]' | while read -r task_string; 
do
    task_id=$(echo $task_string | jq -r '.id')
    task_name=$(echo $task_string | jq -r '.name')
    task_name=${task_name%.*} #Remove potential file extensions (e.g., .mp4)
    task_status=$(echo $task_string | jq -r '.status')
    task_pid=$(echo $task_string | jq -r '.project_id')

    if [ "$task_status" = "validation" ] && [ "$task_pid" = "$project_id" ]; then
        input_xml_file=$(find "$input_cvat_xml" -type f -name "${task_name}.xml")
        output_zip_file="${output_folder}/$(realpath --relative-to="$input_cvat_xml" "${input_xml_file%.xml}").zip"
        output_dir=$(dirname "$output_zip_file")
        mkdir -p "$output_dir"

        cvat-cli --server-port 8080 --auth $credentials \
        task export-dataset $task_id $output_zip_file \
        --format "CVAT for video 1.1"

        unzip "$output_zip_file" -d "$output_dir"
        rm "$output_zip_file"
        mv "$output_dir/annotations.xml" "${output_zip_file%.zip}.xml"
    fi
done