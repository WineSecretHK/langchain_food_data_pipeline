#!/bin/bash

# Function to check if a file exists and is not empty
check_file() {
    if [ -f "$1" ] && [ -s "$1" ]; then
        return 0
    else
        return 1
    fi
}

# Function to wait for file creation
wait_for_file() {
    local file="$1"
    local max_attempts=60  # 5 minutes with 5-second intervals
    local attempts=0

    echo "Waiting for $file to be created..."
    while [ $attempts -lt $max_attempts ]; do
        if check_file "$file"; then
            echo "$file has been created successfully."
            return 0
        fi
        echo "Waiting... ($((attempts + 1))/$max_attempts)"
        sleep 5
        attempts=$((attempts + 1))
    done

    echo "Timeout waiting for $file to be created."
    return 1
}

# Step 1: Run the AI server data pipeline with the input file
echo "Starting AI server data pipeline..."
python3 food_ai_server_data_pipeline.py "input/wshk/food_url_list.csv"

# Check if the script executed successfully
if [ $? -ne 0 ]; then
    echo "Error: AI server data pipeline failed to execute."
    exit 1
fi

# Step 2: Wait for processed_wine_data.csv to be created
processed_file="processed_output/processed_food_data.csv"
if ! wait_for_file "$processed_file"; then
    echo "Error: Timeout waiting for processed_food_data.csv"
    exit 1
fi

# Step 3: Run the web app data pipeline
echo "Starting web app data pipeline..."
python3 food_web_app_data_pipeline.py "$processed_file"

# Check if the script executed successfully
if [ $? -ne 0 ]; then
    echo "Error: Web app data pipeline failed to execute."
    exit 1
fi

# Step 4: Verify the enriched data file was created
enriched_file="food_pairing_output/enriched_processed_food_data.csv"
if ! check_file "$enriched_file"; then
    echo "Error: Enriched wine data file was not created."
    exit 1
fi

echo "All pipelines completed successfully!"
echo "Processed data: $processed_file"
echo "Enriched data: $enriched_file" 