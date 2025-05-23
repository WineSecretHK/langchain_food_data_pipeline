# Food Data Pipeline

This project implements a data pipeline for processing food data and generating wine pairing recommendations using AWS Bedrock and LangChain. The pipeline consists of two main components: an AI server data pipeline for initial data processing and a web app data pipeline for generating wine pairing recommendations.

## Project Structure

```
data_pipelines/
├── food_ai_server_data_pipeline.py    # Initial data processing pipeline
├── food_web_app_data_pipeline.py      # Wine pairing generation pipeline
├── food_run_pipelines.sh              # Script to run both pipelines sequentially
├── input/
│   └── wshk/
│       └── food_url_list.csv        # Input food name and url
├── processed_output/
│   └── processed_food_data.csv   # Output from AI server pipeline
└── wine_pairing_output/
    └── enriched_processed_food_data.csv  # Final output with wine pairings
```

## Prerequisites

- Python 3.8 or higher
- AWS account with Bedrock access
- Required Python packages (install using `pip install -r requirements.txt`):
  - langchain
  - langchain-aws
  - pandas
  - python-dotenv
  - boto3

## Environment Setup

1. Create a `.env` file in the project root with the following variables:
   ```
   AWS_REGION=your_aws_region
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Pipeline Components

### 1. AI Server Data Pipeline (`ai_server_data_pipeline.py`)

This pipeline processes the initial wine data, performing the following tasks:
- Loads input food data from CSV
- Validates required columns (only food/dish name)
- Processes each food entry
- Generates processed output with enhanced food information

### 2. Web App Data Pipeline (`web_app_data_pipeline.py`)

This pipeline generates wine pairing recommendations for each food, including:
- Three distinct wine pairings per wine
- Detailed pairing descriptions
- Food and wine acidity interactions
- Sweetness and spiciness notes

### 3. Run Script (`food_run_pipelines.sh`)

A shell script that orchestrates the execution of both pipelines:
- Runs the AI server data pipeline with the input file from input/wshk directory
- Waits for the processed data file to be created
- Runs the web app data pipeline
- Verifies the output files

## Usage

1. Prepare your input data in CSV format with the following columns:
   - Product Name
   - Image URL

     * Note: there is another repository for mapping food name to food image url through webscraping
     * 
2. Place your input CSV file in the `input` directory.

3. Run the complete pipeline:
   ```bash
   chmod +x food_run_pipelines.sh
   ./food_run_pipelines.sh
   ```

## Output Files

1. `processed_output/processed_food_data.csv`
   - Contains the processed food data with enhanced information
   - Used as input for the wine pairing pipeline

2. `wine_pairing_output/enriched_processed_food_data.csv`
   - Final output containing wine data with wine pairing recommendations
   - Includes three distinct wine pairings per food with detailed descriptions

## Wine Pairing Data Structure

The enriched output includes the following information for each wine:
    - Tasting_Notes: Suggested description on the basis of tasting notes
    - Suitability: Rating of suggestion on a scale of 1 to 10
    - Grape_Food_Type: Explanation of how the wine’s grape variety matches the food type
    - Sweetness_Spiciness: Commentary on balance between food's sweetness/spiciness and the wine
    - Minerality_Freshness: Notes on how the wine’s freshness/minerality interacts with the dish

## Error Handling

The pipeline includes comprehensive error handling:
- Validates input data structure
- Checks for required columns
- Verifies file creation and content
- Implements retry logic for API calls
- Provides detailed error messages

## Notes

- The pipeline uses AWS Bedrock for generating food pairing recommendations
- Each food gets three distinct food pairings with varying types and courses
- The script includes delays between API calls to avoid rate limiting
- Output files are created in separate directories for better organization

## Troubleshooting

If you encounter issues:
1. Check your AWS credentials in the `.env` file
2. Verify that the input CSV file has all required columns
3. Ensure you have the necessary AWS permissions for Bedrock
4. Check the error messages in the console output
5. Verify that the output directories exist and are writable

## Contributors

Built by [kaulmesanyam](https://github.com/kaulmesanyam) and [kin-mrqz](https://github.com/kin-mrqz), with ❤️ 
