# Wine Data Pipeline

This project implements a data pipeline for processing wine data and generating food pairing recommendations using AWS Bedrock and LangChain. The pipeline consists of two main components: an AI server data pipeline for initial data processing and a web app data pipeline for generating food pairing recommendations.

## Project Structure

```
data_pipelines/
├── ai_server_data_pipeline.py    # Initial data processing pipeline
├── web_app_data_pipeline.py      # Food pairing generation pipeline
├── run_pipelines.sh              # Script to run both pipelines sequentially
├── input/
│   └── vine_wine/
│       └── wine_data.csv        # Input wine data
├── processed_output/
│   └── processed_wine_data.csv   # Output from AI server pipeline
└── food_pairing_output/
    └── enriched_processed_wine_data.csv  # Final output with food pairings
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
- Loads input wine data from CSV
- Validates required columns
- Processes each wine entry
- Generates processed output with enhanced wine information

### 2. Web App Data Pipeline (`web_app_data_pipeline.py`)

This pipeline generates food pairing recommendations for each wine, including:
- Three distinct food pairings per wine
- Detailed pairing descriptions
- Food and wine acidity interactions
- Regional pairing information
- Sweetness and spiciness notes

### 3. Run Script (`run_pipelines.sh`)

A shell script that orchestrates the execution of both pipelines:
- Runs the AI server data pipeline with the input file from input/vine_wine directory
- Waits for the processed data file to be created
- Runs the web app data pipeline
- Verifies the output files

## Usage

1. Prepare your input data in CSV format with the following columns:
   - Product Name
   - Wine Type
   - Wine Grapes
   - Region
   - Country
   - Body
   - Acidity
   - Alcohol
   - Fruitiness
   - Minerality
   - Sweetness / Dry
   - Description

2. Place your input CSV file in the `input/vine_wine` directory.

3. Run the complete pipeline:
   ```bash
   chmod +x run_pipelines.sh
   ./run_pipelines.sh
   ```

## Output Files

1. `processed_output/processed_wine_data.csv`
   - Contains the processed wine data with enhanced information
   - Used as input for the food pairing pipeline

2. `food_pairing_output/enriched_processed_wine_data.csv`
   - Final output containing wine data with food pairing recommendations
   - Includes three distinct food pairings per wine with detailed descriptions

## Food Pairing Data Structure

The enriched output includes the following information for each wine:

### First Pairing
- Food Pairing 1
- Pairing Type 1
- Course 1
- Pairing Description 1
- Food & Wine Acidity 1
- Regional Pairing 1
- Sweetness & Spiciness 1
- Pairing Suitability 1

### Second Pairing
- Food Pairing 2
- Pairing Type 2
- Course 2
- Pairing Description 2
- Food & Wine Acidity 2
- Regional Pairing 2
- Sweetness & Spiciness 2
- Pairing Suitability 2

### Third Pairing
- Food Pairing 3
- Pairing Type 3
- Course 3
- Pairing Description 3
- Food & Wine Acidity 3
- Regional Pairing 3
- Sweetness & Spiciness 3
- Pairing Suitability 3

## Error Handling

The pipeline includes comprehensive error handling:
- Validates input data structure
- Checks for required columns
- Verifies file creation and content
- Implements retry logic for API calls
- Provides detailed error messages

## Notes

- The pipeline uses AWS Bedrock for generating food pairing recommendations
- Each wine gets three distinct food pairings with varying types and courses
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

Built by [kaulmesanyam](https://github.com/kaulmesanyam), with ❤️ 