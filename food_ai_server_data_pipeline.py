import os
import time
import boto3
import pandas as pd
from langchain_aws import Bedrock
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
MODEL_ID = "meta.llama3-70b-instruct-v1:0"
MODEL_PARAMS = {
    "temperature": 0.2,
    "top_p": 0.9
}


REQUIRED_COLUMNS = [
    "Food Name",
    # "Image Link"
    ]

# Note: the lines for generating food descriptions have been commented

DEFAULT_VALUES = {
    "Image Link": "",
    "Price": "0.00",
    "Currency": "HKD",
    "Sizing": "Medium",
    "Food Type": "",
    "Course": "",
    "Sweetness": "5",
    "Saltiness": "5",
    "Sourness": "5",
    "Bitterness": "5",
    "Umami": "5",
    "Tags": "",
    # "Description": ""
}

def setup_llm():
    """Set up and return the AWS Bedrock LLM."""
    # Set up AWS Bedrock client
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION
    )

    # Initialize Llama-3-70B-instruct model
    llm = Bedrock(
        client=bedrock_runtime,
        model_id=MODEL_ID,
        model_kwargs=MODEL_PARAMS
    )
    
    return llm

def create_processing_chain(llm):
    """Create the processing chain using Runnable components."""
    output_parser = StrOutputParser()
    
    # Base information prompt
    base_info_prompt = PromptTemplate.from_template(
        """
        You are a food expert with deep knowledge of food globally.
        
        Based only on the food product name: {product_name}, 
        generate the following information as key-value pairs, one per line, in the format:
        
        Price: Listed or average price of food item
        Currency: Preferrably in HKD
        Sizing: (Small, Medium, Large)
        Food type: Main ingredient or category (e.g., Pork, Chicken, Seafood, Vegan)
        Course: (Appetizer, Main Course, Sharing, Desert)

        Return only the key-value pairs, nothing else.
        """
    )
    
    base_info_chain = base_info_prompt | llm | output_parser
    
    # Characteristics prompt
    characteristics_prompt = PromptTemplate.from_template(
        """
        You are a food expert with deep knowledge of culinary arts, gastronomy, and food characteristics.

        Based on the food name: {product_name},
        generate the following characteristics as key-value pairs, one per line, in the format:
        
        Sweetness: rating - 1 to 10
        Saltiness: rating - 1 to 10
        Sourness: rating - 1 to 10
        Bitterness: rating - 1 to 10
        Umami: rating - 1 to 10
        Tags: Comma-separated keywords for filtering or searching

        Return only the key-value pairs, nothing else.
        """
    )
    
    characteristics_chain = characteristics_prompt | llm | output_parser
    
    # # Description prompt
    # description_prompt = PromptTemplate.from_template(
    #     """
    #     You are a food expert.

    #     Based on the wine product name: {product_name}, grape variety: {grape}, region: {region}, and characteristics: {characteristics_json},
    #     write a short description of the food (e.g., "Fruity, with hints of oak and a long, smooth finish.").
    #     Return only the description, nothing else.
    #     """
    # )
    
    # description_chain = description_prompt | llm | output_parser
    
    # Define the sequential processing pipeline using RunnableSequence
    def full_pipeline(inputs):
        # First get base info
        base_info_json = base_info_chain.invoke({"product_name": inputs["product_name"]})
        base_info = parse_key_value_pairs(base_info_json)
        
        # Fill in missing inputs with base info (Assuming only food name is given)
        price = inputs.get("Price", "Unknown")
        if price == "Unknown":
            price = base_info.get("Price", "Unknown")

        currency = inputs.get("Currency", "Unknown")
        if currency == "Unknown":
            currency = base_info.get("Currency", "Unknown")

        food_type = inputs.get("Food Type", "Unknown")
        if food_type == "Unknown":
            food_type = base_info.get("Food Type", "Unknown")
        
        course = inputs.get("Course", "Unknown")
        if course == "Unknown":
            course = base_info.get("Course", "Unknown")
        
        # Get characteristics
        characteristics_json = characteristics_chain.invoke({
            "product_name": inputs["product_name"],
        })

        # description = description_chain.invoke({
        #     "product_name": inputs["product_name"],
        #     "food type": food_type,
        #     "course": course,
        #     "characteristics_json": characteristics_json
        # })
        
        # Return all results in a dictionary
        return {
            "base_info_json": base_info_json,
            "characteristics_json": characteristics_json,
            # "description": description
        }
    
    return full_pipeline

def extract_input_data(row):
    """Extract input data from a CSV row."""
    return {
        "product_name": row.get("Product Name", "Unknown Wine"),
        # "price": str(row.get("Retail Price", "0.00")), # price excluded from required column
    }

def parse_key_value_pairs(text):
    """Parse key-value pairs from LLM output."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
            
        # Handle other key-value pairs
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()
    
    return result

def process_csv(input_file, output_file):
    """Process a CSV file and generate the enhanced output."""
    print(f"Processing {input_file}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Verify required minimum columns
        if "Product Name" not in df.columns:
            print("Error: Required column 'Product Name' not found in CSV")
            return False
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False
    
    # Set up LLM and create processing chain
    llm = setup_llm()
    processing_chain = create_processing_chain(llm)
    
    # Process each row
    results = []
    for index, row in df.iterrows():
        print(f"Processing row {index+1}/{len(df)}: {row.get('Product Name', 'Unknown')}")
        
        # Extract input data
        inputs = extract_input_data(row)
        
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                # Run the pipeline
                result = processing_chain(inputs)
                
                # Verify that we have all required data
                if not all(key in result for key in ["base_info_json", "characteristics_json"]):
                    # "description" omitted
                    raise ValueError("Missing required data in LLM response")
                
                # Parse the generated information
                characteristics = parse_key_value_pairs(result["characteristics_json"])
                base_info = parse_key_value_pairs(result["base_info_json"])
                
                # Create a new row with all required fields
                food_data = {
                    # Guaranteed columns or generated values
                    "Product Name": row.get("Product Name", ""),
                    "Image Link": row.get("Image Link", DEFAULT_VALUES["Image Link"]),
                    
                    # Use CSV value if present, otherwise use generated or default
                    "Price": row.get("Price", base_info.get("Price", DEFAULT_VALUES["Price"])),
                    "Currency": row.get("Currency", base_info.get("Currency", DEFAULT_VALUES["Currency"])),
                    "Sizing": row.get("Sizing", base_info.get("Sizing", DEFAULT_VALUES["Sizing"])),
                    "Food Type": row.get("Food Type", base_info.get("Food Type", DEFAULT_VALUES["Food Type"])),
                    "Course": row.get("Course", base_info.get("Course", DEFAULT_VALUES["Course"])),
                    
                    # Characteristics
                    "Sweetness": characteristics.get("Sweetness", DEFAULT_VALUES["Sweetness"]),
                    "Saltiness": characteristics.get("Saltiness", DEFAULT_VALUES["Saltiness"]),
                    "Sourness": characteristics.get("Sourness", DEFAULT_VALUES["Sourness"]),
                    "Bitterness": characteristics.get("Bitterness", DEFAULT_VALUES["Bitterness"]),
                    "Umami": characteristics.get("Umami", DEFAULT_VALUES["Umami"]),
                    "Tags": characteristics.get("Tags", DEFAULT_VALUES["Tags"])
                }
                results.append(food_data)
                success = True
                
                # Add a longer delay to avoid rate limiting and ensure complete responses
                time.sleep(0.5)  # Increased from 0.5 to 2.0 seconds
                
            except Exception as e:
                retry_count += 1
                print(f"Error processing row {index} (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    print("Retrying after a longer delay...")
                    time.sleep(5.0)  # Longer delay between retries
                else:
                    print("Max retries reached. Using default values for this row.")
                    # Add row with default values for failed processing
                    basic_data = {col: row.get(col, "") for col in REQUIRED_COLUMNS if col in row}
                    for col in REQUIRED_COLUMNS:
                        if col not in basic_data:
                            basic_data[col] = DEFAULT_VALUES.get(col, "")
                    results.append(basic_data)
    
    # Create final dataframe
    final_df = pd.DataFrame(results)
    
    # Validate that all required columns are present
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in final_df.columns]
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        # Add missing columns with default values
        for col in missing_columns:
            final_df[col] = DEFAULT_VALUES.get(col, "")
    else:
        print("All required columns present!")
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"CSV processing complete! Output saved to {output_file}")
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = os.path.join("processed_output", f"processed_{os.path.basename(input_file)}")
    else:
        # Default for testing
        input_file = os.path.join("sample_input", "food_data.csv")
        output_file = os.path.join("processed_output", "processed_food_data.csv")
    
    process_csv(input_file, output_file)