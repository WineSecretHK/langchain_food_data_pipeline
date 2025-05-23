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
    "top_p": 0.9,
}

# Define the column names for the food pairing data
WINE_PAIRING_COLUMNS = [
    # First Pairing
    "Wine Pairing 1", "Tasting Notes 1", "Suitability 1", "Grape and Food Type 1",
    "Sweetness & Spiciness 1", "Minerality and Freshness 1",
    # Second Pairing
    "Wine Pairing 2", "Tasting Notes 2", "Suitability 2", "Grape and Food Type 2",
    "Sweetness & Spiciness 2", "Minerality and Freshness 2",
    # Third Pairing
    "Wine Pairing 3", "Tasting Notes 3", "Suitability 3", "Grape and Food Type 3",
    "Sweetness & Spiciness 3", "Minerality and Freshness 3",
]

# Default values for food pairing columns
DEFAULT_WINE_PAIRING_VALUES = {
    "Wine Pairing 1": "", 
    "Tasting Notes 1": "",             # Tasting Notes => Pairing Description in prompt template
    "Suitability 1": "",               # Suitability => Pairing Suitability in prompt template
    "Grape and Food Type 1": "",
    "Sweetness & Spiciness 1": "", 
    "Minerality and Freshness 1": "",

    "Wine Pairing 2": "", 
    "Tasting Notes 2": "", 
    "Suitability 2": "", 
    "Grape and Food Type 2": "",
    "Sweetness & Spiciness 2": "", 
    "Minerality and Freshness 2": "",

    "Wine Pairing 3": "", 
    "Tasting Notes 3": "", 
    "Suitability 3": "", 
    "Grape and Food Type 3": "",
    "Sweetness & Spiciness 3": "", 
    "Minerality and Freshness 3": "",
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

def create_food_pairing_chains(llm):
    """Create chains for generating food pairing recommendations and details."""
    
    # Chain 1: Generate three distinct food pairings with type and course
    wine_pairings_prompt = PromptTemplate.from_template(
        """
        You are a food expert with expertise in food and wine pairings.
        
        Food information:
        - Name: {product_name}
        - Type: {food_type}
        - Characteristics:
          - Sweetness: {sweetness_dry}
          - Saltiness: {saltiness}
          - Sourness: {sourness}
          - Bitterness: {bitterness}
          - Umami: {umami}
        
        Provide three DISTINCT recommended wine pairings for this food in key-value format as follows:
        
        Wine_Pairing_1: [Suggested wine]

        Wine_Pairing_2: [Suggested wine - MUST be different from first]

        Wine_Pairing_3: [Suggested wine - MUST be different from first and second]

        IMPORTANT GUIDELINES:
        1. Each wine pairing must be unique and different from the others
        2. Consider the food's specific characteristics:
           - For rich, fatty foods: pair high acidity wines
           - For hearty, robust dishes: pair with full-bodies wines
           - For delicate, subtle dishes: pair with light-bodied wines
           - For spicy or salty foods: pair with sweet wines
           - For rich or fatty foods: pair with dry wines
        3. Consider the wine's origin:
           - Include at least one traditional pairing from the food's region
           - Include at least one modern or fusion pairing
        4. Vary the courses:
           - Don't repeat the same course type
           - Include a mix of appetizers, main courses, sides, and desserts
        5. Vary the food types:
           - Don't repeat the same wine type (e.g., if first is red wine, others should be different)
        6. Consider the food's unique characteristics:
           - If the food is salty or rich in umami, consider pairing with high minerality wines
           - if the food has fruity notes, consider pairing with wines that complements or contrasts the fruitiness
           - if the food is fatty, consider pairing with high alcohol
        7. Specific constraints for diversity:
           - For specific course types (e.g., main course, appetizer, dessert), suggest a mix of wine types
           - For vegetarian options, include at least one white wine option
           - for meat-based dish, include at least one red wine options
           - For dessert courses, include sweet wines
        8. Regional considerations:
           - For traditional French cuisine: include French wines
           - for traditional Italian dishes: include Italian wines
           - for traditional and fusion dishes: include New World wines
        
        Respond with only the key-value pairs above.
        """
    )
    
    wine_pairings_chain = wine_pairings_prompt | llm | StrOutputParser()
    
    # Chain 2: Generate description for a single pairing
    pairing_description_prompt = PromptTemplate.from_template(
        """
        You are a food expert specializing in explaining food and wine pairings.

        Food information:
        - Name: {product_name}
        - Type: {food_type}
        - Characteristics:
          - Sweetness: {sweetness_dry}
          - Saltiness: {saltiness}
          - Sourness: {sourness}
          - Bitterness: {bitterness}
          - Umami: {umami}
        
        Wine pairing to describe: 
        {wine_pairing}
        
        Provide a BRIEF 1-2 line description explaining why this wine pairs well with the food.
        Focus on how the wine's flavor complement the food's characteristics.
        Consider:
        - How the wine's acidity interacts with the food
        - How the wine's body matches the food's weight
        - How the wine's flavor profile complements the dish
        - Any regional or traditional pairing connections
        
        Format your response as key-value pairs:
        
        Pairing_Description: [Brief explanation]
        Pairing_Suitability: [Rating from 1-10]
        
        Respond with only the key-value pairs above.
        """
    )
    
    pairing_description_chain = pairing_description_prompt | llm | StrOutputParser()
    
    # Chain 3: Generate notes for a single pairing
    pairing_notes_prompt = PromptTemplate.from_template(
        """
        You are a wine and food science expert.
        
        Food information:
        - Name: {product_name}
        - Type: {food_type}
        - Characteristics:
          - Sweetness: {sweetness_dry}
          - Saltiness: {saltiness}
          - Sourness: {sourness}
          - Bitterness: {bitterness}
          - Umami: {umami}
        
        Food pairing: {wine_pairing}
        
        Provide BRIEF 1-2 line notes on:
        1. How the wine’s grape variety matches the food type
        2. The balance between food's sweetness/spiciness and the wine
        3. How the wine’s freshness/minerality interacts with the dish
        
        Consider:
        - The specific characteristics of the wine and how they interact with the food
        - Modern interpretations of classic pairings
        - How the pairing enhances both the wine and food
        
        Format your response as key-value pairs:

        Grape_Food_Type: [Explanation of specific grape variety and food type]
        Sweetness_Spiciness: [Commentary on balance]
        Minerality_Freshness: [Note on freshness and minerality]

        
        Respond with only the key-value pairs above.
        """
    )
    
    pairing_notes_chain = pairing_notes_prompt | llm | StrOutputParser()
    
    def process_single_pairing(food_data, wine_pairing, pairing_number):
        """Process description and notes for a single wine pairing."""
        try:
            # Get pairing description
            pairing_data = {**food_data, "wine_pairing": wine_pairing}
            pairing_description = pairing_description_chain.invoke(pairing_data)
            pairing_description_dict = parse_keyval_response(pairing_description)
            
            # Get pairing notes
            pairing_notes = pairing_notes_chain.invoke(pairing_data)
            pairing_notes_dict = parse_keyval_response(pairing_notes)
            
            # Add pairing number to keys
            result = {}
            for key, value in pairing_description_dict.items():
                result[f"{key}_{pairing_number}"] = value
            for key, value in pairing_notes_dict.items():
                result[f"{key}_{pairing_number}"] = value
                
            return result
            
        except Exception as e:
            print(f"Error processing pairing {pairing_number}: {e}")
            return None
    
    def process_food_info(food_data):
        """Process all three pairings for a wine."""
        try:
            # First, get all three wine pairings together
            wine_pairings = wine_pairings_chain.invoke(food_data)
            wine_pairings_dict = parse_keyval_response(wine_pairings)
            
            # Verify we have three distinct pairings
            pairings = [ 
                wine_pairings_dict.get(f"Wine_Pairing_{i}", "")
                for i in range(1, 4)
            ]
            
            # Check for duplicates
            if len(set(pairings)) < 3:
                raise ValueError("Generated wine pairings are not distinct")
            
            result = wine_pairings_dict
            
            # Process descriptions and notes for each pairing
            for i in range(1, 4):
                wine_pairing = f"Wine_Pairing_{i}: {wine_pairings_dict.get(f'Wine_Pairing_{i}', '')}\n" 
                
                pairing_result = process_single_pairing(food_data, wine_pairing, i)
                if pairing_result:
                    result.update(pairing_result)
                    # Add delay between pairings
                    time.sleep(1.0)
                else:
                    raise ValueError(f"Failed to process pairing {i}")
            
            return result
            
        except Exception as e:
            print(f"Error processing wine info: {e}")
            raise
    
    return process_food_info

def extract_food_data(row):
    """Extract food data from a CSV row for pairing generation."""
    return {
        "product_name": row.get("Product Name", "Unknown Food"),
        "food_type": row.get("Food Type", "Unknown Food Type"),
        "course": row.get("Course", "Unknown Food Course"),
        "sweetness": row.get("Sweetness", "Unknown"),
        "saltiness": row.get("Saltiness", "Unknown"),
        "sourness": row.get("Sourness", "Unknown"),
        "bitterness": row.get("Bitterness", "Unknown"),
        "umami": row.get("Umami", "Unknown"),
        "tags": row.get("Tags", "Unknown"),
    }

def parse_keyval_response(response_text):
    """Parse key-value pair response from LLM."""
    result = {}
    
    # Clean response text
    response_text = response_text.strip()
    
    # Remove any code block markers if present
    if response_text.startswith("```") and response_text.endswith("```"):
        response_text = response_text[3:-3].strip()
    elif response_text.startswith("*/") and response_text.endswith("*/"):
        response_text = response_text[2:-2].strip()
    
    # Debug print
    print("Raw response text:")
    print(response_text)
    print("---")
    
    # Split by lines and process each line
    lines = response_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line or ':' not in line:
            continue
            
        # Split by first colon
        parts = line.split(':', 1)
        if len(parts) < 2:
            continue
            
        key = parts[0].strip().replace(' ', '_')
        value = parts[1].strip()
        
        # Debug print
        print(f"Parsed key-value: {key} -> {value}")
        
        result[key] = value
    
    return result

def clean_text(text):
    """Clean text by removing special characters and normalizing whitespace."""
    if not isinstance(text, str):
        return text
        
    # Remove special characters but keep basic punctuation and spaces
    cleaned = ''.join(char for char in text if char.isalnum() or char in ' .,;:!?()-/')
    
    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

def process_food_csv(input_file, output_file):
    """Process a food CSV file and generate wine pairing data."""
    print(f"Processing {input_file} for wine pairing data...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False
    
    # Set up LLM and create processing chains
    llm = setup_llm()
    food_pairing_pipeline = create_food_pairing_chains(llm)
    
    # Create a new dataframe to store all data
    result_data = []
    
    # Process each row
    for index, row in df.iterrows():
        print(f"Processing row {index+1}/{len(df)}: {row.get('Product Name', 'Unknown')}")
        
        # Extract wine data for pairing generation
        wine_data = extract_food_data(row)
        
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                # Run the food pairing pipeline
                result = food_pairing_pipeline(wine_data)
                
                # Debug print
                print("Raw pipeline result:")
                print(result)
                print("---")
                
                # Create a new row with all the original data and the new pairing data
                new_row = {col: clean_text(row.get(col, "")) for col in df.columns}
                
                # Add wine pairing data for each of the three pairings
                for i in range(1, 4):
                    new_row[f"Wine Pairing{i}"] = clean_text(result.get(f"Wine_Pairing_{i}", ""))
                    new_row[f"Tasting Notes {i}"] = clean_text(result.get(f"Pairing_Description_{i}", ""))
                    new_row[f"Suitability {i}"] = clean_text(result.get(f"Pairing_Suitability_{i}", ""))
                    new_row[f"Grape and Food Type {i}"] = clean_text(result.get(f"Grape_Food_Type_{i}", ""))
                    new_row[f"Sweetness and Spiciness {i}"] = clean_text(result.get(f"Sweetness_Spiciness_{i}", ""))
                    new_row[f"Minerality and Freshness {i}"] = clean_text(result.get(f"Minerality_Freshness_{i}", ""))
                
                result_data.append(new_row)
                success = True
                
                # Add a longer delay between wines
                time.sleep(2.0)
                
            except Exception as e:
                retry_count += 1
                print(f"Error processing row {index} (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    print("Retrying after a longer delay...")
                    time.sleep(5.0)  # Longer delay between retries
                else:
                    print("Max retries reached. Using default values for this row.")
                    # Add the original row with empty pairing data
                    new_row = {col: clean_text(row.get(col, "")) for col in df.columns}
                    
                    # Add empty values for all food pairing columns
                    for col in WINE_PAIRING_COLUMNS:
                        new_row[col] = clean_text(DEFAULT_WINE_PAIRING_VALUES.get(col, ""))
                        
                    result_data.append(new_row)
    
    # Create final dataframe
    final_df = pd.DataFrame(result_data)
    
    # Clean all text columns in the final dataframe
    for column in final_df.columns:
        if final_df[column].dtype == 'object':
            final_df[column] = final_df[column].apply(clean_text)
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"Wine pairing processing complete! Output saved to {output_file}")
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = os.path.join("wine_pairing_output", f"enriched_{os.path.basename(input_file)}")
    else:
        # Default for testing
        input_file = os.path.join("processed_output", "processed_food_data.csv")
        output_file = os.path.join("wine_pairing_output", "enriched_food_data.csv")
    
    process_food_csv(input_file, output_file)
