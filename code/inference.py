import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
import helper

# Model configurations: model_id -> short_name mapping
model_ids = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama", 
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral"
}

# Dataset and output file configuration
dataset_file = "MRBench_V1.json"
output_file = "result_MRBench_V1.json"

# Set up directory paths
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, "..", "data")
data_dir = os.path.abspath(data_dir)

# Load prompt templates
MathDialPrompt = ""
BridgePrompt = ""

# Read Bridge prompt template
bridge_prompt_path = os.path.join(data_dir, "prompt", "prompt_Bridge.txt")
with open(bridge_prompt_path, "r", encoding="utf-8") as f:
    BridgePrompt = f.read()

# Read MathDial prompt template  
mathdial_prompt_path = os.path.join(data_dir, "prompt", "prompt_MathDial.txt")
with open(mathdial_prompt_path, "r", encoding="utf-8") as f:
    MathDialPrompt = f.read()

# Load dataset
dataset_path = os.path.join(data_dir, "MRBench", dataset_file)
with open(dataset_path, "r", encoding="utf-8") as fp:
    json_data = json.load(fp)

print(f"Loaded {len(json_data)} samples from {dataset_path}")
print(f"Bridge prompt template loaded from: {bridge_prompt_path}")
print(f"MathDial prompt template loaded from: {mathdial_prompt_path}")

def _take_text(generation_result):
    """
    Extract generated text from model output
    
    Args:
        generation_result: Model generation output (can be list or dict)
    
    Returns:
        str: Generated text content
    """
    if isinstance(generation_result, list):
        return generation_result[0]['generated_text']
    return generation_result['generated_text']

# Main inference loop for each model
for model_id, model_name in model_ids.items():
    print(f"\n=== Processing model: {model_id} ===")
    
    final_result = []
    org = True  # Original formatting flag (kept for compatibility)
    
    # Load tokenizer and set padding configuration
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id) 
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # Load model with optimized settings
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use torch_dtype instead of dtype
        device_map="auto",
        trust_remote_code=True  # Add for some models that require it
    ).eval()
    
    # Create text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    
    # Configure generation parameters
    gen_cfg = GenerationConfig(
        do_sample=False,  # Use greedy decoding for consistency
        max_new_tokens=200,  # Maximum number of new tokens to generate
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.1,  # Reduce repetition
        temperature=0.1  # Low temperature for more focused responses
    )
    
    # Prepare prompts for all data samples
    print("Preparing prompts...")
    idx_list, prompt_list = [], []
    for x in range(len(json_data)):
        cur_data = json_data[x]
        # Select appropriate prompt based on data source
        if cur_data['Data'] == "MathDial":
            prompt = helper.MathDial_Prompt(MathDialPrompt, cur_data, org)
        else:
            prompt = helper.Bridge_Prompt(BridgePrompt, cur_data, org)
        
        idx_list.append(x)
        prompt_list.append(prompt)
    
    # Process in batches for memory efficiency
    batch_size = 16  # Reduced batch size for better memory management
    print(f"Processing {len(prompt_list)} samples in batches of {batch_size}...")
    
    for start in tqdm(range(0, len(prompt_list), batch_size), desc=f"Generating with {model_name}", unit="batch"):
        end = min(start + batch_size, len(prompt_list))
        batch_prompts = prompt_list[start:end]
        batch_idxs = idx_list[start:end]

        # Generate responses for current batch
        generations = generator(
            batch_prompts,
            batch_size=len(batch_prompts),   
            generation_config=gen_cfg,
            return_full_text=False,  # Only return generated part
            truncation=True,
            max_length=1024  # Total max length including input
        )

        # Process each generation result
        for x, g in zip(batch_idxs, generations):
            cur_data = json_data[x]
            temp = {}
            
            # Clean and format the generated response
            result = helper.safe_cut_at_first_heading(_take_text(g))
            
            # Store results with original data fields
            temp["result"] = result
            temp["Data"] = cur_data["Data"]
            temp["conversation_history"] = cur_data["conversation_history"]
            temp["Topic"] = cur_data["Topic"]
            temp["Ground_Truth_Solution"] = cur_data["Ground_Truth_Solution"]
            temp["conversation_id"] = cur_data.get("conversation_id", f"sample_{x}")
            
            final_result.append(temp)
            
            # Progress indicator
            if len(final_result) % 100 == 0:
                print(f"Processed {len(final_result)} samples...")
    
    # Save results to file
    output_path = os.path.join(data_dir, f"{model_name}_{output_file}")
    print(f"Saving {len(final_result)} results to {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    print(f"Completed processing {model_id}. Results saved to {output_path}")
    
    # Clean up GPU memory
    del model
    del generator
    torch.cuda.empty_cache()

print("\n=== All models processed successfully! ===")