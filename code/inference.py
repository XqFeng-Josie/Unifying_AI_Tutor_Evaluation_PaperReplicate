import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
import helper
from argparse import ArgumentParser

# Model configurations: model_id -> short_name mapping
model_ids_map = {
    # "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "llame": "/u/xfeng4/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
}
def load_data(args):
    # Dataset and output file configuration
    dataset_path = os.path.join(args.dataset_file)
    with open(dataset_path, "r", encoding="utf-8") as fp:
        json_data = json.load(fp)
    print(f"Loaded {len(json_data)} samples from {args.dataset_file}")
    return json_data

def load_prompt(args):
    with open(args.bridge_prompt_path, "r", encoding="utf-8") as f:
        BridgePrompt = f.read()
    with open(args.mathdial_prompt_path, "r", encoding="utf-8") as f:
        MathDialPrompt = f.read()
    print(f"Bridge prompt template loaded from: {args.bridge_prompt_path}")
    print(f"MathDial prompt template loaded from: {args.mathdial_prompt_path}")
    return BridgePrompt, MathDialPrompt

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

def inference(args, json_data, BridgePrompt, MathDialPrompt):  
    print(f"\n=== Processing model: {model_id} ===")
    model_path = model_ids_map[args.model_name]
    
    final_result = []    # Load tokenizer and set padding configuration
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path) 
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # Load model with optimized settings
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,  # Use torch_dtype instead of dtype
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
        max_new_tokens=args.max_new_tokens,  # Maximum number of new tokens to generate
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=args.repetition_penalty,  # Reduce repetition
        temperature=args.temperature  # Low temperature for more focused responses
    )
    
    # Prepare prompts for all data samples
    print("Preparing prompts...")
    idx_list, prompt_list = [], []
    for x in range(len(json_data)):
        cur_data = json_data[x]
        # Select appropriate prompt based on data source
        if cur_data['Data'] == "MathDial":
            prompt = helper.MathDial_Prompt(MathDialPrompt, cur_data)
        else:
            prompt = helper.Bridge_Prompt(BridgePrompt, cur_data)
        
        idx_list.append(x)
        prompt_list.append(prompt)
    
    # Process in batches for memory efficiency
    batch_size = args.batch_size  # Reduced batch size for better memory management
    print(f"Processing {len(prompt_list)} samples in batches of {batch_size}...")
    
    for start in tqdm(range(0, len(prompt_list), batch_size), desc=f"Generating with {args.model_name}", unit="batch"):
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
            max_length=args.max_length  # Total max length including input
        )

        # Process each generation result
        for x, g in zip(batch_idxs, generations):
            cur_data = json_data[x]
            temp = {}
            original_response = _take_text(g)
            # Clean and format the generated response
            result = helper.safe_cut_at_first_heading(original_response)
            # Store results with original data fields
            temp['original_response'] = original_response
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
    print(f"Saving {len(final_result)} results to {args.output_file}")
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    print(f"Completed processing {args.model_name}. Results saved to {args.output_file}")
    
    # Clean up GPU memory
    del model
    del generator
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--org", type=bool, default=True)
    parser.add_argument("--dataset_file", type=str, default="../data/MRBench/MRBench_V1.json")
    parser.add_argument("--bridge_prompt_path", type=str, default="../data/prompt/prompt_Bridge.txt")
    parser.add_argument("--mathdial_prompt_path", type=str, default="../data/prompt/prompt_MathDial.txt")

    args = parser.parse_args()
    args.output_file = os.path.join(os.path.dirname(args.dataset_file), args.model_name + "_result.json")
    print("*"*100)
    print(args)
    print("*"*100)
    json_data = load_data(args)
    BridgePrompt, MathDialPrompt = load_prompt(args)
    inference(args, json_data, BridgePrompt, MathDialPrompt)