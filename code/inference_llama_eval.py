import transformers
import torch
import numpy as np
import random
import os
import json
from prompt.prompt_llm_eval import gen_eval_prompt

from argparse import ArgumentParser
from tqdm import tqdm

# set seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

model_id = "/u/xfeng4/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"dtype": torch.bfloat16},
    device_map="auto",
)


def load_data(dataset_path):
    # Dataset and output file configuration
    with open(dataset_path, "r", encoding="utf-8") as fp:
        json_data = json.load(fp)
    print(f"Loaded {len(json_data)} samples from {dataset_path}")
    return json_data


def inference(args, json_data):
    final_result = []
    for x in tqdm(range(len(json_data))):
        cur_data = json_data[x]
        messages = gen_eval_prompt(cur_data, args.rubric_name)
        outputs = pipeline(
            messages,
            max_new_tokens=1024,
            pad_token_id=pipeline.tokenizer.eos_token_id
        )
        # print(cur_data['Ground_Truth_Solution'])
        # print("*"*100)
        # print(outputs[0]["generated_text"][-1])
        # print("*"*100)
        response = outputs[0]["generated_text"][-1]['content']
        cur_data[f'eval_{args.rubric_name}_prompt'] = messages
        cur_data[f'eval_{args.rubric_name}_result'] = response
        final_result.append(cur_data)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_file", type=str, default="../data/MRBench/MRBench_V1_Mistral-7B-Instruct-v0.1.json")
    args = parser.parse_args()
    input_file = args.dataset_file.split("/")[-1].replace(".json", "")
    args.output_file = os.path.join(os.path.dirname(args.dataset_file), input_file+"_llama_eval.json")
    print("*"*100)
    print(args)
    print("*"*100)
    MRBench_V1_data = load_data("../data/MRBench/MRBench_V1.json")
    conversation_history_map = {data['conversation_id']: data['conversation_history'] for data in MRBench_V1_data}
    args.dataset_file = args.output_file
    json_data = load_data(args.dataset_file)
    for data in json_data:
        data['conversation_history'] = conversation_history_map[data['conversation_id']]
    # rubric_name_list = ["mistake_identification", "mistake_location", "revealing_answer", "providing_guidance", "actionability", "tutor_tone", "humanness"]
    rubric_name_list = ["coherent"]
    for rubric_name in rubric_name_list:
        print(f"*"*100)
        print(f"Evaluating {rubric_name}...")
        print(f"*"*100)
        args.rubric_name = rubric_name
        
        inference(args, json_data)
        args.dataset_file = args.output_file
    print(f"Evaluation completed.")