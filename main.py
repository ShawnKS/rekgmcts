# Main program entry
import argparse
from model import HUGGINGFACE_LLM
from mcts import MCTSNode, MCTSPathFinder
from sparql_utils import *
import os
import json
import pickle
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig


DATA_NAMES = ["cwq", "WebQSP"]

import argparse
# Define the mapping between models and data
MODEL_PATHS = {
    "GPT4": "api/v1..",
    "Llama-3.1-70B": "/path/to/Llama-3.1-70B",
    "Qwen2.5-72B": "/path/to/Qwen/Qwen2.5-72B-Instruct",
    "Llama-3.1-8B": "/path/to/Llama-3.1-8B",
    "Mistral-7B": "/path/to/Mistral-7B-Instruct-v0.3",
    "Qwen2.5-7B": "/path/to/Qwen2.5-7B-Instruct",
    "deepseek-llm-7b": "/path/to/deepseek-llm-7b-chat",
}

def parse_args():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Model and data selection strategy")

    # Add model selection parameter
    parser.add_argument(
        "--model",
        type=str,
        choices=MODEL_PATHS.keys(),
        default="deepseek-llm-7b",
        help="Select model (default: deepseek-llm-7b)",
    )

    # Add data selection parameter
    parser.add_argument(
        "--data",
        type=str,
        choices=DATA_NAMES,
        default="cwq",
        help="Select data (default: cwq)",
    )

    parser.add_argument(
        "--exploration_constant",
        type=float,
        default=1,
        help="UCT exploration constant (default: 1)"
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="maximum depth"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=5,
        help="maximum depth"
    )

    parser.add_argument(
        "--iterations",
        type=float,
        default=5,
        help="maximum iterations"
    )
    # Parse arguments
    args = parser.parse_args()

    # Set local_model_path based on model selection
    args.local_model_path = MODEL_PATHS[args.model]

    return args

args = parse_args()
def main():
    # Parse command line arguments
    args = parse_args()
    model_name = args.model
    local_model_path = args.local_model_path

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    llm = HUGGINGFACE_LLM(local_model_path=local_model_path, max_new_tokens=120)
    llm.tokenizer.pad_token = llm.tokenizer.eos_token
    llm.set_answer_prompt(answer_prompt)

    # Set output directory and file paths
    output_dir = f"/path/to/{args.data}_mcts/"
    os.makedirs(output_dir, exist_ok=True)
    intermediate_path = (
        f"{output_dir}/mcts_results_re_part_{model_name}_exploration_constant_"
        f"{str(args.exploration_constant)}.pkl"
    )
    final_output_path = (
        f"{output_dir}/mcts_results_re_{model_name}_exploration_constant_"
        f"{str(args.exploration_constant)}.pkl"
    )

    # Load data
    data = json.load(open(f"/path/to/data/{args.data}.json", "r"))
    data_len = len(data)

    # Initialize results list
    results = []

    # Try to load intermediate results
    if os.path.exists(intermediate_path):
        with open(intermediate_path, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded existing results with {len(results)} entries")

        # Reprocess error entries
        reprocess_indices = [
            idx for idx, res in enumerate(results) 
            if res["answer"] == "error" or res["root_node"] is None
        ]
        print(f"Found {len(reprocess_indices)} entries to reprocess")

        for idx in tqdm(reprocess_indices, desc="Reprocessing errors"):
            try:
                question, top_ent = get_kg_ent(data, idx)
                finder_test = MCTSPathFinder(
                    question=question,
                    topic_entities=top_ent,
                    llm=llm,
                    max_depth=args.depth,
                    num_retain_entity=args.width,
                    max_iterations=args.iterations,
                    score_threshold=0.8,
                )
                finder_test.search()
                best_node = finder_test._get_best_node()
                answer = llm.generate_answer(question, best_node.y) if best_node else None

                results[idx] = {
                    "question": question,
                    "answer": answer,
                    "root_node": finder_test.root,
                }
            except Exception as e:
                print(f"Error reprocessing index {idx}: {e}")
                results[idx] = {
                    "question": question,
                    "answer": "error",
                    "root_node": None,
                }

            # Immediately save modified results
            try:
                with open(intermediate_path, "wb") as f:
                    pickle.dump(results, f)
            except Exception as e:
                print(f"Failed to save after reprocessing index {idx}: {e}")

    # Process new data
    start_idx = len(results)
    for idx in tqdm(range(start_idx, data_len), desc="Processing new questions"):
        try:
            question, top_ent = get_kg_ent(data, idx)

            finder_test = MCTSPathFinder(
                question=question,
                topic_entities=top_ent,
                llm=llm,
                max_depth=3,
                num_retain_entity=5,
                max_iterations=5,
                score_threshold=0.8,
                exploration_constant=args.exploration_constant,
            )

            finder_test.search()
            best_node = finder_test._get_best_node()
            answer = llm.generate_answer(question, best_node.y) if best_node else None

            results.append({
                "question": question,
                "answer": answer,
                "root_node": finder_test.root,
            })

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            results.append({
                "question": question,
                "answer": "error",
                "root_node": None,
            })

        if (idx + 1) % 10 == 0 or idx == data_len - 1:
            try:
                with open(intermediate_path, "wb") as f:
                    pickle.dump(results, f)
                print(f"Saved intermediate results up to index {idx + 1}")
            except Exception as e:
                print(f"Failed to save results at index {idx + 1}: {e}")

    # Final save to a different file
    try:
        with open(final_output_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Final results saved to {final_output_path}")
    except Exception as e:
        print(f"Failed to save final results: {e}")

    print("Processing completed.")

if __name__ == "__main__":
    main()