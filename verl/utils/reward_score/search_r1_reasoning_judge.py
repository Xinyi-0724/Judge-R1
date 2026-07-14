# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from search_r1_like_qa_em.py for reasoning judge tasks

import random
import re
import json
from typing import Set, Union


def f1_score(pred_set: Set[int], truth_set: Set[int]) -> float:
    """
    Compute F1 score between two sets of integers.
    
    Args:
        pred_set: Set of predicted violated principles
        truth_set: Set of ground truth violated principles
        
    Returns:
        float: F1 score between 0 and 1
    """
    intersection = len(pred_set & truth_set)
    
    if not pred_set and not truth_set:
        return 1.0  # edge case: both empty => perfect
    if not pred_set or not truth_set:
        return 0.0
        
    precision = intersection / len(pred_set)
    recall = intersection / len(truth_set)
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * precision * recall / (precision + recall)


def extract_solution(solution_str):
    """Extract the JSON solution from the last <answer>...</answer> tags."""
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0 matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    """Count opening and closing answer tags."""
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")
    return opening_tags, closing_tags


def parse_json_answer(answer_str):
    """
    Parse the JSON answer and extract comparison and violated_principles.
    
    Args:
        answer_str: JSON string from <answer> tags
        
    Returns:
        tuple: (comparison, violated_principles_set, is_valid_format)
    """
    if not answer_str:
        return None, set(), False
        
    try:
        data = json.loads(answer_str)
        
        # Check if required keys exist
        if not isinstance(data, dict) or 'comparison' not in data or 'violated_principles' not in data:
            return None, set(), False
            
        comparison = data['comparison']
        violated_principles = data['violated_principles']
        
        # Validate comparison format
        if comparison not in ['A>B', 'A<B']:
            return None, set(), False
            
        # Validate violated_principles format
        if not isinstance(violated_principles, list):
            return None, set(), False
            
        # Convert to set of integers and validate range
        violated_set = set()
        for item in violated_principles:
            if not isinstance(item, int) or item < 1 or item > 6:
                return None, set(), False
            violated_set.add(item)
            
        return comparison, violated_set, True
        
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, set(), False

def extract_search_turns(solution_str):
    """Extract the number of <search>...</search> turns in the solution string."""
    search_pattern = r"<search>.*?</search>"
    matches = re.findall(search_pattern, solution_str, re.DOTALL)
    return len(matches)


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """
    The scoring function for search-augmented reasoning judge tasks.
    
    Scoring breakdown:
    - R0 (Format): 0.0 for malformed JSON, format_score for well-formed but incorrect
    - R1 (Comparison): 2.0 if exact match with GT ("A>B" or "A<B"), else -2.0
    - R2 (Violations): 2*(F1-0.5) where F1 is between predicted and ground-truth sets
    - Final reward = (R1 + R2 + 3) / 6, guarantees output in [0, 1]
    - Anti-spam: Format penalties for too many answer tags
    
    Args:
        solution_str: the solution text
        ground_truth: the ground truth data
        method: the method to extract the solution (unused, kept for compatibility)
        format_score: the score for well-formed but incorrect answers
        score: the maximum score for correct answers (unused in this implementation)
    """
    # Extract search turns
    num_search = extract_search_turns(solution_str)



    # Extract answer from <answer> tags
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Ground truth: {ground_truth}")
        if answer is not None:
            print(f"Extracted answer: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    # Parse ground truth
    if isinstance(ground_truth, dict) and 'target' in ground_truth:
        # Handle the nested structure from parquet data
        gt_targets = ground_truth['target']
        if hasattr(gt_targets, 'tolist'):  # numpy array
            gt_targets = gt_targets.tolist()
        if isinstance(gt_targets, list) and len(gt_targets) > 0:
            gt_str = gt_targets[0]  # Take first target
        else:
            return {"score": 0.0, "R1": 0.0, "R2": 0.0, "final_reward": 0.0, "num_search": num_search}
    else:
        gt_str = ground_truth
    
    try:
        gt_data = json.loads(gt_str)
        gt_comparison = gt_data['comparison']
        gt_violated = gt_data['violated_principles']
        if hasattr(gt_violated, 'tolist'):  # numpy array
            gt_violated_set = set(gt_violated.tolist())
        else:
            gt_violated_set = set(gt_violated)
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"score": 0.0, "R1": 0.0, "R2": 0.0, "final_reward": 0.0, "num_search": num_search}

    # If no answer extracted, return 0
    if answer is None:
        return {"score": 0.0, "R1": 0.0, "R2": 0.0, "final_reward": 0.0, "num_search": num_search}

    # Parse the JSON answer
    pred_comparison, pred_violated_set, is_valid_format = parse_json_answer(answer)
    
    # If format is invalid, return 0
    if not is_valid_format:
        return {"score": 0.0, "R1": 0.0, "R2": 0.0, "final_reward": 0.0, "num_search": num_search}
    
    # Anti-spam guard: penalize too many answer tags
    spam_penalty = 1.0
    if open_count > 10 or close_count > 10:
        spam_penalty = 0.25
    
    # Compute R1 (comparison score)
    R1 = 2.0 if pred_comparison == gt_comparison else -2.0
    
    # Compute R2 (violated principles F1 score)
    f1 = f1_score(pred_violated_set, gt_violated_set)
    R2 = 2.0 * (f1 - 0.5)  # Scale F1 to [-1, 1] range
    
    # Final reward: (R1 + R2 + 3) / 6 to guarantee output in [0, 1]
    # R1 ∈ [-2, 2], R2 ∈ [-1, 1], so R1 + R2 ∈ [-3, 3]
    # Therefore (R1 + R2 + 3) ∈ [0, 6], and (R1 + R2 + 3) / 6 ∈ [0, 1]
    final_reward = (R1 + R2 + 3.0) / 6.0
    
    # Apply spam penalty
    final_reward *= spam_penalty
    
    if do_print:
        print(f"Predicted comparison: {pred_comparison}, GT: {gt_comparison}")
        print(f"Predicted violated: {pred_violated_set}, GT: {gt_violated_set}")
        print(f"R1: {R1}, R2: {R2}, F1: {f1}")
        print(f"Final reward: {final_reward}")
        print(f"Spam penalty: {spam_penalty}")
    
    return {
        "score": final_reward,
        "R1": R1,
        "R2": R2, 
        "final_reward": final_reward,
        "num_search": num_search
    }



def compute_score_detailed(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """
    Detailed scoring function that returns breakdown of scores for analysis.
    
    Returns:
        dict: Detailed score breakdown including R1, R2, F1, and final reward
    """
    # Extract answer from <answer> tags
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)

    # Parse ground truth
    if isinstance(ground_truth, dict) and 'target' in ground_truth:
        gt_targets = ground_truth['target']
        if hasattr(gt_targets, 'tolist'):  # numpy array
            gt_targets = gt_targets.tolist()
        if isinstance(gt_targets, list) and len(gt_targets) > 0:
            gt_str = gt_targets[0]
        else:
            return {"score": 0.0, "R1": 0.0, "R2": 0.0, "f1_score": 0.0, "format_valid": False, "spam_penalty": 1.0}
    else:
        gt_str = ground_truth
    
    try:
        gt_data = json.loads(gt_str)
        gt_comparison = gt_data['comparison']
        gt_violated = gt_data['violated_principles']
        if hasattr(gt_violated, 'tolist'):
            gt_violated_set = set(gt_violated.tolist())
        else:
            gt_violated_set = set(gt_violated)
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"score": 0.0, "R1": 0.0, "R2": 0.0, "f1_score": 0.0, "format_valid": False, "spam_penalty": 1.0}

    if answer is None:
        return {"score": 0.0, "R1": 0.0, "R2": 0.0, "f1_score": 0.0, "format_valid": False, "spam_penalty": 1.0}

    # Parse the JSON answer
    pred_comparison, pred_violated_set, is_valid_format = parse_json_answer(answer)
    
    if not is_valid_format:
        return {"score": 0.0, "R1": 0.0, "R2": 0.0, "f1_score": 0.0, "format_valid": False, "spam_penalty": 1.0}
    
    # Anti-spam guard
    spam_penalty = 1.0
    if open_count > 10 or close_count > 10:
        spam_penalty = 0.25
    
    # Compute scores
    R1 = 2.0 if pred_comparison == gt_comparison else -2.0
    f1 = f1_score(pred_violated_set, gt_violated_set)
    R2 = 2.0 * (f1 - 0.5)
    # Final reward: (R1 + R2 + 3) / 6 to guarantee output in [0, 1]
    final_reward = (R1 + R2 + 3.0) / 6.0
    final_reward *= spam_penalty
    
    return {
        "score": final_reward,
        "R1": R1,
        "R2": R2,
        "f1_score": f1,
        "format_valid": is_valid_format,
        "spam_penalty": spam_penalty,
        "comparison_correct": pred_comparison == gt_comparison,
        "predicted_comparison": pred_comparison,
        "predicted_violated": list(pred_violated_set),
        "gt_comparison": gt_comparison,
        "gt_violated": list(gt_violated_set)
    }