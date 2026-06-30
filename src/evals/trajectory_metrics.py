
import re
import json
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
 
import torch
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer

@dataclass
class Trajectory_Metrics:
    
    answer_format_rate: float
    av_num_reasoning_steps: float
    has_reasoning: float
    has_tool_call: float
    av_num_tool_calls: float
    tool_error_rate: float
    av_num_turns_per_traj: float
    
    def to_dict(self):
        return {
            'answer_format_rate': self.answer_format_rate,
            'av_num_reasoning_steps': self.av_num_reasoning_steps,
            'has_reasoning': self.has_reasoning,
            'has_tool_call': self.has_tool_call,
            'av_num_tool_cals': self.av_num_tool_calls,
            'tool_error_rate': self.tool_error_rate,
            'av_num_turns_per_traj': self.av_num_turns_per_traj
        }


class Trajectory_Evaluator:
    def __init__(self):
        """Evaluate answer grounding in retrieved evidence"""
        pass

    def count_turns(self, trajectory: List[Dict]) -> int:
        """Count conversation turns"""
        return len(trajectory)


    def extract_tool_calls(self, trajectory: List[Dict]) -> List[Dict]:
        """Extract all tool calls from trajectory"""
        tool_calls = []
        for msg in trajectory:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Parse <tool_call> blocks
                pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
                for match in re.finditer(pattern, content, re.DOTALL):
                    try:
                        tool_call = json.loads(match.group(1))
                        tool_calls.append(tool_call)
                    except json.JSONDecodeError:
                        continue
        return tool_calls
      

    def compute_tool_calls(self, trajectories: List[List[Dict]]) -> Tuple[float, List[int]]:
        """Compute % of trajectories with tool calls"""
        tool_call_counts = []
        for trajectory in trajectories:
            tool_calls = self.extract_tool_calls(trajectory)
            tool_call_counts.append(len(tool_calls)) 
        return tool_call_counts


    def count_tool_errors(self, trajectory: List[Dict]) -> int:
        """Count failed tool calls"""
        errors = 0
        for msg in trajectory:
            if msg.get('role') == 'tool':
                content = msg.get('content', '').lower()
                if 'error' in content or 'failed' in content:
                    errors += 1
        return errors
    

    def extract_reasoning_blocks(self, trajectory: List[Dict]) -> List[str]:
        """
        Extract all well-formed reasoning blocks from trajectory.
        
        Validates that blocks have both opening and closing tags.
        
        Returns list of reasoning block contents (empty list if none found).
        """
        reasoning_blocks = []
        for msg in trajectory:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Find all well-formed <think>...</think> blocks
                pattern = r'<think>([\s\S]*?)</think>'
                matches = re.findall(pattern, content)
                reasoning_blocks.extend(matches)
        
        return reasoning_blocks
    

    def count_reasoning_blocks(self, trajectory: List[Dict]) -> int:
        """
        Count well-formed reasoning blocks in trajectory.
        
        Returns count of <think>...</think> blocks.
        """
        return len(self.extract_reasoning_blocks(trajectory))
    

    def has_reasoning(self, trajectory: List[Dict]) -> bool:
        """
        Check if trajectory contains at least one well-formed reasoning block.
        
        A well-formed reasoning block has both <think> and </think> tags.
        Returns True if count > 0, False otherwise.
        """
        return self.count_reasoning_blocks(trajectory) > 0


    def can_extract_answer(self, trajectory: List[Dict]) -> bool:
        """Check if answer can be extracted"""
        for msg in reversed(trajectory):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                match = re.search(r'<answer>([\s\S]*?)</answer>', content)
                if match:
                    return True
        return False


    def compute(self, trajectories: List[List[Dict]]) -> GroundingMetrics:
        n = len(trajectories)

        # number of trajectories that have an answer in the valid format
        total_answers_within_format = sum(1 for traj in trajectories if self.can_extract_answer(traj))
        
        # number of reasoning steps within the format
        total_number_of_reasoning_steps = sum(self.count_reasoning_blocks(traj) for traj in trajectories)
        total_number_of_traj_has_reasoning = sum(1 for traj in trajectories if self.has_reasoning(traj))

        
        # Tools 
        tool_call_counts = self.compute_tool_calls(trajectories)
        total_number_of_traj_has_tool_call = sum(1 for c in tool_call_counts if c > 0)
        total_number_of_tool_cals = sum(tool_call_counts)
        total_number_tool_errors = sum(self.count_tool_errors(traj) for traj in trajectories)

        total_num_turns_in_traj  = sum(self.count_turns(traj) for traj in trajectories)

        
        return Trajectory_Metrics(
            answer_format_rate = float(total_answers_within_format) / n if n > 0 else 0.0,
            av_num_reasoning_steps = float(total_number_of_reasoning_steps) / n if n > 0 else 0.0,
            has_reasoning = float(total_number_of_traj_has_reasoning) / n if n > 0 else 0.0,
            has_tool_call = float(total_number_of_traj_has_tool_call) / n if n > 0 else 0.0,
            av_num_tool_calls = float(total_number_of_tool_cals) / n if n > 0 else 0.0,
            tool_error_rate = float(total_number_tool_errors) / total_number_of_tool_cals if total_number_of_tool_cals > 0 else 0.0,
            av_num_turns_per_traj = float(total_num_turns_in_traj) / n if n > 0 else 0.0
        )
