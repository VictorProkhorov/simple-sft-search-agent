import re
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field

from .answer_correctness_metrics import CorrectnessEvaluator


@dataclass
class EvalResult:
    """Full evaluation result"""
    correctness: CorrectnessMetrics
    grounding: GroundingMetrics
    reliability: ReliabilityMetrics
    efficiency: EfficiencyMetrics
    trajectory_analyses: List[TrajectoryAnalysis] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'correctness': self.correctness.to_dict(),
            'grounding': self.grounding.to_dict(),
            'reliability': self.reliability.to_dict(),
            'efficiency': self.efficiency.to_dict(),
        }
    def summary(self) -> str:
        """Human-readable summary"""
        return f"""
                ╔════════════════════════════════════════════╗
                ║             EVALUATION SUMMARY             ║
                ╚════════════════════════════════════════════╝
 
                CORRECTNESS
                    Token F1:           {self.correctness.token_f1:.3f}
                    Exact Match (EM):        {self.correctness.exact_match:.3f}
                    Case-Insensitive (EM):   {self.correctness.case_insensitive_em:.3f}
                    Semantic Sim (Semantic Answer Similarity):       {self.correctness.semantic_similarity_sas or 'N/A'}
                    Semantic Sim (LLM-as-a-Judge):       {self.correctness.semantic_similarity_judge or 'N/A'}
                """

    


class Evaluator:
    """Main evaluation orchestrator"""
    def __init__(self,
                trajectories_path:str='',
                answers_path:str='',
                compute_semantic_sas: bool = False,
                compute_semantic_judge: bool = False):
    
        print("Loading trajectories...", file=sys.stderr)
        self.trajectories:  List[List[Dict]] = self.load_trajectories(trajectories_path)
    
        print("Loading gold answers...", file=sys.stderr)
        self.gold_answers: List[str] = self.load_gold_answers(answers_path)

        self.answer_correctness_evaluator = CorrectnessEvaluator(compute_semantic_sas=compute_semantic_sas,
                                                                compute_semantic_judge=compute_semantic_judge)


    
    def evaluate(self) -> ComprehensiveEvalResult:
        """
        Full evaluation pipeline.
        
        Args:
            trajectories: List of message trajectories
            gold_answers: Ground truth answers
            compute_semantic: Whether to compute expensive semantic similarity
        """
        assert len(self.trajectories) == len(self.gold_answers), "Length mismatch"
        
        # Extract predicted answers
        predicted_answers = [
            self.extract_answer(traj) 
            for traj in self.trajectories
        ]

        questions = [
            self.extract_question(traj)
            for traj in self.trajectories
        ]
        
        # Compute metrics
        correctness = self.answer_correctness_evaluator.compute_correctness(
            self.gold_answers, predicted_answers, questions
        )
        """
        grounding = GroundingEvaluator.compute_grounding(trajectories, gold_answers)
        reliability = ReliabilityEvaluator.compute_reliability(trajectories)
        efficiency = EfficiencyEvaluator.compute_efficiency(trajectories)
        
        # Per-trajectory analysis
        analyses = []
        for i, (traj, gold, pred) in enumerate(zip(trajectories, gold_answers, predicted_answers)):
            tool_calls = GroundingEvaluator.extract_tool_calls(traj)
            reasoning = ""
            for msg in traj:
                if msg.get('role') == 'assistant':
                    match = re.search(r'<think>([\s\S]*?)</think>', msg.get('content', ''))
                    if match:
                        reasoning = match.group(1).strip()
                        break
            
            is_correct = gold.strip().lower() == (pred.strip().lower() if pred else "")
            retrieved = GroundingEvaluator.extract_retrieved_content(traj)
            coverage = GroundingEvaluator.compute_coverage(gold, retrieved)
            
            analyses.append(TrajectoryAnalysis(
                question_id=i,
                gold_answer=gold,
                predicted_answer=pred,
                is_correct=is_correct,
                tool_calls=tool_calls,
                num_turns=len(traj),
                reasoning=reasoning,
                retrieved_docs=[retrieved],
                answer_supported=coverage > 0.5
            ))
        """
        grounding = None
        reliability = None
        efficiency = None
        analyses = None
        return EvalResult(
            correctness=correctness,
            grounding=grounding,
            reliability=reliability,
            efficiency=efficiency,
            trajectory_analyses=analyses
        )


    @staticmethod
    def load_trajectories(file_path: str) -> List[List[Dict]]:
        """Load trajectory data (JSONL format)"""
        trajectories = []
        with open(file_path, 'r') as f:
            for line in f:
                trajectory = json.loads(line)
                trajectories.append(trajectory)
        return trajectories
 
    @staticmethod
    def load_gold_answers(file_path: str) -> List[str]:
        """Load gold answer data (one per line)"""
        answers = []
        with open(file_path, 'r') as f:
            for line in f:
                answers.append(line.strip())
        return answers
 
    
    @staticmethod
    def extract_tool_calls(trajectory: List[Dict]) -> List[Dict]:
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
    
    @staticmethod
    def extract_retrieved_content(trajectory: List[Dict]) -> str:
        """Extract all retrieved/searched content from trajectory"""
        content = []
        for msg in trajectory:
            if msg.get('role') == 'tool':
                content.append(msg.get('content', ''))
        return '\n'.join(content)
    
    @staticmethod
    def extract_answer(trajectory: List[Dict]) -> Optional[str]:
        """Extract final answer from trajectory"""
        for msg in reversed(trajectory):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                match = re.search(r'<answer>([\s\S]*?)</answer>', content)
                if match:
                    return match.group(1).strip()
        return None
    
    def extract_question(self, trajectory: List[Dict]):
        question = trajectory[0]['content'].split('Question:')[1].strip()
        #print(question)
        #exit()
        return question
    

    
    
 

def main():
    model_name = 'Qwen2.5-0.5B-Instruct-Search-LoRA'
    #model_name = 'Qwen2.5-0.5B-Instruct'
    dataset_split = 'validation'

    base_dir = Path(__file__).parent.parent.parent
    trajectories_path = f'{base_dir}/data/trajectories/hotpotqa_{model_name}_{dataset_split}.json'
    gold_answers_path = f'{base_dir}/data/trajectories/hotpotqa_gold_answers_{model_name}_{dataset_split}.txt'
    
    evaluator = Evaluator(trajectories_path=trajectories_path,
                        answers_path=gold_answers_path,
                        compute_semantic_sas=True,
                        compute_semantic_judge=True)
    result = evaluator.evaluate()
    print(result.summary())


if __name__ == '__main__':
    main()