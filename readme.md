# A Nano Search-Agent (Under Development)

## (SFT) Fine-Tuning Pipeline
 
A pipeline for synthesizing, filtering, and fine-tuning a Qwen2.5-Instruct model to perform multi-turn tool-use (web search / Wikipedia retrieval) reasoning on HotpotQA-style question answering, following a Search-R1 / R1-Searcher-style `<think>/<tool_call>/<tool_response>/<answer>` format.
 
## Pipeline Overview

## Project Structure
 
python3 -m src.data.synthesis.trajectory_collection

python3 -m src.evals.evaluate