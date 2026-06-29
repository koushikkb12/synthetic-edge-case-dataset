# Synthetic High-Tier Edge-Case Dataset

This repository contains a high-quality, synthetic Q&A dataset generated entirely from latent knowledge using OpenAI's gpt-4o-mini.

Instead of scraping generic textbooks, this dataset was built using a Self-Instruct / Topic Intersection methodology. The generator randomly cross-pollinates completely different advanced engineering domains (e.g., "Kubernetes Networking" + "CSS Grid Edge Cases") and forces the AI to invent catastrophic edge-case scenarios and explain how to debug them.

This results in a dataset that trains LLMs on reasoning and architectural troubleshooting rather than memorization.

## Repository Contents

* generate_from_scratch.py: The asynchronous Python script used to generate the dataset. It leverages AsyncOpenAI, Pydantic Structured Outputs, and infinite-loop rate-limit handling.
* from_scratch_dataset.jsonl: The massive raw dataset of generated Q&A pairs.

## How to Generate Your Own Data

You can use the provided script to generate your own datasets efficiently. The script respects rate limits and ensures 100% structured JSON outputs.

### 1. Install Dependencies
```bash
pip install openai pydantic
```

### 2. Set your API Key
Set your OpenAI API key in your environment variables:
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### 3. Customize the Topics
Open generate_from_scratch.py and modify the tech_topics list at the bottom of the file to whatever domain you want to train your model on (e.g., Sales, Medicine, Rust Programming). The broader the topics, the more creative the intersections will be.

### 4. Run the Script
```bash
python generate_from_scratch.py
```
The script will run continuously, automatically batching requests and pausing for rate limits, until your OpenAI budget runs out.

## Dataset Structure
The output is a JSONL file where each line is a guaranteed perfect JSON object containing:
- question: A highly complex scenario question.
- answer: A detailed, step-by-step diagnostic answer.
- topics_used: The two intersecting topics that generated the scenario.
