import asyncio
import json
import os
import random
import openai
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 1. Enforce structured JSON output
class QAPair(BaseModel):
    question: str = Field(description="A highly complex, difficult question that combines the two topics.")
    answer: str = Field(description="A detailed, step-by-step, accurate answer explaining the technical nuances.")
    topics_used: str = Field(description="The specific intersection of the two topics.")

class DatasetBatch(BaseModel):
    qa_pairs: list[QAPair] = Field(description="A list of generated Q&A pairs.")

# Global flag to track if we've run out of credits
OUT_OF_MONEY = False

# 2. Worker function
async def generate_synthetic_data(domain_topics: list[str], semaphore: asyncio.Semaphore, output_file: str):
    global OUT_OF_MONEY
    
    # If we already know we're broke, abort early
    if OUT_OF_MONEY:
        return
        
    async with semaphore:
        # Double check inside the lock just in case
        if OUT_OF_MONEY:
            return
            
        try:
            selected_topics = random.sample(domain_topics, 2)
            topic_prompt = f"Topic 1: {selected_topics[0]} | Topic 2: {selected_topics[1]}"
            
            response = await client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a master dataset generator for training advanced AI models. "
                            "I will provide two distinct technical or domain topics. "
                            "Your task is to invent a highly complex, real-world edge-case scenario where these "
                            "two topics intersect. Then, generate 2 to 3 extremely difficult Q&A pairs based on that scenario. "
                            "Do NOT generate basic textbook questions. Focus on debugging, architecture, theoretical limits, or catastrophic failures."
                        )
                    },
                    {
                        "role": "user", 
                        "content": topic_prompt
                    }
                ],
                response_format=DatasetBatch,
                temperature=0.85,
            )
            
            result = response.choices[0].message.parsed
            
            if result and result.qa_pairs:
                with open(output_file, "a", encoding="utf-8") as f:
                    for qa in result.qa_pairs:
                        f.write(qa.model_dump_json() + "\n")
                        
            print(".", end="", flush=True)
            
        except openai.RateLimitError as e:
            error_message = str(e).lower()
            # OpenAI throws a rate limit error when you exceed your quota/billing limit
            if "quota" in error_message or "exceeded" in error_message or "billing" in error_message:
                if not OUT_OF_MONEY:
                    print(f"\n[ALERT] API Budget Exhausted! You have successfully drained your remaining credits.")
                    OUT_OF_MONEY = True
            else:
                # Standard rate limit (e.g., Requests Per Minute limit). Just wait a moment and back off.
                await asyncio.sleep(5)
                
        except openai.APIError as e:
             if "quota" in str(e).lower() or "billing" in str(e).lower():
                 if not OUT_OF_MONEY:
                    print(f"\n[ALERT] API Budget Exhausted! You have successfully drained your remaining credits.")
                    OUT_OF_MONEY = True
                    
        except Exception as e:
            # We silently pass other random network errors so the script doesn't crash overnight
            pass

async def main():
    tech_topics = [
        "Kubernetes Networking", "React Server Components", "PostgreSQL Deadlocks",
        "Rust Memory Safety", "High-Frequency Trading Algorithms", "GraphQL Rate Limiting",
        "WebSockets", "OAuth 2.0 Security Flows", "Microservices Event Sourcing",
        "Docker Container Escapes", "Linux Kernel Panics", "B-Tree Database Indexing",
        "CUDA GPU Programming", "JWT Token Hijacking", "Distributed Consensus (Raft)",
        "CSS Grid Edge Cases", "Python Asyncio Event Loops", "Serverless Cold Starts",
        "Kafka Consumer Groups", "Graph Databases (Neo4j)", "WebRTC Video Streaming"
    ]
    
    output_file = "from_scratch_dataset.jsonl"
    
    # We will leave the existing file alone and just append to it!
    
    semaphore = asyncio.Semaphore(20)
    
    print("=====================================================")
    print("Starting endless dataset generation until credits run out.")
    print("=====================================================")
    
    batch_size = 50
    total_batches = 0
    
    # Infinite loop! It will only stop when OpenAI tells us we are out of money.
    while not OUT_OF_MONEY:
        # We process in batches of 50 to avoid blowing up the computer's memory with an infinite list of tasks
        tasks = [generate_synthetic_data(tech_topics, semaphore, output_file) for _ in range(batch_size)]
        
        # Wait for this batch to finish
        await asyncio.gather(*tasks)
        
        if not OUT_OF_MONEY:
            total_batches += 1
            print(f" [Generated {total_batches * batch_size} API calls... still going!]")
            
    print(f"\nDone! Your credits are drained. Your final dataset is safe in '{output_file}'.")

if __name__ == "__main__":
    asyncio.run(main())
