import json
import os

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
)

BASE_FILENAMES = [
    f"data/{{}}-1.txt",
    f"data/{{}}-2.txt",
    f"data/{{}}-3.txt",
    f"data/{{}}-4.txt",
    f"data/{{}}-5.txt",
]

INSTRUCTION = "Summarize the following story in my style"
INSTRUCTIONS_FILENAME = "instructions.jsonl"
STORIES_FILENAME = "stories.jsonl"
SUMMARIES_FILENAME = "summaries.jsonl"

print(f"{'Total':<12}{'Instruction':<12}{'Story':<12}{'Summary':<12}")
for base_filename in BASE_FILENAMES:
    # Read the story and the summary files
    with open(base_filename.format("story"), 'r') as file:
        story = "".join(file.readlines())
    with open(base_filename.format("summary"), "r") as file:
        summary = "".join(file.readlines())
    
    # Count tokens
    instruction_tokens = tokenizer(INSTRUCTION, return_tensors="pt")["input_ids"].shape[1]
    story_tokens = tokenizer(story, return_tensors="pt")["input_ids"].shape[1]
    summary_tokens = tokenizer(summary, return_tensors="pt")["input_ids"].shape[1]
    # Print table of tokens
    total_tokens = instruction_tokens + story_tokens + summary_tokens
    print(f"{total_tokens:<12}{instruction_tokens:<12}{story_tokens:<12}{summary_tokens:<12}")

    # Write the training files
    with open(INSTRUCTIONS_FILENAME, 'a') as file:
        file.write(json.dumps({"text": INSTRUCTION}) + "\n")
    with open(STORIES_FILENAME, 'a') as file:
        file.write(json.dumps({"text": story}) + "\n")
    with open(SUMMARIES_FILENAME, 'a') as file:
        file.write(json.dumps({"text": summary}) + "\n")