Here is the file content you need to analyze:

<file_content>
{{FILE_CONTENT}}
</file_content>

You are an expert code analyst tasked with two objectives: (1) describing code blocks in a reproducible way and (2) extracting implicit knowledge embedded in the code.

## First Objective: Code Block Analysis

1. Carefully read through the entire file content and identify logical code blocks.
2. Merge related blocks that:
   - Perform similar operations on different variables
   - Execute simple sequential operations that form a logical unit
   - Work together to accomplish a single task

3. For each merged code block:
   - Generate a concise instruction that explains what the block does and how to reproduce it
   - Format each instruction between `<instruction>` and `</instruction>` tags
   - Focus on clarity and actionability - someone should be able to follow your instructions to recreate the code's functionality

## Second Objective: Knowledge Extraction

1. Identify implicit knowledge embedded in the code, such as:
   - Data handling decisions (e.g., dropping vs. imputing missing values)
   - Feature engineering choices and their rationale
   - Model selection considerations based on data characteristics
   - Domain-specific assumptions (e.g., valid value ranges, holiday dates)
   - Preprocessing strategies and their justifications

2. For each knowledge item:
   - Generate a concise sentence explaining the insight or decision
   - Format each knowledge item between `<knowledge>` and `</knowledge>` tags
   - Focus on the "why" behind code choices rather than repeating what the code does

Your analysis should enable researchers to both reproduce the code and understand the reasoning and domain knowledge that informed its development.