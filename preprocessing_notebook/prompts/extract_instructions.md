Here is the file content you need to analyze:

<file_content>
{{FILE_CONTENT}}
</file_content>

You are an expert code analyst tasked with describing code blocks in a way that enables researchers to reproduce the code accurately. Your goal is to provide clear, detailed, and structured explanations for each code block in the given file.

Instructions:
1. Carefully read through the entire file content.
2. Identify each distinct code block within the file.
3. For each code block, follow these steps:
   a. Analyze the code thoroughly.
   b. Write a detailed description of the code block.
   c. Ensure your description serves as a clear set of instructions for researchers to reproduce the code.

Important guidelines:
- Provide one complete description for each code block.
- If several blocks perform the same operation on parallel variables (e.g., filling missing values in different columns), merge their explanations into one consolidated instruction.
- Write your description as a single, cohesive paragraph.
- Avoid using itemized lists or bullet points in your descriptions.
- Focus on explaining what the code does and how to reproduce it step by step.
- Include any necessary context or prerequisites for running the code.

Before providing your final output, wrap your analysis inside <code_analysis> tags in your thinking block. In this analysis:
- Break down the code block into its main components.
- Identify any libraries or dependencies used.
- List out the key functions or operations performed.
- Provide a step-by-step breakdown of the code's execution flow.

This will help ensure a thorough interpretation of the code.

Output Format:
For each code block, structure your output as follows:

<code_block>
[Insert the code block here]
</code_block>

<code_analysis>
[Your detailed analysis of the code block, including key components, functionality, and any important considerations]
</code_analysis>

<description>
[A single, cohesive paragraph describing the code block and providing instructions for reproduction]
</description>

Please proceed with your analysis and description of each code block in the file. Your final output should consist only of the structured content (code blocks, descriptions) and should not duplicate or rehash any of the work you did in the code analysis section.
