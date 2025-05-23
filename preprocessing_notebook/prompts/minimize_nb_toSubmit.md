Given the markdown file with multiple code blocks, please:

1. Extract numerical metric evaluation function. Note that the evaluation function is presented in the **last code block** of the entire notebook. Wrap the extracted code (including any required imports) between <evaluation> and </evaluation> tags. Apply modifications to the evaluation function as follows:

   - Make the evaluation function self-contained. For example, import necessary libaries that are used in the function.


2. Determine the name of the **response variable**'s column in the dataset (e.g., if the prediction is df['growth_rate'], then 'growth_rate' is the response column name). Be accurate and ensure the column name corresponds directly to the original dataset used in the code.

   - Format your extracted result as follows:
<main_result>
{
  "metric_name": "Brief description of the evaluation metric in context",
  "response_columns": ["col_name_1", "col_name_2"], // list of response-variable column names,
  "is_higher_better": true, // true or false. For example, this is true for accuracy, and false for MSE
  "theoretical_best": 0.0 // The theoretically best achievable value of this metric. For example, this is 0.0 for MSE and 1.0 for accuracy
}
</main_result>

3. Note that the final result of the file is creating a submission file. Analyze the entire file to determine which code blocks directly contribute to producing this final result by:

   - Identifying data loading/import steps

   - Tracking data transformations that modify the dataset (dropping rows/columns, creating new variables, etc.)

   - Finding the calculation steps that lead to the final result

   - Keep definitions of variables

   - Noting which sections are purely exploratory or visualization-focused

4. Create a cleaned version of the markdown file that:

   - Retains all section headers and code blocks necessary to reproduce the final result

   - Completely removes sections that don't affect the final numerical output (like plotting, data exploration, or checks that don't lead to modifications)

   - Removes the evaluation function, since you have already extracted it between the <evaluation> tags.

   - Preserves the exact format and content of the necessary code blocks

   - Preserves submitting the file at the end

5. Explain which sections were kept and why they're essential to reproducing the final result.

6. Explain which sections were removed and why they're not essential. 

For example, data loading, cleaning operations that modify the dataset, and final calculations should be kept, while checks that don't lead to modifications, exploratory analysis, and visualization code can be removed.

7. Please provide the markdown formatted between <markdown> and </markdown> tags. Place the entire markdown content between these tags and do NOT use the strings "<markdown>" or "</markdown>" anywhere else in your response.

8. Code quality in the markdown:
   - Remove any unused library imports
   - Remove all package installation commands (like "!pip install", "!conda install", etc.)
   - Replace any "submission.csv" with "baseline_submission.csv"
   - Remove Jupyter notebook magic commands (like "%matplotlib inline", "%%time", etc.)
   - Ensure all code is clean, properly formatted, and ready for production use
   