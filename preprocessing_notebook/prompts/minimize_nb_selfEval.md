Given the markdown file with multiple code blocks, please:

1. Extract numerical result, metric and response variable:

- Identify the **most important** quantitative conclusion or final numerical result presented in this file. This typically involves a **performance metric** (e.g., accuracy, RMSE, MAE, MSE, RMLSE) used to evaluate predictions of a specific **response variable**. **Never use R^2 as the evaluation metric if there are other metrics**

- Determine the name of the **response variable**'s column in the dataset (e.g., if the prediction is df['growth_rate'], then 'growth_rate' is the response column name). Be accurate and ensure the column name corresponds directly to the original dataset used in the code.

- Format your extracted result as follows:

<main_result>
{
  "metric_name": "Brief description of the evaluation metric in context",
  "metric_value": 123.456,                        // numeric outcome from the original notebook
  "response_columns": ["col_name_1", "col_name_2"] // list of response-variable column names
}
</main_result>


2. Analyze the entire file to determine which code blocks directly contribute to producing this final result by:

   - Identifying data loading/import steps

   - Tracking data transformations that modify the dataset (dropping rows/columns, creating new variables, etc.)

   - Finding the calculation steps that lead to the final result

   - Keep definitions of variables

   - Noting which sections are purely exploratory or visualization-focused

3. Create a cleaned version of the markdown file that:

   - Retains all section headers and code blocks necessary to reproduce the final result

   - Completely removes sections that don't affect the final numerical output (like plotting, data exploration, or checks that don't lead to modifications)

   - Preserves the exact format and content of the necessary code blocks

4. Explain which sections were kept and why they're essential to reproducing the final result.

5. Explain which sections were removed and why they're not essential. 

For example, data loading, cleaning operations that modify the dataset, and final calculations should be kept, while checks that don't lead to modifications, exploratory analysis, and visualization code can be removed.

6. Please provide the markdown formatted between <markdown> and </markdown> tags. Place the entire markdown content between these tags and do NOT use the strings "<markdown>" or "</markdown>" anywhere else in your response.

7. Code quality in the markdown:
   - Remove any unused library imports
   - Remove all package installation commands (like "!pip install", "!conda install", etc.)
   - Remove Jupyter notebook magic commands (like "%matplotlib inline", "%%time", etc.)
   - Ensure all code is clean, properly formatted, and ready for production use