Here is the file content you need to process:
<file_content>
{{FILE_CONTENT}}
</file_content>
with the identified evaluation-metric name:
<numerical_info>
{{METRIC_INFO}}
</numerical_info>
and all data files under the directory
<directory>{{DATA_DIR}}</directory>

# Required Modifications to the file content

1. Train / test files
 - Load the training data from the given directory, using the file name <original_name>_train.
 - Load the test feature set from the given directory, using <original_name>_test_features.
 - Remove any code that performs a train-test split.
 - All data will be under the directory <directory>{{DATA_DIR}}</directory>, set this directory as the prefix of any .csv files you import.

2. Data processing & modelling
 - Apply all cleaning, preprocessing, and feature-engineering steps only to the training set, then fit the model.
 - Apply the identical transformations to the test feature set. Note that there will not be missing values in the test feature set, so do not drop any rows.
 - Use the fitted model to generate predictions for the test feature set.

3. Save predictions
 - Write the predictions to the data path <path>{{SUBMISSION_PATH}}</path>.

4. Output format:
 - Return the modified script wrapped between <code> and </code> tags.
 - Make only the minimal edits needed to satisfy the requirements above.

# Evaluation-function extraction
- Isolate the evaluation function corresponding to {{METRIC_INFO}}.
- Wrap the extracted code (including any required imports) between <evaluation> and </evaluation> tags.
- Standardise the interface:
    The function must accept two CSV-file paths:
```python
    def evaluate(y_true_path: str, y_pred_path: str) -> float:
    ...
```
    - y_true_path -> <original_name>_test.csv
    - y_pred_path -> gt_submission.csv (produced by your previous modified file content)
- Load ground truth inside the function:
    Read <original_name>_test.csv and extract the column(s) containing the true response values before computing the metric.
- Mirror the type-conversion logic used in <file_content>.
    Inspect the code inside <file_content> (and the edits you output inside <code>). Note any transformation applied to the response column—-e.g., mapping "yes" -> 1 and "no" -> 0, or a label-encoder that converts categories to integers.
    Reproduce the same conversion on both y_true (loaded from <original_name>_test.csv) inside the evaluate function.
    Make sure the metric runs without introducing NaNs due to mismatched encodings.

- Remove any code that generates predictions; the block should only evaluate them.
- Keep only essentials; include nothing beyond the imports, helper code, and the evaluation function required for it to run.

Remember: double-check your response before submitting.









