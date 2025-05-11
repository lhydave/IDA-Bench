import pandas as pd
import numpy as np

def evaluate_submission(submission_file: str, ground_truth_file: str) -> float:
    """
    Evaluate the classification accuracy of sentiment predictions.
    
    Args:
        submission_file (str): Path to the submission CSV file containing predicted sentiments
        ground_truth_file (str): Path to the ground truth CSV file containing true sentiments
        
    Returns:
        float: Classification accuracy (percentage of correct predictions)
    """
    # Read the CSV files
    submission_df = pd.read_csv(submission_file)
    ground_truth_df = pd.read_csv(ground_truth_file)
    
    # Ensure both dataframes have the same columns
    required_columns = ['id', 'sentiment']
    for df in [submission_df, ground_truth_df]:
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Both files must have: {required_columns}")
    
    # Merge the dataframes on id
    merged_df = pd.merge(
        submission_df,
        ground_truth_df,
        on='id',
        suffixes=('_pred', '_true')
    )
    
    # Calculate accuracy
    correct_predictions = (merged_df['sentiment_pred'] == merged_df['sentiment_true']).sum()
    total_predictions = len(merged_df)
    accuracy = float(correct_predictions / total_predictions)
    
    return accuracy

if __name__ == "__main__":
    # Example usage
    submission_file = "submission.csv"
    ground_truth_file = "groundtruth_df.csv"
    
    try:
        accuracy = evaluate_submission(submission_file, ground_truth_file)
        print(f"Classification Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    except Exception as e:
        print(f"Error evaluating submission: {str(e)}") 