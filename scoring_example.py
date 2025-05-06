# Set up the logger, this is necessary for error recovering
# this must be imported and configured before any other module
from logger import logger, configure_global_logger

configure_global_logger(log_file="scoring.log")

# Import the necessary classes and modules
from data_manager import NotebookManager, DatasetManager  # noqa: E402
from scoring.scoring_base import Scoring  # noqa: E402
from data_manager.utils import id_to_filename  # noqa: E402
import statistics  # noqa: E402
from typing import Dict, Union, Any, List, Tuple, Set  # noqa: E402
import argparse  # noqa: E402
import sys  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import json  # noqa: E402


def load_skip_ids(file_path: str) -> Set[str]:
    """
    Load notebook IDs to skip from a file.
    
    Args:
        file_path (str): Path to the file containing notebook IDs to skip, one per line.
        
    Returns:
        Set[str]: Set of notebook IDs to skip.
    """
    skip_ids = set()
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Try to open and read the file
        with open(file_path, 'r') as f:
            for line in f:
                notebook_id = line.strip()
                if notebook_id:  # Skip empty lines
                    skip_ids.add(notebook_id)
        logger.info(f"Loaded {len(skip_ids)} notebook IDs to skip from {file_path}")
    except FileNotFoundError:
        logger.warning(f"Skip file {file_path} not found. No notebooks will be skipped.")
    except Exception as e:
        logger.error(f"Error loading skip file {file_path}: {str(e)}")
        logger.warning("No notebooks will be skipped.")
    
    return skip_ids


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Score notebooks and sample them.")
    parser.add_argument('--skip-file', type=str, help='Path to file containing notebook IDs to skip, one per line',
                        default=None)
    parser.add_argument('--scoring-method', type=str, default='sample_with_code_size',
                       help='Scoring method to use. Default: sample')
    parser.add_argument('--sample-count', type=int, default=100,
                       help='Number of notebooks to sample. Default: 100')
    args = parser.parse_args()
    
    # Initialize the managers
    # NOTE: please ensure that you have already downloaded notebooks (meta info, files, code info)
    # and datasets (meta info)
    notebook_manager = NotebookManager("test_data/notebooks")
    dataset_manager = DatasetManager("test_data/datasets")

    # Initialize the scoring module
    scoring = Scoring(
        dataset_manager=dataset_manager,
        notebook_manager=notebook_manager,
        store_path="test_data/scoring/scoring_results.json",
    )

    try:
        # Load notebook IDs to skip
        skip_ids = set()
        if args.skip_file:
            skip_ids = load_skip_ids(args.skip_file)
        
        # Remove any skipped notebooks from previously calculated scores
        for notebook_id in list(scoring.scores.keys()):
            if notebook_id in skip_ids:
                logger.info(f"Removing previously scored notebook {notebook_id} from scores")
                del scoring.scores[notebook_id]
        
        # Force sync to disk after removing notebooks
        scoring.sync(False)
        
        # Step 1: Get the list of available notebooks, excluding those to skip
        logger.info("Starting notebook scoring...")
        all_notebook_ids = notebook_manager.kept_notebooks_ids
        notebook_ids = [nb_id for nb_id in all_notebook_ids if nb_id not in skip_ids]
        
        logger.info(f"Found {len(notebook_ids)} notebooks to score (skipping {len(skip_ids)} notebooks)")

        # Step 2: Score the notebooks
        if not notebook_ids:
            logger.warning("No notebooks available for scoring.")
            return
            
        # Define the scoring method to use
        scoring_method = args.scoring_method
        logger.info(f"Scoring notebooks using method: {scoring_method}")

        # Score all notebooks
        scores = scoring.score_notebooks(set(notebook_ids), scoring_method)
        logger.info(f"Successfully scored {len(scores)} notebooks")

        # Step 3: Print summary of scored notebooks
        # Get total scores for statistics
        score_values = []
        for score_data in scores.values():
            if isinstance(score_data, dict) and "total_score" in score_data:
                score_values.append(float(score_data["total_score"]))
            elif isinstance(score_data, (float, int)):
                score_values.append(float(score_data))
            else:
                logger.warning(f"Unexpected score data type: {type(score_data)}")
                score_values.append(0.0)  # Default value

        # Calculate statistics
        avg_score = statistics.mean(score_values)
        median_score = statistics.median(score_values)
        min_score = min(score_values)
        max_score = max(score_values)

        logger.info("Scoring statistics:")
        logger.info(f"  - Average score: {avg_score:.4f}")
        logger.info(f"  - Median score: {median_score:.4f}")
        logger.info(f"  - Min score: {min_score:.4f}")
        logger.info(f"  - Max score: {max_score:.4f}")

        # Helper function to sort by score
        def get_score_value(item: Tuple[str, Union[float, Dict[str, float]]]) -> float:
            _, score_data = item
            if isinstance(score_data, dict) and "total_score" in score_data:
                return float(score_data["total_score"])
            elif isinstance(score_data, (float, int)):
                return float(score_data)
            else:
                logger.warning(f"Unexpected score data type: {type(score_data)}")
                return 0.0  # Default value

        # Print the details of top 5 notebooks
        top_notebooks = sorted(scores.items(), key=get_score_value, reverse=True)[:5]
        logger.info("Top 5 notebooks by score:")
        for i, (notebook_id, score_data) in enumerate(top_notebooks):
            notebook_info = notebook_manager.get_meta_info(notebook_id)
            if notebook_info:
                logger.info(f"Notebook {i + 1}: {notebook_info.title}")
                
                # Display component scores if available
                if isinstance(score_data, dict) and "total_score" in score_data:
                    logger.info(f"  - Total Score: {float(score_data['total_score']):.4f}")
                    logger.info(f"  - Popularity Score: {float(score_data['popularity_score']):.4f}")
                    logger.info(f"  - Complexity Score: {float(score_data['complexity_score']):.4f}")
                    logger.info(f"  - Dataset Score: {float(score_data['dataset_score']):.4f}")
                    logger.info(f"  - Resource Score: {float(score_data['resource_score']):.4f}")
                    logger.info(f"  - Plot Penalty: {float(score_data['plot_penalty']):.4f}")
                elif isinstance(score_data, (float, int)):
                    logger.info(f"  - Score: {float(score_data):.4f}")
                else:
                    logger.warning(f"Unexpected score data type: {type(score_data)}")
                    logger.info(f"  - Score: 0.0000")
                
                logger.info(f"  - URL: {notebook_info.url}")

        sample_count = args.sample_count
        scoring.sample_scored_notebooks(
            num=sample_count, method="uniform", store_path="test_data/scoring/uniform_sample"
        )
        logger.info("Sampled notebooks saved to test_data/scoring/uniform_sample")

        # Step 5: sample top notebooks for manual inspection
        logger.info(f"Sampling top {sample_count} notebooks for inspection...")
        scoring.sample_scored_notebooks(num=sample_count, method="topk", store_path="test_data/scoring/topk_sample")
        logger.info("Sampled notebooks saved to test_data/scoring/topk_sample")

        # Save the top 100 notebook IDs to a file
        # Sort notebooks by score in descending order
        sorted_notebooks = sorted(scores.items(), key=get_score_value, reverse=True)
        
        # Get the top notebook IDs
        top_notebook_ids = [notebook_id for notebook_id, _ in sorted_notebooks[:sample_count]]
        
        # Save to file
        top_ids_path = "test_data/scoring/top_notebook_ids.txt"
        with open(top_ids_path, "w") as f:
            for notebook_id in top_notebook_ids:
                f.write(f"{notebook_id}\n")
        
        logger.info(f"Saved top {sample_count} notebook IDs to {top_ids_path}")
        
        # Compare with manually picked notebooks
        manually_picked_path = "test_data/scoring/manually_picked_notebook_ids.txt"
        if os.path.exists(manually_picked_path):
            manually_picked_ids = set()
            with open(manually_picked_path, 'r') as f:
                for line in f:
                    notebook_id = line.strip()
                    if notebook_id:  # Skip empty lines
                        manually_picked_ids.add(notebook_id)
            
            # Get top 50 notebooks (or less if not enough)
            top_50_count = min(sample_count, len(sorted_notebooks))
            top_50_ids = set(notebook_id for notebook_id, _ in sorted_notebooks[:top_50_count])
            
            # Find overlap
            overlapping_ids = manually_picked_ids.intersection(top_50_ids)
            overlap_count = len(overlapping_ids)
            
            # Find manually picked notebooks that are NOT in the top 50
            picked_but_not_top = manually_picked_ids - top_50_ids
            
            logger.info(f"Manually picked notebooks: {len(manually_picked_ids)}")
            logger.info(f"Top {top_50_count} scored notebooks: {top_50_count}")
            logger.info(f"Overlap count: {overlap_count} ({overlap_count/len(manually_picked_ids)*100:.2f}% of manually picked)")
            logger.info(f"Manually picked but not in top {top_50_count}: {len(picked_but_not_top)}")
            
            # List overlapping notebooks
            logger.info("Overlapping notebook IDs:")
            for idx, notebook_id in enumerate(overlapping_ids):
                # Find position in top notebooks
                position = next(i for i, (nid, _) in enumerate(sorted_notebooks) if nid == notebook_id)
                notebook_info = notebook_manager.get_meta_info(notebook_id)
                title = notebook_info.title if notebook_info else "Unknown"
                logger.info(f"  {idx}. {notebook_id} (rank: {position}) - {title}")
            
            # Save overlap information
            overlap_path = "test_data/scoring/overlap_with_manually_picked.txt"
            with open(overlap_path, 'w') as f:
                f.write(f"Manually picked notebooks: {len(manually_picked_ids)}\n")
                f.write(f"Top {top_50_count} scored notebooks: {top_50_count}\n")
                f.write(f"Overlap count: {overlap_count} ({overlap_count/len(manually_picked_ids)*100:.2f}% of manually picked)\n")
                f.write(f"Manually picked but not in top {top_50_count}: {len(picked_but_not_top)}\n\n")
                f.write("Overlapping notebook IDs:\n")
                for idx, notebook_id in enumerate(overlapping_ids):
                    position = next(i for i, (nid, _) in enumerate(sorted_notebooks) if nid == notebook_id)
                    notebook_info = notebook_manager.get_meta_info(notebook_id)
                    title = notebook_info.title if notebook_info else "Unknown"
                    f.write(f"{idx}. {notebook_id} (rank: {position}) - {title}\n")
            
            logger.info(f"Saved overlap information to {overlap_path}")
            
            ############ Save top 50 notebooks that are not in manually picked list #############
            unpicked_top_ids = top_50_ids - manually_picked_ids
            logger.info(f"Found {len(unpicked_top_ids)} notebooks in top 50 that were not manually picked")
            
            # Load skip notebook IDs to check if unpicked samples are in the skip list
            skip_rankings = {}
            skip_file_path = args.skip_file if args.skip_file else "test_data/scoring/skip_notebook_ids.txt"
            if os.path.exists(skip_file_path):
                try:
                    with open(skip_file_path, 'r') as f:
                        for idx, line in enumerate(f):
                            notebook_id = line.strip()
                            if notebook_id:  # Skip empty lines
                                skip_rankings[notebook_id] = idx
                    # logger.info(f"Loaded {len(skip_rankings)} notebook IDs from skip file for ranking information")
                except Exception as e:
                    # logger.error(f"Error loading skip file for rankings: {str(e)}")
                    pass
            
            # Save the actual notebook files for unpicked top notebooks
            unpicked_store_path = "test_data/scoring/topk_unpicked_sample"
            if os.path.exists(unpicked_store_path):
                # logger.info(f"Removing existing directory: {unpicked_store_path}")
                shutil.rmtree(unpicked_store_path)
            
            # Create store directory
            # logger.info(f"Creating directory: {unpicked_store_path}")
            os.makedirs(unpicked_store_path)
            
            ######## Save the actual notebook files for manually picked but not top 50
            picked_not_top_path = "test_data/scoring/manually_picked_not_top"
            if os.path.exists(picked_not_top_path):
                # logger.info(f"Removing existing directory: {picked_not_top_path}")
                shutil.rmtree(picked_not_top_path)
            
            # Create store directory
            # logger.info(f"Creating directory: {picked_not_top_path}")
            os.makedirs(picked_not_top_path)
            
            # Process and save manually picked notebooks that are not in top 50
            # logger.info(f"Saving {len(picked_but_not_top)} manually picked notebooks that are not in the top {top_50_count}...")
            successful_count = 0
            errors = {}
            
            # Get ranks for all notebooks
            notebook_ranks = {nid: idx for idx, (nid, _) in enumerate(sorted_notebooks)}
            
            for idx, notebook_id in enumerate(picked_but_not_top):
                try:
                    # Get notebook info and dataset infos
                    notebook_info, dataset_infos = scoring._obtain_info(notebook_id)
                    
                    # Find the rank in the current scoring
                    current_rank = notebook_ranks.get(notebook_id, "NA")
                    rank_str = f"{current_rank}" if current_rank != "NA" else "NA"
                    
                    # Get the score data if available
                    if notebook_id in scores:
                        score_data = scores[notebook_id]
                        if isinstance(score_data, dict) and "total_score" in score_data:
                            total_score = score_data["total_score"]
                        else:
                            total_score = score_data
                    else:
                        score_data = "Not scored"
                        total_score = "NA"
                    
                    # logger.info(f"Processing manually picked notebook {notebook_id} with rank {rank_str}")
                    
                    # Prepare info for JSON
                    info_dict = {
                        "score_data": score_data,
                        "rank": current_rank,
                        "notebook_info": notebook_info.to_dict(),
                        "dataset_infos": {
                            dataset_id: dataset_info.to_dict() for dataset_id, dataset_info in dataset_infos.items()
                        },
                    }
                    
                    # Save notebook info JSON
                    notebook_filename = f"{idx}_{rank_str}_{id_to_filename(notebook_id)}"
                    json_path = os.path.join(picked_not_top_path, f"{notebook_filename}.json")
                    with open(json_path, "w") as f:
                        json.dump(info_dict, f, indent=2, default=str)
                    # logger.info(f"Saved notebook info to {json_path}")
                    
                    # Save actual notebook file
                    notebook_path = notebook_info.path
                    if notebook_path and os.path.exists(notebook_path):
                        dest_path = os.path.join(picked_not_top_path, f"{notebook_filename}.ipynb")
                        shutil.copy2(notebook_path, dest_path)
                        # logger.info(f"Saved notebook {notebook_id} (rank: {rank_str}, score: {total_score}) to {dest_path}")
                        successful_count += 1
                    else:
                        error_msg = f"Notebook file for {notebook_id} not found at {notebook_path}"
                        # logger.warning(error_msg)
                        errors[notebook_id] = error_msg
                
                except Exception as e:
                    error_msg = str(e)
                    # logger.error(f"Error processing notebook {notebook_id}: {error_msg}")
                    errors[notebook_id] = error_msg
            
            logger.info(f"Completed saving manually picked but not top notebooks: {successful_count} notebooks saved to {picked_not_top_path}")
            
            if errors:
                error_count = len(errors)
                error_details = "\n".join([f"{nb_id}: {error}" for nb_id, error in errors.items()])
                logger.warning(f"Failed to process {error_count} notebooks.\nDetails:\n{error_details}")

            # Now process unpicked top notebooks
            logger.info(f"Processing {len(unpicked_top_ids)} notebooks in top 50 that were not manually picked...")
            successful_count = 0
            errors = {}
            
            # Get a list of notebook_id and their position in the top 50
            unpicked_with_positions = [(notebook_id, next(i for i, (nid, _) in enumerate(sorted_notebooks) if nid == notebook_id)) 
                                      for notebook_id in unpicked_top_ids]
            
            # Sort by position
            unpicked_with_positions.sort(key=lambda x: x[1])
            
            for notebook_id, position in unpicked_with_positions:
                skip_rank = skip_rankings.get(notebook_id, "NA")
                skip_rank_str = f"{skip_rank}" if skip_rank != "NA" else "NA"
                
                # logger.info(f"Processing unpicked top notebook {notebook_id} with rank {position}, skip rank: {skip_rank_str}")
                try:
                    # Get notebook info and dataset infos
                    notebook_info, dataset_infos = scoring._obtain_info(notebook_id)
                    
                    # Get the score data
                    score_data = scores[notebook_id]
                    if isinstance(score_data, dict) and "total_score" in score_data:
                        total_score = score_data["total_score"]
                    else:
                        total_score = score_data
                    
                    # Prepare info for JSON
                    info_dict = {
                        "score_data": score_data,
                        "rank": position,
                        "skip_rank": skip_rank,
                        "notebook_info": notebook_info.to_dict(),
                        "dataset_infos": {
                            dataset_id: dataset_info.to_dict() for dataset_id, dataset_info in dataset_infos.items()
                        },
                    }
                    
                    # Save notebook info JSON
                    notebook_filename = f"{position}_{skip_rank_str}_{id_to_filename(notebook_id)}"
                    json_path = os.path.join(unpicked_store_path, f"{notebook_filename}.json")
                    with open(json_path, "w") as f:
                        json.dump(info_dict, f, indent=2, default=str)
                    # logger.info(f"Saved notebook info to {json_path}")
                    
                    # Save actual notebook file
                    notebook_path = notebook_info.path
                    if notebook_path and os.path.exists(notebook_path):
                        dest_path = os.path.join(unpicked_store_path, f"{notebook_filename}.ipynb")
                        shutil.copy2(notebook_path, dest_path)
                        # logger.info(f"Saved notebook {notebook_id} (rank: {position}, skip rank: {skip_rank_str}, score: {total_score}) to {dest_path}")
                        successful_count += 1
                    else:
                        error_msg = f"Notebook file for {notebook_id} not found at {notebook_path}"
                        # logger.warning(error_msg)
                        errors[notebook_id] = error_msg
                
                except Exception as e:
                    error_msg = str(e)
                    # logger.error(f"Error processing notebook {notebook_id}: {error_msg}")
                    errors[notebook_id] = error_msg
            
            logger.info(f"Completed saving unpicked top notebooks: {successful_count} notebooks saved to {unpicked_store_path}")
            
            if errors:
                error_count = len(errors)
                error_details = "\n".join([f"{nb_id}: {error}" for nb_id, error in errors.items()])
                logger.warning(f"Failed to process {error_count} notebooks.\nDetails:\n{error_details}")
        else:
            logger.warning(f"Manually picked notebooks file not found: {manually_picked_path}")

    except Exception as e:
        logger.error(f"Error during scoring process: {str(e)}")
        raise

    logger.info("Scoring process completed")


if __name__ == "__main__":
    main()
