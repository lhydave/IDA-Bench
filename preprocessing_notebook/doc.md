### `preprocessing_notebook` Module

Transforms a raw notebook directory into benchmark artefacts via an automated pipeline:

1. **Notebook → Markdown**  
   Convert the source notebook to Markdown → `full_notebook_markdown`.

2. **Markdown Minimization (LLM)**  
   Send `full_notebook_markdown` to an external LLM;
   (1) If the original notebook contains evaluation in itself (classified as `selfEval`), we identify the main numerical result of the `full_notebook_markdown`, specifically identify:
      - metric name
      - original column name of response value (for ground truth)
   Then keep only the cells required to reproduce the primary numerical result → `minimized_notebook_markdown`.

   (2) In the other case, if the original notebook contains evaluation in itself (classified as `toSubmit`), we identify the submission file created. Also there will be an intended metric for the submission file, which we will extract from notebook (TODO)

3. **Markdown → Executable Notebook**  
   Transpile `minimized_notebook_markdown` back into a runnable `.ipynb` → `minimized_notebook`.

4. **Set Dataset Path**  
   Given the path to the `notebook_dataset`, reset the directory of loaded data files in the `minimized_notebook` as the `notebook_dataset` path.

5. **Ground‑Truth Generation**   
   Execute `minimized_notebook` on `notebook_dataset`; record the key numerical outputs → `ground_truth`.

6. **Instruction Extraction (LLM)**   
   Call an external LLM to distil essential operations and prompts from `minimized_notebook` → `instructions`.

Each artefact feeds the next step, producing a compact, self‑contained benchmark package ready for LLM‑agent evaluation.
