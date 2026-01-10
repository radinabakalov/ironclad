# IronClad Analysis and Design Assignment

## Objectives

This assignment will guide you in evaluating the **IronClad System** by:

* Comparing embedding models
* Analyzing the effects of selecting a particular indexing strategy on retrieval performance
* Exploring how to optimize the system by selecting configuration parameters
* Submit a file (`<BASE_DIRECTORY>/CASE_ANALYSIS.md`) to summarize your findings.
* Submit supporting Jupyter notebooks (`<BASE_DIRECTORY>/analysis/task*.ipynb`) to show your work.

## Tasks

Please compile all your findings—including every table, figure, and succinct commentary—into a single, cohesive document (`CASE_ANALYSIS.md`).

* Each **table** and **figure** must be clearly labeled (e.g., *Table 1: Model Comparison Metrics*, *Figure 2: PR Curves*).
* Captions should concisely describe what the reader should observe.
* After each visualization or table, provide a **brief interpretation** explaining the key takeaway (e.g., *Table 3 shows a 7‑point drop in mAP when brightness is reduced by 50%*), and explain how this finding/insight would impact your design choices of your system.
* Do **NOT** include step-by-step code. 

Maintain consistent formatting throughout: uniform fonts, spacing, numbering, and captions. Ensure the document reads smoothly with explanations that remind the reader **how  and why each analysis was performed (methodology)** and **how your results supports your overall conclusion**.

You will be evaluated on:

* Robustness of your methodology and reasoning (are your conclusions well-supported by evidence? and is your reasoning correct?)
* Depth and completeness of your analysis (did you explore trade-offs and alternative explanations?)
* Clarity and cohesion of your presentation (does your report read smoothly as a logical narrative?)

For each of the five tasks, include a dedicated Jupyter Notebook (`analysis/task1.ipynb`, `analysis/task2.ipynb`, …, and `analysis/task5.ipynb`) showing your analysis, intermediate exploration, and any code used. Each notebook should clearly demonstrate how you arrived at your reported results, even if the final report only summarizes them. The evaluation will not consider the contents contained in these notebooks. These will only be used to clarify details missing in your markdown file, if need be.

### Task Breakdown

1. **Model Performance Comparison (Model Selection)**: Compare the overall performance of `casia-webface` and `vggface` on the full IronClad dataset. For this selection process, include the impact of environmental noise (e.g., Gaussian blur, resizing, and brightness adjustments) on the `casia-webface` and `vggface` performance. Argue for which model should be selected.

💡 **Hint:** Use the `BruteForce` approach which will guarantee to return the best identities. 

2. **Threshold Design (Indexing Selection)**: Measure the impact of selecting Brute Force, HNSW, LSH indexing strategy on the systems' retrieval performance. Compare their performance on a billion images (as per the requirements).

3. **Number of Identies Returned (Parameter Configuration)**: Define `N` as the final number of candidate identities the system returns to the user (i.e., Top-N nearest neighbors). Argue and justify for the best `N` for `casia-webface` and `vggface` given the provided dataset (probe and gallery). Finally, show how `N` will change between Brute Force vs HNSW vs LSH.

4. **Optimize the Number of Images in Gallery (Dataset Design)**: Define `m_i` as the number of images the gallery contains for individual `i`. Investigate how retrieval performance on `casia-webface` and `vggface` as `m_i` vary (i.e., `m = 1, 2, 3, ...`). Suggest the optimal `m`, supported by your findings, and discuss dataset-specific factors that may influence your conclusion. 

HINT: This analysis could be a little tricky! Think about how would you design an objective/fair experiment.

5. **Uncertainty Estimation (Robustness Design)**: Identify those individuals in the dataset on which the system performs poorly and those who perform well. Characterize their behavior in the image space and the embedding space. Without manipulating the images in the gallery (i.e., do not use Task 4), propose, implement, and evaluate a strategy to improve the performance of your best model in Task 1.


## Submission

* Commit and push your **`CASE_ANALYSIS.md`** file and your supporting notebooks (e.g., `analysis/task*.ipynb`) into your provisioned GitHub repository.
* Provide your **repository URL** when submitting.


