# IronClad Implementation Assignment

## Prerequisites

Before starting, make sure you have completed the following:

1. **Review lectures** on Visual Search

2. **Study tutorials and resources**:

   * [Introduction to FAISS](https://www.pinecone.io/learn/series/faiss/faiss-tutorial/)
   * Skeleton code provided with docstrings for each function.

3. **Prepare datasets and resources**:

   * `ironclad/storage/probe`: unzipped multi-identity probe dataset
   * `ironclad/storage/gallery`: unzipped multi-identity gallery dataset
   * Skeleton code provided with docstrings for each function. The modules of the Extraction Service (`embedding.py` and `preprocessing.py`) is already implemented for you!
   * Do **NOT** commit these datasets into your GitHub repository.

## Required Libraries

* Standard libraries (`os`, `sys`, `math`, `itertools`, etc.)
* `torch`
* `torchvision`
* `facenet-pytorch`
* `PIL`
* `numpy`
* `matplotlib` / `seaborn`
* `pandas`
* `Flask`

> **Note:** The updated `requirements.txt` includes all required packages. 

## Objectives

By the end of this assignment you will:

* Implement and extend the **Indexing and Search Service**
* Package the Extraction and Retrieval Service as a **Dockerized container**

## Instructions

Use the link in the assignment instructions to fork the IronClad base repository for this assignment into your personal GitHub account. Please update your current repository as there may be updates to the repository. Make your changes as directed by the instructions and push your changes into your repository. *Unit test automatically runs when you push new commits to your repository.*

### Part 1: Inference Service

1. **`bruteforce.py`**

   * Implement `__init__()` to initialize the Brute Force index.
   * Implement `add_embeddings()` to add new embeddings and their associated metadata to the index.
   * Implement `get_metadata()` to retrieves the metadata associated with a particular embedding index.

2. **`hnsw.py`**

   * Implement `__init__()` to initialize the HNSH index.
   * Implement `add_embeddings()` to add new embeddings and their associated metadata to the index.
   * Implement `get_metadata()` to retrieves the metadata associated with a particular embedding index.

3. **`lsh.py`**

   * Implement `__init__()` to initialize the LSH index.
   * Implement `add_embeddings()` to add new embeddings and their associated metadata to the index.
   * Implement `get_metadata()` to retrieves the metadata associated with a particular embedding index.

4. **`search.py`**

   * Implement `search()` to perform a nearest neighbor search and retrieve the associated metadata.

5. **`app.py`**

    * Implment `add()` to add a provided image to the gallery with an associated name.
    * Implment `identify()` to process the probe image to identify top-k identities in the gallery..

> Note: Although it is not required, it is recommended to implement a `utils/metrics.py` to handle the retrieval metrics for the Analysis and Design Assignment.

## Submission & Evaluation

Check in (i.e., git push) all implementation files into your provisioned GitHub repository. Provide the **repository URL** in your Canvas submission before the deadline to receive credit. After checking this in, GitHub Classroom with automatically run the autograder. 

1. **Push your work**  
   Check in (i.e., `git push`) all implementation files into your provisioned GitHub repository.  

2. **Submit your repository URL**  
Provide the **repository URL** before the deadline to receive credit. 

3. **Autograder execution**  
Once you push to GitHub, **GitHub Classroom** will automatically run the autograder on your submission.  
- You can view the results under the **Actions** tab of your repository.  
- Each push to your repository will re-trigger the autograder.  
- The autograder runs unit tests to check correctness of your code and may also check for:
  - File naming conventions  
  - Method signatures  
  - Correctness of outputs  
  - Presence of required files

4. **Checking your grade**  
- Go to your repository on GitHub.  
- Click the **Actions** tab.  
- Select the latest workflow run (triggered by your most recent commit).  
- Expand the **Autograder job** to see detailed test results.  
- A ✅ indicates a passed test; a ❌ indicates a failed test.  

5. **Resubmissions**  
- If you fail tests, you can fix your code and push again.  
- Each new commit will re-run the autograder.  
- Only the **latest successful run before the deadline** counts toward your grade.  

---
