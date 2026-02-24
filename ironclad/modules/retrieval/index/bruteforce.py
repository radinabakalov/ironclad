import numpy as np
import pickle
import faiss

########################################
# TASK: Implement FaissBruteForce
########################################

class FaissBruteForce:
    """
    A brute-force FAISS index for storing embeddings and their associated metadata,
    supporting Euclidean, Cosine, and Dot Product distance measures.
  
    Attributes:
        dim (int): The dimensionality of the embeddings.
        metadata (list): A list to store metadata corresponding to each embedding.
        metric (str): The distance metric to use: 'euclidean', 'cosine', or 'dot_product'.
        index (faiss.IndexFlat): A FAISS flat index initialized based on the specified metric.
    """

    def __init__(self, dim, metric='euclidean'):
        """
        Initializes the FaissBruteForce index.

        Parameters:
            dim (int): The dimensionality of the embeddings.
            metric (str): Distance metric to use. Options are 'euclidean', 'cosine', or 'dot_product'.
        """
        self.dim = int(dim)
        self.metric = str(metric).lower()
        self.metadata = [] # keeps track of whatever info the user attaches to each embedding
        
        # Pick the right FAISS index based on the distance metric
        if self.metric == "euclidean":
            # L2 = straight-line distance between two points
            self.index = faiss.IndexFlatL2(self.dim)
        elif self.metric in {"cosine", "dot_product"}:
            # Both cosine and dot product use inner product under the hood
            # For cosine, we'll normalize the vectors before adding them
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            raise ValueError("Unsupported metric. Use 'euclidean', 'cosine', or 'dot_product'.")

    def _as_float32_matrix(self, embeddings):
        # FAISS only works with float32, so make sure we're in that format
        arr = np.asarray(embeddings, dtype=np.float32)
        # If someone passed a single embedding (1D), wrap it into a 2D matrix
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _maybe_normalize(self, vectors):
        # Normalization only matters for cosine similarity
        # Dot product on unit vectors is the same as cosine similarity
        if self.metric != "cosine":
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid dividing by zero for any zero vectors
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def add_embeddings(self, embeddings, metadata):
        """
        Adds new embeddings and their associated metadata to the index.
      
        Parameters:
            embeddings (list or np.ndarray): A list of embeddings, where each embedding is an array-like of length `dim`.
            metadata (list): A list of metadata corresponding to each embedding.
      
        Raises:
            ValueError: If an embedding does not match the specified dimensionality.
            ValueError: If the number of embeddings and metadata entries do not match.
        """
        # Every embedding needs exactly one metadata entry
        if len(embeddings) != len(metadata):
            raise ValueError("The number of embeddings and metadata entries must match.")

        vectors = self._as_float32_matrix(embeddings)

        # Double-check the embeddings are the right size for this index
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dim}, got {vectors.shape[1]}")

        # Normalize if we're doing cosine similarity
        vectors = self._maybe_normalize(vectors)

        # Add to the FAISS index and store the corresponding metadata
        self.index.add(vectors)
        self.metadata.extend(list(metadata))

    def get_metadata(self, idx):
        """
        Retrieves the metadata associated with a particular embedding index.
      
        Parameters:
           idx (int): The index of the embedding.
      
        Returns:
           The metadata associated with the embedding.
      
        Raises:
           IndexError: If the index is out of range.
        """
        # Make sure the index actually exists before trying to grab it
        if idx < 0 or idx >= len(self.metadata):
            raise IndexError("Metadata index out of range.")
        return self.metadata[idx]

    def save(self, filepath):
        """
        Saves the current FaissBruteForce instance to a file.
      
        Parameters:
           filepath (str): The path to the file where the instance should be saved.
        """
        # pickle the whole object so we can reload it later exactly as it is
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Loads a FaissBruteForce instance from a file.
      
        Parameters:
           filepath (str): The path to the file from which to load the instance.
      
        Returns:
           An instance of FaissBruteForce loaded from the file.
        """
        # unpickle and hand back a fully restored FaissBruteForce instance
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
        return instance

if __name__ == "__main__":
    # Choose the metric: 'euclidean', 'cosine', or 'dot_product'
    metric = 'cosine'
    index = FaissBruteForce(dim=4, metric=metric)

    # Create some dummy embeddings and corresponding metadata.
    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ]
    identity_metadata = [
        "Alice",
        "Bob",
        "Charlie"
    ]

    # Add the embeddings and metadata to the index.
    index.add_embeddings(embeddings, identity_metadata)

    # Define a query vector.
    query = [0.1, 0.2, 0.3, 0.4]
    k = 2  # number of nearest neighbors to retrieve

    # Perform the search using our class method.
    distances, indices = index.search(query, k)
    meta_results = [index.get_metadata(int(i)) for i in indices[0]]
    
    print("Query Vector:", query)
    print("Distances:", distances)
    print("Indices:", indices)
    print("Metadata Results:", meta_results)

    # Save the index to disk.
    filepath = "faiss_bruteforce_index.pkl"
    index.save(filepath)
    print(f"Index saved to {filepath}.")

    # Load the index from disk.
    loaded_index = FaissBruteForce.load(filepath)
    print("Loaded Metadata for index 0:", loaded_index.get_metadata(0))
