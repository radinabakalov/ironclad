import numpy as np
import pickle
import faiss

########################################
# TASK: Implement FaissHNSW
########################################

class FaissHNSW:
    """
    A FAISS HNSW index for storing embeddings and their associated metadata using
    the Hierarchical Navigable Small World (HNSW) algorithm.


    This class leverages FAISS's HNSW index for similarity search in large datasets.
    HNSW indexes provide fast approximate nearest neighbor search without the need for a
    separate training step. The index directly adds embeddings, partitions them into a graph-based
    structure, and uses this structure to perform rapid similarity queries.


    Attributes:
        dim (int): The dimensionality of the embeddings.
        metadata (list): A list storing metadata associated with each embedding.
        index (faiss.IndexHNSWFlat): The underlying FAISS HNSW index object.
    """

    def __init__(self, dim, **kwargs):
        """
        Initializes the FaissHNSW instance.


        This method creates a FAISS HNSW index for flat (L2) search. It accepts keyword arguments
        to configure parameters of the HNSW algorithm, including:
            - M (int): The number of neighbors in the HNSW graph. A higher value leads to a more connected graph
                    and can improve search quality at the expense of increased memory usage. Default is 32.
            - efConstruction (int): The size of the dynamic candidate list during the index construction phase.
                                 Larger values improve index quality but increase construction time. Default is 40.


        After initialization, the index is ready to accept embeddings without a training step.


        Parameters:
            dim (int): The dimensionality of the embeddings.
            **kwargs: Optional keyword arguments to configure the HNSW index. Recognized keys include 'M' and 'efConstruction'.
        """
        self.dim = int(dim)

        # Default to euclidean; cosine is also supported
        metric = kwargs.get("metric", "euclidean")
        self.metric = str(metric).lower()

        if self.metric not in {"euclidean", "cosine"}:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")

        # M controls graph connectivity -> more connections = better recall, more memory
        self.m = int(kwargs.get("M", 32))
        # efConstruction controls build quality -> higher = slower build but better graph
        self.efConstruction = int(kwargs.get("efConstruction", 40))

        # Build the actual FAISS index depending on the chosen metric
        if self.metric == "euclidean":
            self.index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_L2)
        else:
            # Cosine similarity = inner product on unit-normalized vectors
            self.index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_INNER_PRODUCT)

        # Apply the efConstruction setting to the underlying HNSW graph
        self.index.hnsw.efConstruction = self.efConstruction
        self.metadata = []

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

    def add_embeddings(self, new_embeddings, new_metadata):
        """
        Adds new embeddings and their associated metadata to the HNSW index.


        This method processes each new embedding by converting it into a NumPy array and verifying
        its dimensionality. If an embedding's dimension does not match the expected dimension, a
        ValueError is raised. The corresponding metadata for each embedding is stored in an internal
        list. The embeddings are reshaped to a two-dimensional array before being added to the index.


        Parameters:
            new_embeddings (list or np.ndarray): A list or array of new embeddings to be added.
                Each embedding should be an array-like object with a length equal to `dim`.
            new_metadata (list): A list of metadata items corresponding to each embedding.


        Raises:
            ValueError: If the number of embeddings does not match the number of metadata entries.
            ValueError: If any individual embedding does not match the specified dimensionality.
        """
        # Every embedding needs exactly one metadata entry
        if len(new_embeddings) != len(new_metadata):
            raise ValueError("The number of embeddings and metadata entries must match.")

        vectors = self._as_float32_matrix(new_embeddings)

        # Double-check the embeddings are the right size for this index
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dim}, got {vectors.shape[1]}")

        # Normalize if we're doing cosine similarity
        vectors = self._maybe_normalize(vectors)

        # Add vectors to the HNSW graph and store their metadata side by side
        self.index.add(vectors)
        self.metadata.extend(list(new_metadata))

    def get_metadata(self, idx):
        """
        Retrieves the metadata associated with a specific embedding based on its index.


        The given index corresponds to the position in the metadata list where the metadata was stored
        when the embedding was added. This method validates the index and raises an IndexError if it is out of bounds.


        Parameters:
            idx (int): The integer index of the embedding for which metadata is being requested.


        Returns:
            The metadata object associated with the specified embedding.


        Raises:
            IndexError: If the provided index is negative or exceeds the number of stored embeddings.
        """
        # Make sure the index actually exists before trying to grab it
        if idx < 0 or idx >= len(self.metadata):
            raise IndexError("Metadata index out of range.")
        return self.metadata[idx]

    def save(self, filepath):
        """
        Serializes and saves the FaissHNSW instance to a file using pickle.


        This method saves the entire state of the instance, including the FAISS index, metadata,
        and configuration parameters (dim, M, and efConstruction). The saved file can later be loaded
        to restore the index for future use without needing to reinitialize or rebuild.


        Parameters:
            filepath (str): The file path where the serialized instance will be saved.
        """
        # pickle the whole object so we can reload it later exactly as it is
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Loads a serialized FaissHNSW instance from a file.


        This class method reads the pickle file at the given file path and returns a FaissHNSW
        instance with its state restored. This allows users to resume operations on a previously
        saved index.


        Parameters:
            filepath (str): The file path from which the instance will be loaded.


        Returns:
            An instance of FaissHNSW with the state restored from the specified file.
        """
        # unpickle and hand back a fully restored FaissHNSW instance
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
        return instance


if __name__ == "__main__":

    # Initialize a FaissHNSW index with embedding dimension 4 using custom HNSW parameters.
    index = FaissHNSW(dim=4, M=16, efConstruction=50)

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

    # Let's search the index with a query vector.
    query = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    k = 2  # number of nearest neighbors to retrieve

    # The FAISS search directly returns distances and indices.
    distances, indices = index.index.search(query, k)
    meta_results = [index.get_metadata(int(i)) for i in indices[0]]

    print("Query Vector:", query)
    print("Distances:", distances)
    print("Indices:", indices)
    print("Metadata Results:", meta_results)

    # Save the index to disk.
    filepath = "faiss_hnsw_index.pkl"
    index.save(filepath)
    print(f"Index saved to {filepath}.")

    # Load the index from disk.
    loaded_index = FaissHNSW.load(filepath)
    print("Loaded Metadata for index 0:", loaded_index.get_metadata(0))
