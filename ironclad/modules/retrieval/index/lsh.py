import numpy as np
import pickle
import faiss

########################################
# TASK: Implement FaissLSH
########################################

class FaissLSH:
    """
    An LSH-based FAISS index for storing embeddings and their associated metadata.
  
    This class uses Locality-Sensitive Hashing (LSH) to build an approximate nearest neighbor index.
    LSH works by hashing input vectors into buckets such that similar vectors are likely to hash to the same bucket.
    This reduces the search space significantly compared to brute-force search, making it especially effective for large datasets.
    However, note that because the method is approximate, the search results might not always be perfectly accurate.
  
    Attributes:
        dim (int): The dimensionality of the embeddings.
        metadata (list): A list to store metadata corresponding to each embedding.
        index (faiss.IndexLSH): A FAISS LSH index initialized with the specified embedding dimension and number of bits.
    """

    def __init__(self, dim, **kwargs):
        """
        Initializes the FaissLSH index.
      
        This method sets up internal storage for embeddings and metadata and creates a FAISS LSH index.
        The LSH algorithm reduces the search space by hashing vectors into buckets using random projections.
        Similar vectors are likely to hash to the same bucket, so searching within buckets is more efficient than
        comparing every vector in the dataset.
      
        Parameters:
            dim (int): The dimensionality of the embeddings.
            **kwargs: Optional keyword arguments to configure the LSH index. Recognized keys include:
                        - nbits (int): The number of bits to use for hashing in the LSH index.
                        A higher number of bits generally provides finer-grained buckets, which may improve accuracy
                        but at the cost of increased memory usage and slower search times. Default is 128.
        """
        self.dim = int(dim)
        # more bits = finer hash buckets = better accuracy, but more memory and slower search
        self.nbits = int(kwargs.get("nbits", 128))
        self.metadata = []

        # Build the LSH index -> it uses random projections to turn vectors into binary hash codes
        self.index = faiss.IndexLSH(self.dim, self.nbits)

    def _as_float32_matrix(self, embeddings):
        # FAISS only works with float32, so make sure we're in that format
        arr = np.asarray(embeddings, dtype=np.float32)
        # If someone passed a single embedding (1D), wrap it into a 2D matrix
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def add_embeddings(self, embeddings, metadata):
        """
        Adds new embeddings and their associated metadata to the index.
      
        The embeddings are added both to the FAISS LSH index and the internal metadata list.
        Since LSH relies on hashing, each embedding is projected into a binary hash code;
        during a search, the index quickly narrows down potential candidates based on hash collisions.
      
        Parameters:
            embeddings (list or np.ndarray): A list of embeddings, where each embedding is an array-like
                of length `dim`.
            metadata (list): A list of metadata corresponding to each embedding.
      
        Raises:
            ValueError: If an embedding does not match the specified dimensionality.
            ValueError: If the lengths of embeddings and metadata do not match.
        """
        # Every embedding needs exactly one metadata entry
        if len(embeddings) != len(metadata):
            raise ValueError("The number of embeddings and metadata entries must match.")

        vectors = self._as_float32_matrix(embeddings)

        # Double-check the embeddings are the right size for this index
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dim}, got {vectors.shape[1]}")

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
        Saves the current FaissLSH instance to a file.
      
        The instance is serialized using pickle. When loading back, both the index and associated metadata are restored.
      
        Parameters:
            filepath (str): The path to the file where the instance should be saved.
        """
        # pickle the whole object so we can reload it later exactly as it is
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Loads a FaissLSH instance from a file.
      
        This method deserializes the stored index and metadata, restoring the state of the object.
      
        Parameters:
            filepath (str): The path to the file from which to load the instance.
      
        Returns:
            An instance of FaissLSH loaded from the file.
        """
        # unpickle and hand back a fully restored FaissLSH instance
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
        return instance


if __name__ == "__main__":
    # Initialize a FaissLSH index with embedding dimension 4 using a custom nbits value.
    index = FaissLSH(dim=4, nbits=256)

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

    # Query the index with a vector.
    query = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    k = 2  # number of nearest neighbors to retrieve
    distances, indices = index.index.search(query, k)
    meta_results = [index.get_metadata(int(i)) for i in indices[0]]
    
    print("Query Vector:", query)
    print("Distances:", distances)
    print("Indices:", indices)
    print("Metadata Results:", meta_results)

    # Save the index to disk.
    filepath = "faiss_lsh_index.pkl"
    index.save(filepath)
    print(f"Index saved to {filepath}.")

    # Load the index from disk.
    loaded_index = FaissLSH.load(filepath)
    print("Loaded Metadata for index 0:", loaded_index.get_metadata(0))
