import numpy as np

######################
# TASK: Implement FaissSearch
######################

class FaissSearch:
    def __init__(self, faiss_index, metric='euclidean', p=3):
        """
        Initialize the search class with a FaissIndex instance and distance metric.
        
        :param faiss_index: A FaissBruteForce instance.
        :param metric: The distance metric ('euclidean', 'dot_product', 'cosine', 'minkowski').
        :param p: The parameter for Minkowski distance (p=2 for Euclidean, p=1 for Manhattan, etc.).
        """
        self.index = faiss_index.index
        self.metric = metric.lower()
        self.p = p  # parameter for Minkowski distance
        self.faiss_index = faiss_index

    def search(self, query_vector, k=5):
        """
        Perform a nearest neighbor search and retrieve the associated metadata.
        
        :param query_vector: The vector to query (numpy array of shape (1, dim) or (dim,)).
        :param k: Number of nearest neighbors to return.
        :return: Tuple of (distances, indices, metadata) for the nearest neighbors.
        """
        query_vector = np.array(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if self.metric == 'euclidean':
            # Default FAISS search with Euclidean (L2) distance.
            pass

        elif self.metric == 'cosine':
            # For cosine similarity, normalize the query so that inner product corresponds to cosine.
            pass

        elif self.metric == 'dot_product':
            # For dot product, no normalization is applied (assuming the index is built appropriately).
            pass

        elif self.metric == 'minkowski':
            # For Minkowski, perform candidate selection via Euclidean search and then re-rank.
            # NOTE: Because FAISS does not support a minkowski metric, we will need to approximate the distance metric
            # by retrieving a smaller set of candidates... 
            candidate_k = max(50, k * 10)
            distances_candidate, indices_candidate = self.index.search(query_vector, candidate_k)
            # Reconstruct candidate vectors using the FAISS index.
            candidate_vectors = np.array([
                self.index.reconstruct(int(idx)) for idx in indices_candidate[0]
            ])
            # Compute Minkowski distances between the query and each candidate.
            minkowski_distances = self._compute_minkowski(query_vector[0], candidate_vectors, self.p)
            # Sort the candidates by the Minkowski distance.
            sorted_idx = np.argsort(minkowski_distances)[:k]
            selected_indices = indices_candidate[0][sorted_idx]
            distances = np.array(minkowski_distances)[sorted_idx].reshape(1, -1)
            indices = selected_indices.reshape(1, -1)

        else:
            raise ValueError("Unsupported metric. Use 'euclidean', 'cosine', 'dot_product', or 'minkowski'.")

        # Retrieve metadata for the nearest neighbors.
        metadata_results = [self.faiss_index.get_metadata(int(i)) for i in indices[0]]
        return distances, indices, metadata_results

    def _compute_minkowski(self, query_vector, nearest_vectors, p):
        """
        Compute Minkowski distance between the query vector and a set of candidate vectors.
        
        :param query_vector: The query vector (numpy array of shape (dim,)).
        :param nearest_vectors: Array of candidate vectors (numpy array of shape (n_candidates, dim)).
        :param p: The Minkowski distance parameter.
        :return: List of Minkowski distances.
        """
        # Compute elementwise differences, absolute value, power, sum over dimensions, and then take the p-th root.
        distances = np.sum(np.abs(nearest_vectors - query_vector) ** p, axis=1) ** (1 / p)
        return distances


if __name__ == "__main__":
    from index.bruteforce import FaissBruteForce  # assuming FaissBruteForce is implemented in index/bruteforce.py

    # Create some random vectors (for example, 10,000 vectors of dimension 256).
    vectors = np.random.random((10000, 256)).astype('float32')
    metadata = [f"Vector_{i}" for i in range(10000)]
    query_vector = np.random.random((1, 256)).astype('float32')
    
    # Construct the FaissBruteForce index with Euclidean measure.
    faiss_index_bf = FaissBruteForce(dim=256, metric="euclidean")
    faiss_index_bf.add_embeddings(vectors, metadata=metadata)

    print("\nExample: BruteForce Search with `euclidean` measure")
    search_euclidean = FaissSearch(faiss_index_bf, metric='euclidean')
    distances, indices, meta_results = search_euclidean.search(query_vector, k=5)
    for i in range(5):
        print(f"Nearest Neighbor {i+1}: Index {indices[0][i]}, Distance {distances[0][i]}, Metadata: {meta_results[i]}")

    print("\nExample: BruteForce Search with `cosine` measure")
    search_cosine = FaissSearch(faiss_index_bf, metric='cosine')
    distances, indices, meta_results = search_cosine.search(query_vector, k=5)
    for i in range(5):
        print(f"Nearest Neighbor {i+1}: Index {indices[0][i]}, Cosine Similarity {distances[0][i]}, Metadata: {meta_results[i]}")