import unittest
import numpy as np
import faiss

from ironclad.modules.retrieval.search import FaissSearch
from ironclad.modules.retrieval.index.bruteforce import FaissBruteForce


class TestFaissSearch(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility, do not change this...
        np.random.seed(42)

        # Create 10 random vectors of dimension 1000
        self.vectors = np.random.random((10, 1000)).astype('float32')
        self.metadata = [f"Vec_{i}" for i in range(10)]

        # Query vector also of dimension 1000
        self.query_vector = np.random.random((1, 1000)).astype('float32')

    def test_euclidean_search_returns_correct_index(self):
        # FAISS BruteForce index for Euclidean
        faiss_index_bf = FaissBruteForce(dim=1000, metric="euclidean")
        faiss_index_bf.add_embeddings(self.vectors, metadata=self.metadata)

        # FaissSearch with Euclidean
        searcher = FaissSearch(faiss_index_bf, metric='euclidean')
        distances, indices, meta = searcher.search(self.query_vector, k=1)

        # Ground-truth Euclidean distance
        dists = np.linalg.norm(self.vectors - self.query_vector[0], axis=1)
        expected_index = np.argmin(dists)

        np.testing.assert_array_equal(indices[0], [expected_index], err_msg="Euclidean index mismatch")

    def test_cosine_search_returns_correct_index(self):
        # Normalize vectors for cosine similarity
        vectors_norm = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
        query_norm = self.query_vector / np.linalg.norm(self.query_vector)

        # FAISS BruteForce index for Cosine
        faiss_index_bf = FaissBruteForce(dim=1000, metric="cosine")
        faiss_index_bf.add_embeddings(self.vectors, metadata=self.metadata)

        # FaissSearch with Cosine
        searcher = FaissSearch(faiss_index_bf, metric='cosine')
        distances, indices, meta = searcher.search(self.query_vector, k=1)

        

        # Ground-truth cosine similarity (argmax = most similar)
        sims = np.dot(vectors_norm, query_norm[0])
        expected_index = np.argmax(sims)

        np.testing.assert_array_equal(indices[0], [expected_index], err_msg="Cosine index mismatch")

    def test_dot_product_search_returns_correct_index(self):
        # FAISS BruteForce index for Dot Product
        faiss_index_bf = FaissBruteForce(dim=1000, metric="dot_product")
        faiss_index_bf.add_embeddings(self.vectors, metadata=self.metadata)

        # FaissSearch with Dot Product
        searcher = FaissSearch(faiss_index_bf, metric='dot_product')
        distances, indices, meta = searcher.search(self.query_vector, k=1)

        # Ground-truth dot product similarity
        dot_products = np.dot(self.vectors, self.query_vector[0])
        expected_index = np.argmax(dot_products)

        np.testing.assert_array_equal(indices[0], [expected_index], err_msg="Dot Product index mismatch")


import unittest
import io
import numpy as np
import torch
from unittest.mock import patch
from PIL import Image

torch.set_num_threads(1) # Only used for MacOS. See: https://github.com/apple/ml-stable-diffusion/issues/8

from ironclad.app import app, index  # Adjust path as necessary

class BaseFlaskTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

        # Create a dummy image in memory
        self.image = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
        self.img_bytes = io.BytesIO()
        self.image.save(self.img_bytes, format='JPEG')
        self.img_bytes.seek(0)


class TestFlaskAppIdentify(BaseFlaskTestCase):

    @patch("ironclad.app.model.encode", return_value=np.random.rand(512).astype(np.float32))
    @patch("ironclad.app.preprocessor.process", return_value=torch.rand(3, 224, 224))
    @patch("ironclad.app.search.search")
    def test_identify_success(self, mock_search, mock_process, mock_encode):
        mock_search.return_value = (
            np.array([[0.1, 0.2, 0.3]]),
            np.array([[0, 1, 2]]),
            [["Alice", "Bob", "Charlie"]]
        )

        response = self.client.post(
            '/identify',
            content_type='multipart/form-data',
            data={
                'image': (self.img_bytes, 'test.jpg'),
                'k': 3
            }
        )
        json_data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("ranked identities", json_data)
        self.assertEqual(len(json_data["ranked identities"]), 3)

    def test_identify_missing_image(self):
        response = self.client.post('/identify', data={'k': 3})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_identify_invalid_k(self):
        response = self.client.post(
            '/identify',
            content_type='multipart/form-data',
            data={
                'image': (self.img_bytes, 'test.jpg'),
                'k': 'not_a_number'
            }
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())


if __name__ == '__main__':
    unittest.main()
