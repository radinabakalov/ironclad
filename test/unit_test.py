import unittest
import numpy as np
import os
import tempfile
from ironclad.modules.retrieval.index.lsh import FaissLSH  # Adjust this import to your actual module path
from ironclad.modules.retrieval.index.hnsw import FaissHNSW


class TestFaissHNSW(unittest.TestCase):
    def setUp(self):
        self.dim = 4
        self.embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]
        ]
        self.metadata = ["Alice", "Bob", "Charlie"]

    def test_initialization_euclidean(self):
        index = FaissHNSW(dim=self.dim, metric="euclidean", M=16, efConstruction=30)
        self.assertEqual(index.dim, self.dim)
        self.assertEqual(index.metric, "euclidean")
        self.assertEqual(index.m, 16)
        self.assertEqual(index.efConstruction, 30)

    def test_initialization_invalid_metric(self):
        with self.assertRaises(ValueError):
            FaissHNSW(dim=self.dim, metric="invalid_metric")

    def test_add_embeddings_and_get_metadata(self):
        index = FaissHNSW(dim=self.dim, metric="euclidean")
        index.add_embeddings(self.embeddings, self.metadata)

        # Check metadata
        for i in range(len(self.metadata)):
            self.assertEqual(index.get_metadata(i), self.metadata[i])

        # Check actual FAISS index size
        self.assertEqual(index.index.ntotal, 3)

    def test_add_embeddings_dimension_mismatch(self):
        index = FaissHNSW(dim=self.dim)
        bad_embeddings = [[0.1, 0.2, 0.3]]  # dimension = 3

        with self.assertRaises(ValueError):
            index.add_embeddings(bad_embeddings, ["BadDim"])

    def test_add_embeddings_count_mismatch(self):
        index = FaissHNSW(dim=self.dim)
        with self.assertRaises(ValueError):
            index.add_embeddings(self.embeddings, self.metadata[:2])  # length mismatch

    def test_get_metadata_index_out_of_bounds(self):
        index = FaissHNSW(dim=self.dim)
        index.add_embeddings(self.embeddings, self.metadata)
        with self.assertRaises(IndexError):
            index.get_metadata(-1)
        with self.assertRaises(IndexError):
            index.get_metadata(10)

    def test_save_and_load(self):
        index = FaissHNSW(dim=self.dim)
        index.add_embeddings(self.embeddings, self.metadata)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_hnsw.pkl")
            index.save(filepath)

            # Load and check
            loaded_index = FaissHNSW.load(filepath)
            self.assertEqual(loaded_index.dim, self.dim)
            self.assertEqual(loaded_index.metadata, self.metadata)
            self.assertEqual(loaded_index.index.ntotal, 3)

    def test_cosine_normalization(self):
        index = FaissHNSW(dim=self.dim, metric="cosine")
        index.add_embeddings(self.embeddings, self.metadata)

        # Cosine similarity requires normalized vectors
        vectors = np.array(self.embeddings).astype(np.float32)
        faiss_vectors = np.empty_like(vectors)
        for i, vec in enumerate(vectors):
            norm = np.linalg.norm(vec)
            faiss_vectors[i] = vec / norm if norm > 0 else vec

        query = faiss_vectors[0].reshape(1, -1)
        distances, indices = index.index.search(query, k=1)
        self.assertEqual(indices[0][0], 0)



class TestFaissLSH(unittest.TestCase):
    def setUp(self):
        self.dim = 4
        self.embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]
        ]
        self.metadata = ["Alice", "Bob", "Charlie"]

    def test_initialization(self):
        index = FaissLSH(dim=self.dim, nbits=256)
        self.assertEqual(index.dim, self.dim)
        self.assertEqual(index.nbits, 256)
        self.assertEqual(index.index.d, self.dim)

    def test_add_embeddings_and_metadata(self):
        index = FaissLSH(dim=self.dim)
        index.add_embeddings(self.embeddings, self.metadata)
        self.assertEqual(index.index.ntotal, 3)
        self.assertEqual(index.metadata, self.metadata)

    def test_add_embeddings_dimension_mismatch(self):
        index = FaissLSH(dim=self.dim)
        bad_embeddings = [[0.1, 0.2, 0.3]]  # dim != 4
        with self.assertRaises(ValueError):
            index.add_embeddings(bad_embeddings, ["BadDim"])

    def test_add_embeddings_count_mismatch(self):
        index = FaissLSH(dim=self.dim)
        with self.assertRaises(ValueError):
            index.add_embeddings(self.embeddings, self.metadata[:2])

    def test_get_metadata_valid_and_invalid(self):
        index = FaissLSH(dim=self.dim)
        index.add_embeddings(self.embeddings, self.metadata)
        self.assertEqual(index.get_metadata(1), "Bob")

        with self.assertRaises(IndexError):
            index.get_metadata(-1)
        with self.assertRaises(IndexError):
            index.get_metadata(10)

    def test_save_and_load(self):
        index = FaissLSH(dim=self.dim)
        index.add_embeddings(self.embeddings, self.metadata)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "lsh_test.pkl")
            index.save(filepath)

            loaded = FaissLSH.load(filepath)
            self.assertEqual(loaded.dim, self.dim)
            self.assertEqual(loaded.metadata, self.metadata)
            self.assertEqual(loaded.index.ntotal, 3)

    def test_index_search_returns_results(self):
        index = FaissLSH(dim=self.dim)
        index.add_embeddings(self.embeddings, self.metadata)

        query = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        distances, indices = index.index.search(query, k=2)
        self.assertEqual(distances.shape, (1, 2))
        self.assertEqual(indices.shape, (1, 2))
        self.assertTrue(all(0 <= i < len(self.metadata) for i in indices[0]))


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

class TestFlaskAppAdd(BaseFlaskTestCase):

    @patch("ironclad.app.model.encode", return_value=np.random.rand(512).astype(np.float32))
    @patch("ironclad.app.preprocessor.process", return_value=torch.rand(3, 224, 224))
    def test_add_success(self, mock_process, mock_encode):
        name = "Test_User"

        # Remove if name already exists
        if name in index.metadata:
            index.metadata.remove(name)
            index.index.reset()
            for i, meta in enumerate(index.metadata):
                vec = np.random.rand(index.dim).astype(np.float32).reshape(1, -1)
                index.index.add(vec)

        initial_size = index.index.ntotal

        response = self.client.post(
            '/add',
            content_type='multipart/form-data',
            data={
                'image': (self.img_bytes, 'test.jpg'),
                'name': name
            }
        )

        json_data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", json_data)
        self.assertIn(name, index.metadata)
        self.assertEqual(index.index.ntotal, initial_size + 1, "FAISS index did not grow after adding image")

    def test_add_missing_image(self):
        response = self.client.post('/add', data={'name': 'Test'})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Error", response.get_json())

    def test_add_missing_name(self):
        response = self.client.post(
            '/add',
            content_type='multipart/form-data',
            data={'image': (self.img_bytes, 'test.jpg')}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Error", response.get_json())

    @patch("ironclad.app.model.encode", return_value=np.random.rand(512).astype(np.float32))
    @patch("ironclad.app.preprocessor.process", return_value=torch.rand(3, 224, 224))
    def test_add_duplicate_name(self, mock_process, mock_encode):
        name = "Duplicate_User"
        if name not in index.metadata:
            index.metadata.append(name)

        response = self.client.post(
            '/add',
            content_type='multipart/form-data',
            data={
                'image': (self.img_bytes, 'test.jpg'),
                'name': name
            }
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Error", response.get_json())

if __name__ == '__main__':
    unittest.main()
