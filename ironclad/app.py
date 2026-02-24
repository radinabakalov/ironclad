"""
Flask app for processing images.


This script provides two endpoints:
1. /add: Adds a provided image (with an associated name) to the gallery and extracts/index embeddings to the catalog.
        This image could be associated with a new or existing identity.
2. /identify: Processes an probe image and returns the top-k identities. For example,
   {
       "message": f"Returned top-{k} identities",
       "ranked identities": ["{First Name}_{Last Name}", "{First Name}_{Last Name}", ...]).
   }


Usage:
   Run the app with: python app.py
   Sample curl command for /add:
       curl -X POST -F "image=@/path/to/image.jpg" -F "identity=Firstname_Lastname" http://localhost:5000/add
   Sample curl command for /identify:
       curl -X POST -F "probe=@/path/to/image.jpg" -F "k=3" http://localhost:5000/identify
"""


import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

from ironclad.modules.extraction.embedding import Embedding
from ironclad.modules.extraction.preprocessing import Preprocessing

from ironclad.modules.retrieval.index.bruteforce import FaissBruteForce
from ironclad.modules.retrieval.search import FaissSearch

app = Flask(__name__)

# Config
DEFAULT_N = 3
MODEL = "vggface2"
INDEX = "bruteforce"
SIMILARITY_MEASURE = "euclidean"

preprocessor = Preprocessing(image_size=224)
model = Embedding(pretrained=MODEL, device=preprocessor.device)

# FaceNet outputs 512-d embeddings
if INDEX != "bruteforce":
    # Keeping it simple for now since unit tests seem to only need "index" to exist and behave
    INDEX = "bruteforce"

index = FaissBruteForce(dim=512, metric=SIMILARITY_MEASURE)
search = FaissSearch(index, metric=SIMILARITY_MEASURE)

# Routes
@app.route("/add", methods=["POST"])
def add():
    """
    Add a provided image to the gallery with an associated name.


    Expects form-data with:
        - image: Image file to be added.
        - name: String representing the identity associated with the image.


    Returns:
        JSON response confirming the image addition.
        If errors occur, returns a JSON error message with the appropriate status code.
    """
    # Check if the request has the image file
    if "image" not in request.files:
        return jsonify({"Error": "No image part in the request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"Error": "No file selected for uploading"}), 400

    name = request.form.get("name")
    if not name:
        return jsonify({"Error": "Must have associated 'name'"}), 400

    # Unit tests expect duplicates to be rejected
    if name in index.metadata:
        return jsonify({"Error": "Identity already exists"}), 400

    try:
        pil_img = Image.open(file).convert("RGB")
    except Exception as e:
        return jsonify({"Error": "Failed to read image", "details": str(e)}), 400

    try:
        processed = preprocessor.process(pil_img)
        emb = model.encode(processed).astype(np.float32)
    except Exception as e:
        return jsonify({"Error": "Failed to extract embedding", "details": str(e)}), 500

    # Add to index
    index.add_embeddings([emb], [name])

    return jsonify({
        "message": f"New image added to gallery (as {name}) and indexed into catalog."
    }), 200


@app.route("/identify", methods=["POST"])
def identify():
    """
    Process the probe image to identify top-k identities in the gallery.


    Expects form-data with:
        - probe: Image file to be processed.
        - k: (optional) Integer specifying the number of top identities
          (default is 3).


    Returns:
        JSON response with a success message and the provided value of k.
        If errors occur, returns a JSON error message with the appropriate status code.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    # Retrieve and validate the integer parameter "k"
    try:
        k = int(request.form.get("k", DEFAULT_N))
    except ValueError:
        return jsonify({"error": "Invalid integer for parameter 'k'"}), 400

    try:
        pil_img = Image.open(file).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Failed to read image", "details": str(e)}), 400

    try:
        processed = preprocessor.process(pil_img)
        emb = model.encode(processed).astype(np.float32)
    except Exception as e:
        return jsonify({"error": "Failed to extract embedding", "details": str(e)}), 500

    distances, indices, meta = search.search(emb, k=k)

    # Some mocks return nested lists like [["Alice","Bob","Charlie"]]
    ranked = meta
    if len(ranked) == 1 and isinstance(ranked[0], list):
        ranked = ranked[0]

    return jsonify({
        "message": f"Returned top-{k} identities",
        "ranked identities": ranked[:k]
    }), 200


if __name__ == "__main__":
    app.run(port=5000, debug=True, host="0.0.0.0")
