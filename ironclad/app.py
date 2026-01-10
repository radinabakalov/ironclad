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

app = Flask(__name__)

## List of designed parameters: 
# (Configure these parameters according to your design decisions)
DEFAULT_N = '3'
MODEL = 'vggface2'
INDEX = 'bruteforce'
SIMILARITY_MEASURE = 'euclidean'
# Add more if needed...

@app.route("/add", methods=['POST'])
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
    if 'image' not in request.files:
        return jsonify({"Error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"Error": "No file selected for uploading"}), 400

    # Convert the image into a NumPy array
    try:
        image = np.array(Image.open(file))
        print(image)
    except Exception as e:
        return jsonify({
            "error": "Failed to convert image to numpy array",
            "details": str(e)
        }), 500

    # Retrieve the 'name' parameter
    identity = request.form.get('identity')
    if not identity:
        return jsonify({"Error": "Must have associated 'identity'"}), 400

    ########################################
    # TASK: Implement `/add` endpoint to
    #       add the provided image to the 
    #       catalog/gallery.
    ########################################

    return jsonify({
        "message": f"New image added to gallery (as {identity}) and indexed into catalog."
    })


@app.route('/identify', methods=['POST'])
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
    # Check if the request has the image file
    if 'probe' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['probe']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Retrieve and validate the integer parameter "k"
    try:
        k = int(request.form.get('k', DEFAULT_N))
    except ValueError:
        return jsonify({"error": "Invalid integer for parameter 'k'"}), 400

    # Convert the image into a NumPy array
    try:
        image = np.array(Image.open(file))
        print(image)
    except Exception as e:
        return jsonify({
            "error": "Failed to convert probe to numpy array",
            "details": str(e)
        }), 500

    ########################################
    # TASK: Implement /identify endpoint
    #         to return the top-k identities
    #         of the provided probe.
    ########################################

    # Example return statement. You'll need to change this!
    return jsonify({
        "message": f"Returned top-{k} identities",
        "ranked identities": [
                              "Firstname_Lastname", # Top 1 prediction
                              "Firstname_Lastname", # Top 2 prediction
                              "Firstname_Lastname", # Top 3 prediction
                              # ... more if needed, depending on `k`...
                             ]
    }), 200




if __name__ == '__main__':
    app.run(port=5000, debug=True, host='0.0.0.0')
