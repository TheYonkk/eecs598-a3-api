from flask import Flask, redirect, url_for, request, jsonify
from flask_cors import CORS, cross_origin

from transformers import pipeline

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

vqa_pipeline = pipeline("visual-question-answering")

@app.route("/")
@cross_origin()
def index():
    return "<p>Send some data to the /ask endpoint via a post request!</p>"

# Path: flask/app.py
@app.route("/ask", methods=["POST"])
@cross_origin()
def ask():

    # get the data from the request
    data = request.get_json()
    question = data["question"]
    image_link = data["image_url"]

    # check for missing or empty data. return 400 if missing.
    if question is None or image_link is None:
        return "Missing question or image link", 400
    elif question == "" or image_link == "":
        return "Empty question or image link", 400

    # run the pipeline and return the answers
    ans = vqa_pipeline(image_link, question)
    return jsonify(ans)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, ssl_context='adhoc')
