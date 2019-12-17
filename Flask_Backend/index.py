from flask import Flask, request, jsonify, send_from_directory
from tools import summarize_text, get_topics, segment
app = Flask(__name__, static_url_path='')

@app.route("/")
def hello():
    return "Hello World"

@app.route("/nlp", methods=["POST"])
def get_text():
	if not request.json or not "content" in request.json:
		abort(400)
	raw_text = request.json["content"]
	summary = summarize_text(raw_text)
	topics_whole = get_topics(raw_text)
	segmented_text = segment(raw_text)
	topics = [get_topics(segment) for segment in segmented_text if len(segment)>10]

	return jsonify({"summary":summary, "raw_topics":topics_whole, "segmented_text":segmented_text, "topics":topics})

@app.route("/lmao")
def hard():
	return send_from_directory("/","hard.html")
		

if __name__ == "__main__":
	app.run(host='0.0.0.0', debug=True)
