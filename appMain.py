from flask import Flask, request, jsonify
import bertClassifier

app = Flask(__name__)
classifier = bertClassifier.BertClassifier()


@app.route('/')
def hello_world():
    return 'Hello, I am Main!'


@app.route('/check', methods=['POST'])
def check_correctness():
    sents = request.json.get('sentence')
    data = [classifier.classify(sent) for sent in sents]
    return jsonify(isError=False,
                   message="Success",
                   statusCode=200,
                   data=data), 200


if __name__ == '__main__':
    app.run("0.0.0.0")
