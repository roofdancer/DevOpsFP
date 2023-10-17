from flask import Flask, request, jsonify
from com.fallback.nlp import simpleClassifier

app = Flask(__name__)
classifier = simpleClassifier.SimpleClassifier()


@app.route('/')
def hello_world():
    return 'Hello, I am Fallback!'


@app.route('/check', methods=['POST'])
def check_correctness():
    sents = request.json.get('sentence')
    print(sents)
    # data = [simpleClassifier.check_correctness(sent) for sent in sents]
    data = [classifier.classify(sent) for sent in sents]
    return jsonify(isError=False,
                   message="Success",
                   statusCode=200,
                   data=data), 200


if __name__ == '__main__':
    app.run("0.0.0.0")
