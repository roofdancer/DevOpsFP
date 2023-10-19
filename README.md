# DevOpsFP

In this project I created two web apps that provide POST endpoint /check that obtains a JSON request with a 'sentence' array of Russian sentences and classifies them as grammatically correct (1) or incorrect (0). Services return a JSON array with 1s and 0s corresponding to each sentence in the request. An example request can be found in resources folder. For exampleRequest.json the main app returns [1, 0, 0] and the fallback app [1, 0, 1].

Main implementation uses a pre-trained mBERT model that was fine-tuned for this task. The model resides in model folder and is added to the app's docker as a tar-gz achive. The app will find the model in MODEL_PATH env variable.

Fallback implementation just checks whether all words in a sentence are "real" (i.e. correct) by attempting to parse a sentence with pymorphy2 library and analyzing the results.

The request is redirected to the fallback app in case of two error codes (401 and 503), this list can be extended in the ingress config file.
