from flaskblog.model.ner_model import NERModel
import json
import nltk

# will use this function to make predictions.

def get_model_api():
    with open('flaskblog/model/cfg.json','r') as f:
        cfg = json.load(f)
    model = NERModel(cfg)

    def model_api(input_data):
        return model.full_predict(input_data)

    return model_api
