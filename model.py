from tensorflow.keras.models import model_from_json







class FacialExpressionModel(object):

    a = 'Not wearing Mask'
    b = 'Wearing mask'
                     
                     
                     

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        if self.preds > 0.5 :
            return FacialExpressionModel.a
        else :
            return FacialExpressionModel.b