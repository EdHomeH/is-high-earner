from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pickle


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = LogisticRegression(max_iter=1000, random_state=23)
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : LogisticRegression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    
    return model.predict(X)


def save_model(model, encoder, lb, path):
    """ Save the model as a pickel file

    Args:
        model : LogisticRegression
            Trained machine learning model
        encoder: 
            procesed encoder
        lb:
            label binarizer
        path : str
            Path to save the model
    """

    pickle.dump(model, open(path+'logistic.sav', 'wb'))
    pickle.dump(encoder, open(path+'encoder.sav', 'wb'))
    pickle.dump(lb, open(path+'lb.sav', 'wb'))


def load_model(path, return_encoder_and_lbl_binarizer=False):
    """ Load the model and the encoder if indicated from pickle files

    Args:
        path : str
            Path where the model and encoder are saved
        return_encoder : bool
            indicator to have the encoder also returned

    Returns:
        A LogisticRegression model | (LogisticRegression model, encoder)
    """
    if not return_encoder_and_lbl_binarizer:
        return pickle.load(open(path+'logistic.sav', 'rb'))
    return pickle.load(open(path+'logistic.sav', 'rb')), pickle.load(open(path+'encoder.sav', 'rb')), pickle.load(open(path+'lb.sav', 'rb'))
