# Model Card

## Model Details
Logistic Regression model using OneHotEncoder to encode categorical values and LabelBinarizer for the labels.

## Intended Use
This model is intended to be a reference to deploy a model that predicts if a user is a high earner.

## Training Data
The data used for this model is a subset of the census.csv provided by udacity.

## Evaluation Data
The data used for this model is a subset of the census.csv provided by udacity. 80% was used for training and 20% for testing.

## Metrics
precision:  0.7170731707317073

recall:  0.2780580075662043

fbeta:  0.40072694229895506

## Ethical Considerations
This model has not been tested for bias. It's recommended to do this before considering deploying this model. The use of a model without bias testing could be introducing ethical concerns, such as having a preference to weight a specific race over another to predict high earnings. 

## Caveats and Recommendations
This model is quite basic. Using a slightly more advanced model might bring better results. 
