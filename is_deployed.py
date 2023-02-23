import requests


headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

json_data = {
    'age': 39,
    'workclass': 'State-gov',
    'fnlght': 77516,
    'education': 'Bachelors',
    'education-num': 13,
    'marital-status': 'Never-married',
    'occupation': 'Adm-clerical',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Male',
    'capital-gain': 2174,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 'United-States',
}

response = requests.post('https://is-high-earner.onrender.com/predict/', headers=headers, json=json_data)

if response:
    print("yes!!")
else:
    print("no response... :(")