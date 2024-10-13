from django.http import JsonResponse
from rest_framework.decorators import api_view
import joblib
import numpy as np

# Load the model
from tensorflow import keras

rf_model = joblib.load('MLApi/data/random_forest_model.pkl')
nn_model = keras.models.load_model('MLApi/data/neural_model.h5')



@api_view(['POST'])
def predict_profile(request):
    # Define required fields
    required_fields = [
        'profile_pic', 'nums_len', 'fullname_len', 'name_len','name_Eq_username'
        'description_len', 'external_URL', 'private',
        'posts', 'followers', 'follows'
    ]

    # Check for missing fields
    for field in required_fields:
        if field not in request.data:
            return JsonResponse({'error': f'Missing field: {field}'}, status=400)

    # Extract data from request
    data = request.data

    # Prepare the input data as a numpy array
    input_data = np.array([
        data['profile_pic'],
        data['nums_len'],
        data['fullname_len'],
        data['name_len'],
        data['name_Eq_username'],
        data['description_len'],
        data['external_URL'],
        data['private'],
        data['posts'],
        data['followers'],
        data['follows']
    ]).reshape(1, -1)  # Reshape for the model


    # Get predictions from both models
    rf_probabilities = rf_model.predict_proba(input_data)[0]
    nn_probabilities = nn_model.predict(input_data)[0]

    # Construct the response data
    response_data = {
        "RandomForest": {
            f"{int(rf_probabilities[1] * 100)}%": 1,  # Fake
            f"{int(rf_probabilities[0] * 100)}%": 0   # Genuine
        },
        "NeuralNetwork": {
            f"{int(nn_probabilities[0] * 100)}%": 1,  # Fake
            f"{int((1 - nn_probabilities[0]) * 100)}%": 0  # Genuine
        }
    }

    return JsonResponse(response_data)