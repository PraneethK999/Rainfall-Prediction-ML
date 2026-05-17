def predict_rain(model, sample_data):
    
    prediction = model.predict(sample_data)

    return prediction