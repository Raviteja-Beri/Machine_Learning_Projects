import streamlit as st
import pickle
import numpy as np

# Load the saved model.  Use a raw string to handle the Windows path.
model = pickle.load(open(r"C:\Users\HP\VS Code Projects\Machine Learning\Simple Linear Regression\House Price Prediction\House_Price_Prediction_SLR.pkl", "rb"))

# Set the title of the Streamlit app
st.title("House Price Prediction")

# Add a brief description
st.write("This app predicts the price of a house based on its square footage.")

# Add input widget for user to enter square footage
square_footage = st.number_input(
    "Enter the square footage of the house:",
    min_value=0.0,  # Added a minimum value
    value=1000.0, # Added a default value
    step=100.0, # Added a step value
    format="%.0f" #Added format to remove decimal
)

# When the button is clicked, make predictions
if st.button("Predict Price"):
    # Make a prediction using the trained model
    # Convert the input to a 2D array for prediction
    sqft_input = np.array([[square_footage]])
    prediction = model.predict(sqft_input)

    # Display the result
    st.success(
        f"The predicted price for a house with {square_footage:.0f} square feet is: ${prediction[0]:,.2f}"
    )

# Display information about the model
st.write(
    "The model was trained using a dataset of house prices and their corresponding square footage."
)
