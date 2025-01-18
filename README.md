# **Loan Prediction System**

This project provides a loan prediction system that uses a pre-trained machine learning model to predict whether a loan application will be accepted or rejected. The system is built using **FastAPI** for the backend, **Pydantic** for data validation, and **Bootstrap** for a simple user interface. The model and its dependencies (encoder, scaler) are loaded from pickle files.

## **Project Structure**

The project is structured as follows:


## **Features**

- **Loan Prediction:** Predict if a loan application is accepted or rejected.
- **Model Loading:** Loads the machine learning model, encoder, and scaler from pickle files.
- **API Endpoint:** Exposes a FastAPI POST endpoint (`/predict`) to accept loan data and return a prediction.
- **Frontend Interface:** A simple HTML form powered by Bootstrap to collect loan applicant data and display the result.

## **Requirements**

- Python 3.7+
- FastAPI
- Uvicorn (for running the FastAPI server)
- scikit-learn
- numpy
- pydantic
- pickle

Install dependencies via `pip`:
```bash
pip install -r requirements.txt
```

## **Backend**
### **LoanClassifier Class**

The LoanClassifier class is responsible for loading the model, encoder, and quantile transformer from pickle files, preprocessing input data, and making predictions.

   - load_pickle(path): A helper function that loads a pickle object (model/encoder/scaler).
   - preprocessing_data(input_data): Processes the input data by encoding categorical variables and scaling numerical values.
   - prediction(preprocessed_data): Uses the model to predict loan approval.
   - predict(input_data): A convenience method to preprocess the input data and return the loan status (Accepted or Rejected).

## **LoanData Model**

The LoanData class defines the data structure for the input data. It's a Pydantic model that validates and ensures proper data types for each field.


## **Frontend**

The HTML file contains a form for collecting loan application details. It makes a POST request to the FastAPI /predict endpoint and displays the result (loan approval or rejection).
Main Features:

   - Form for entering applicant data
   - Bootstrap styling
   - JavaScript fetch API for submitting the form and handling responses

## **How to Run the Project**
1. Train the Model

You can train your model using the dataset from the train_model/dataset directory. The model is trained and saved as a pickle file under loan_models/Random_Forest_model.pkl. Ensure that the encoder and quantile transformer are also saved for use in prediction.

2. Start the Backend

Run the FastAPI app with uvicorn:

```bash
uvicorn app.main:app --reload
```

This will start the FastAPI server locally at http://127.0.0.1:8000.

3. Open the Frontend

Open the app/views directory in your browser or host the HTML file on any web server. It will allow you to input data for loan prediction and show the result.

