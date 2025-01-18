import pickle
import numpy as np

class LoanClassifier:
    def __init__(self, model_path, encoder_path, quantile_transformer_path):
        self.model = self.load_pickle(model_path)
        self.encoder = self.load_pickle(encoder_path)
        self.quantile_transformer = self.load_pickle(quantile_transformer_path)
    
    def load_pickle(self, path):
        """Generalized function to load pickle objects."""
        with open(path, 'rb') as file:
            return pickle.load(file)
    
    def preprocessing_data(self, input_data):
        """Process input data by transforming categorical and numerical features."""
        categorical_data = np.array([
            input_data.person_gender,
            input_data.person_education,
            input_data.person_home_ownership,
            input_data.loan_intent,
            input_data.previous_loan_defaults_on_file
        ]).reshape(1, -1)

        numerical_data = np.array([
            input_data.person_age,
            input_data.person_income,
            input_data.person_emp_exp,
            input_data.loan_amnt,
            input_data.loan_int_rate,
            input_data.loan_percent_income,
            input_data.cb_person_cred_hist_length,
            input_data.credit_score
        ]).reshape(1, -1)

        # Apply transformations
        categorical_data = np.char.lower(categorical_data)
        encoded_data = self.encoder.transform(categorical_data)
        scaled_data = self.quantile_transformer.transform(numerical_data)

        return np.hstack([encoded_data, scaled_data])
    
    def prediction(self, preprocessed_data):
        """Make prediction using the preprocessed data."""
        return self.model.predict(preprocessed_data)
    
    def predict(self, input_data):
        """Process data and make prediction."""
        preprocessed_data = self.preprocessing_data(input_data)    
        if self.prediction(preprocessed_data).tolist()[0] == 1:
          return 'Accepted'
        else:  
          return 'Rejected'
