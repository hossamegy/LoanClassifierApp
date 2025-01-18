import os
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from models import LoanData
from controllers import LoanClassifier, ProjectController



# Create a router for inference
inference_router = APIRouter()

projectController = ProjectController()

# Define paths to model, encoder, and quantile transformer
model_path = os.path.join(projectController.base_path, 'loan_models', 'Random_Forest_model.pkl')
encoder_path = os.path.join(projectController.base_path,'loan_models', 'encoder_scaler', 'encoder.pkl')
quantile_transformer_path = os.path.join(projectController.base_path,'loan_models', 'encoder_scaler', 'quantile_transformer.pkl')

# Initialize LoanClassifier once, at server startup
loan_classifier = LoanClassifier(model_path, encoder_path, quantile_transformer_path)

@inference_router.post('/predict')
def loan_predictions(input_data: LoanData):
    try:
        prediction = loan_classifier.predict(input_data)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={'message': prediction}
        )
    
    except Exception as e:
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'message': f"An error occurred: {str(e)}"}
        )