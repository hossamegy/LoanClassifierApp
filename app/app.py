from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from routers import inference_router
from fastapi.middleware.cors import CORSMiddleware

# FastAPI app
app = FastAPI()

# Allow all origins (use with caution in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get('/')
def get_welcome():
    return JSONResponse(
      status_code=status.HTTP_200_OK,
      content={
        'message': 'Welcome to the loan prediction API'
        }
      )
    

app.include_router(inference_router)


