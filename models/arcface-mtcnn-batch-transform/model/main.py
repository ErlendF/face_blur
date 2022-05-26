from sagemaker_inference import model_server

if __name__ == "__main__":
    model_server.start_model_server(handler_service="/app/model/handler.py:handle")

