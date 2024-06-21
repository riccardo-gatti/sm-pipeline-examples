import sagemaker
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.jumpstart.estimator import JumpStartEstimator

# TODO: Endpoint name is just for dev, remove for final version

def jumpstart_deploy(model, finetune_step_ret=None):

    model_id = model["model_id"]

    if finetune_step_ret is None:
        model = JumpStartModel(model_id=model_id)
        predictor = model.deploy(accept_eula=True)
        return {"model_endpoint": predictor.endpoint_name, "model_deployed": True, "is_finetuned_model": False}
    
    else:
        training_job_name = finetune_step_ret["training_job_name"]

        estimator = JumpStartEstimator.attach(training_job_name, model_id=model_id)
        estimator.logs()
        predictor = estimator.deploy(serializer=sagemaker.serializers.JSONSerializer(),
                                     deserializer=sagemaker.deserializers.JSONDeserializer())

        return {"model_endpoint": predictor.endpoint_name, "model_deployed": True, "is_finetuned_model": True, "training_job_name": training_job_name}

