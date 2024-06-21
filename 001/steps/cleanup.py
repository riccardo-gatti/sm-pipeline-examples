import boto3


def cleanup(register_ret, *deploy_rets):

    for deploy_ret in deploy_rets:
        client = boto3.client('sagemaker')
        client.delete_endpoint(EndpointName=deploy_ret["model_endpoint"])
        client.delete_endpoint_config(EndpointConfigName=deploy_ret["model_endpoint"])

    return {"cleanup_done": True}


