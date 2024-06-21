from sagemaker.jumpstart.estimator import JumpStartEstimator


def jumpstart_finetune(model, preprocess_step_ret):

    model_id = model["model_id"]

    train_data_path = preprocess_step_ret["s3_finetune_dataset_path"]

    estimator = JumpStartEstimator(
        model_id=model_id,
        environment={"accept_eula": "true"},
        disable_output_compression=False)

    estimator.fit(inputs={"training": train_data_path})
    training_job_name = estimator.latest_training_job.name

    return {"training_job_name": training_job_name}
