def selection(*evaluate_ret_list):

    # Select best model.
    # We are evaluating Factual Knowledge thus we are looking for the maximum score

    max_score = -1
    best_model_name = None
    best_evaluation_output = None

    for evaluate_ret in evaluate_ret_list:
        print(evaluate_ret)
        model_name = evaluate_ret['model_name']

        eval_result = evaluate_ret['evaluation_output']
        eval_output = eval_result[0][0]
        eval_score = eval_output.dataset_scores[0].value
        print(eval_score)

        if eval_score > max_score:
            max_score = eval_score
            best_model_name = model_name
            best_evaluation_output = eval_result

    return {"evaluation_output": best_evaluation_output, "model_name": best_model_name}
            