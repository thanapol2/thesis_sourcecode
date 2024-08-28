import numpy as np

def update_array(x: np.ndarray, y):
    # Check if the array is empty
    if x.size == 0:
        return x  # If empty, nothing to update

    # Pop the oldest element (leftmost) from the deque
    x = x[1:]

    # Append the new value y to the deque
    x = np.append(x, y)

    return x


## For evaluation OnlineSLE
def result_aggregation(dataset_name, algo_name, input_results):
    # Define error bounds for evaluating accuracy: 0% (exact match) and 20% (within 20% of the correct answer).
    error_bounds = [0, 20]

    # Initialize a list to store the output results.
    output_results = []

    for error_bound in error_bounds:
        count_result = 0
        for row in input_results:
            # If the error bound is 0, set both lower and upper bounds to the exact answer.
            if error_bound == 0:
                lb = row['answer']  # Lower bound is the exact answer.
                ub = row['answer']  # Upper bound is also the exact answer.
            else:
                # For error bounds other than 0, calculate lower and upper bounds based on the error percentage.
                lb = row['answer'] * ((100 - error_bound) / 100)  # Lower bound adjusted by error percentage.
                ub = row['answer'] * ((100 + error_bound) / 100)  # Upper bound adjusted by error percentage.

            pred_result = row['result']
            if (lb <= pred_result) & (pred_result <= ub):
                count_result = count_result + 1

        residual_rate = row['residual_rate']
        # Calculate the accuracy ratio as the number of results within bounds divided by the total number of results.
        accuracy_ratio = count_result / len(input_results)

        # Append the results for the current error bound to the output list.
        output_results.append({
            'dataset_name': dataset_name,
            'algorithms': algo_name,
            'residual_rate': residual_rate,
            'error_bound': error_bound,
            'accuracy_ratio': accuracy_ratio
        })

    # Return the compiled output results.
    return output_results