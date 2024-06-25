import pandas as pd
import mlflow

def get_run_logs(run_id, history_as_df = False):
    """Requires a run_id from mlflow

    requires something like this:
    with mlflow.start_run() as run:
        run_id = run.info.run_id

    then, returns:
    - params: logged parameters of the run
    - metrics: metrics at the end of the run
    - metric_history: a history of values

    optionally, if history_as_df=True, returns:
    - params: logged parameters of the run
    - metrics: metrics at the end of the run
    - df_metric_history: a history of values, as a data frame
    """
    client = mlflow.tracking.MlflowClient()
    params = client.get_run(run_id).data.params
    metrics = client.get_run(run_id).data.metrics
    metric_history = {k: client.get_metric_history(run_id, k) for k, _ in metrics.items()}
    if history_as_df:
        df_metric_history = pd.DataFrame()
        for k, lsv in metric_history.items():
            col = [v.value for v in lsv]
            df_metric_history[k] = col
        return params, metrics, df_metric_history
    return params, metrics, metric_history