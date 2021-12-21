import os

import hydra
import mlflow


# ref. https://ymym3412.hatenablog.com/entry/2020/02/09/034644
class MlflowWriter():
    def __init__(self, model_name):
        # Set path to mlruns directory
        mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
        self.client = mlflow.tracking.MlflowClient()
        try:
            self.experiment_id = self.client.create_experiment(model_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(model_name).experiment_id

        self.run_id = self.client.create_run(self.experiment_id).info.run_id
        self.client.set_tag(self.run_id, "run_id", self.run_id)

        with mlflow.start_run(self.run_id):
            self.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value, step=None):
        self.client.log_metric(self.run_id, key, value, step=step)

    def set_tag(self, key, value):
        self.client.set_tag(self.run_id, key, value)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)
