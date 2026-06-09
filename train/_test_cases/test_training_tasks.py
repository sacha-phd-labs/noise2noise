import os
import subprocess

import pytest
import sys
from pathlib import Path
import mlflow

# Add the noise2noise package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train.trainer import Noise2NoiseTrainer

class TestTrainingTasks:

    @classmethod
    def setup_class(cls):
        cls.setup_mlflow()

    @classmethod
    def setup_mlflow(cls):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file://./mlruns"))
        if not mlflow.get_experiment_by_name("noise2noise_test_experiment"):
            mlflow.create_experiment("noise2noise_test_experiment")
        mlflow.set_experiment("noise2noise_test_experiment")

    @classmethod
    def teardown_class(cls):
        experiment = mlflow.get_experiment_by_name("noise2noise_test_experiment")
        if experiment:
            experiment_id = experiment.experiment_id
            mlflow.delete_experiment(experiment_id)
            subprocess.run(["mlflow", "gc", "--experiment-ids", experiment_id, "--backend-store-uri", "sqlite:////mlflow-sacha-phd-labs/mlflow.db"])

    @pytest.mark.parametrize("nn_domain, projection_consistency, supervised, reconstruction_type", [
        ('image', True, False, 'fbp'),
        ('image', False, False, 'fbp'),
        ('image', True, True, 'fbp'),
        ('image', False, True, 'fbp'),
        ('photon', False, False, 'fbp'),
        ('photon', False, True, 'fbp'),
        ('image', True, False, 'mlem'),
        ('image', False, False, 'mlem'),
        ('image', True, True, 'mlem'),
        ('image', False, True, 'mlem'),
        ('photon', False, False, 'mlem'),
        ('photon', False, True, 'mlem'),
    ])
    def test_training_tasks(self, nn_domain, projection_consistency, supervised, reconstruction_type):
        
        with mlflow.start_run(run_name=f"test_{nn_domain}_{projection_consistency}_{supervised}_{reconstruction_type}"):
            trainer = Noise2NoiseTrainer(
                batch_size=2,
                n_epochs=1,
                dataset_train_size=4,
                dataset_val_size=4,
                simulator_config={
                    'acquisition_time': 100.0,
                },
                nn_domain=nn_domain,
                projection_consistency=projection_consistency,
                supervised=supervised,
                reconstruction_type=reconstruction_type,
                register_model=False,
            )
            trainer.fit()

if __name__ == "__main__":
    TestTrainingTasks().test_training_tasks(nn_domain='photon', projection_consistency=False, supervised=False, reconstruction_type='mlem')