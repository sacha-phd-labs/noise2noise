import os
import mlflow
import datetime
import torchsummary
from train.trainer import Noise2NoiseTrainer
import torch

if __name__ == "__main__":

    unet_input_domain = 'photon'
    unet_output_domain = 'image'
    supervised = False

    trainer = Noise2NoiseTrainer(
            dest_path=f"{os.getenv('WORKSPACE')}/data/noise2noise",
            dataset_train_size=2048,
            dataset_val_size=2048,
            val_freq=1,
            n_epochs=15,
            batch_size=16,
            shuffle=True,
            simulator_config={
                'image_size' : (160,160),
                'voxel_size' : (2,2,2),
                'n_angles' : 300,
                'acquisition_time' : 109.16, #109.16 is a pre-computed value to match 1e6 counts
                'scanner_radius' : 300,
                'nb_counts' : 1e6,
                'scatter_component' :  0.36,
                'random_component' : 0.50,
                'scatter_sigma': 4.0,
            },
            optimizer_config={
                'lr': 5e-4,
                # 'weight_decay': 5e-3,
            },
            model_class_name='DRUNet',
            nn_config = {
                'n_channels': 1,
                'global_conv': 32,
                'downsample_mode': 'maxpool',
                'upsample_mode': 'bilinear',
                'conv_layer_type': 'Conv2d',
                'r_res_conv': 2,
                'use_noise_level_map': False,
                'activation': 'relu'
            },
            # model_class_name='UNet',
            # nn_config={
            #      'n_channels':1,
            #      'n_classes':1,
            #      'global_conv':32,
            #      'n_levels':4,
            #      'bilinear':True,
            #      'conv_layer_type':'Conv2d',
            #      'out_act':'relu', # Softplus is a common choice for the output activation in image-to-image translation tasks as it allows for positive outputs while avoiding the hard saturation of sigmoid. However, this can be changed to 'sigmoid' or None if needed.
            #      'residual':True,
            #      'residual_conv':False,
            #      'init':'none',
            #      'dropout': 0.0,
            #      'norm':None,
            # },
            unet_input_domain=unet_input_domain,
            unet_output_domain=unet_output_domain,
            supervised=supervised,
            reconstruction_type='fbp',
            reconstruction_config={},
            physics="backward_pet_radon",
            n_splits=2,
            num_workers=0,
            objective_type='mse',
            consensus_loss=False,
            prompt_consistency=0.0,
            seed=42
    )

    # setup mlflow
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    #
    # create experiment if not exists
    experiment_name = f"Noise2Noise_2DPET_{unet_input_domain}_to_{unet_output_domain}_v2"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
    mlflow.set_experiment(experiment_name)
    #
    # find if there is a run to resume among not finished ones
    # the run shall have the same hash as the current trainer
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params._id = '{trainer._id}' and attributes.status != 'FINISHED'",
        order_by=["start_time DESC"]
    )
    run_id = None
    run_name = f"Noise2Noise_2DPET_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if len(runs) > 0:
        for _, run in runs.iterrows():
            params_id = run['params._id']
            if params_id == trainer._id:
                run_id = run['run_id']
                run_name = None
                print(f"Resuming run {run_id} ...")
                break
    with mlflow.start_run(run_id=run_id, run_name=run_name) as run:
        # log parameters
        for key, value in trainer.__dict__.items():
            # check if value is json serializable
            try:
                mlflow.log_param(key, value)
            except:
                pass
        # resume model and optimizer if possible
        trainer.load_checkpoint(artifact_path="reboot_model")
        #
        sample = trainer.dataset_train[0][1] # get prompt sinogram sample for input shape
        # torchsummary.summary(trainer.model, input_size=(sample.shape[0], sample.shape[1], sample.shape[2]))
        #
        # start training
        trainer.fit()