import os
import itertools
import numpy as np
import mlflow
import torch
import hashlib, json
import torch.nn.functional as F
from torch.utils.data import DataLoader
import multiprocessing

from data.data_loader import SinogramGenerator, SinogramGeneratorSavedImages
from model.unet_noise2noise import Noise2NoiseBackboneModel, ModelInheritanceHandler, Noise2NoiseGradientStepDenoiser

from pytorcher.trainer import PytorchTrainer
from pytorcher.utils import normalize_batch
from pytorcher.pet_system import PetSystem
from pytorcher.utils.prior import *

from tools.image.metrics import PSNR, SSIM
from tools.image.processing import normalize

import matplotlib.pyplot as plt
import cv2
from tools.image.figure import format_phantom_figure

def get_white_matter_mask(brain):
    white_matter_mask = (brain == 36.0)

    # erosion
    kernel = np.ones((2,2), np.uint8)
    white_matter_mask = cv2.erode(white_matter_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    num_labels, labels_im = cv2.connectedComponents(white_matter_mask.astype(np.uint8))
    sizes = np.bincount(labels_im.ravel())
    largest_label = np.argmax(sizes[1:]) + 1  # Exclude background (label 0)
    white_matter_mask = (labels_im == largest_label)
    return white_matter_mask

def anscombe(x):
    return 2 * torch.sqrt( x + (3/8) )

class Noise2NoiseTrainer(PytorchTrainer):

    def __init__(
            self,
            dest_path=f"{os.getenv('WORKSPACE')}/data/noise2noise",
            dataset_train_size=2048,
            dataset_val_size=512,
            val_freq=1,
            n_epochs=25,
            batch_size=4,
            shuffle=True,
            simulator_config={
                'image_size' : (160,160),
                'voxel_size' : (2,2,2),
                'n_angles' : 300,
                'acquisition_time' : 253.3, # temporary value, will be overridden
                'half_life': 109.8*60,
                'scanner_radius' : 300,
                'nb_counts' : 1e6,
            },
            optimizer_config={
                'lr': 1e-3,
                'weight_decay': 1e-5,
            },
            backbone_model_name='UNet',
            nn_config = {
                'conv_layer_type': 'SinogramConv2d',
                'n_levels': 4,
                'global_conv': 32,
            },
            use_gsd=True,
            nn_domain='image',
            supervised=False,
            reconstruction_type='fbp',
            reconstruction_config={},
            n_splits=2,
            num_workers=0,
            objective_type='poisson',
            projection_consistency=False, # Either to project images before computing the loss.
            consensus_loss=False, # Either to use consensus loss from 10.48550/arXiv.1906.03639 or not.
            prompt_consistency=0.0, # balance for prompt consistency loss, which enforces the projected output to be close to the prompt.
            prior=None, # either 'TV' for total variation or 'Gibbs' for Gibbs prior, or None for no prior.
            prior_weight=0.0, # balance for the prior loss
            seed=42,
            register_model=True
        ):
        self.dest_path = dest_path
        self.dataset_train_size = dataset_train_size
        self.dataset_val_size = dataset_val_size
        self.val_freq = val_freq
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.simulator_config = simulator_config
        self.optimizer_config = optimizer_config
        self.supervised = supervised
        self.image_size = simulator_config.get('image_size', (160,160))
        # Validate task parameters
        assert nn_domain in ['photon', 'image'], "nn_domain must be either 'photon' or 'image'."
        if nn_domain == 'image':
            assert reconstruction_type is not None and reconstruction_type in ['fbp', 'mlem'], "Currently only 'fbp' and 'mlem' reconstructions are supported."
        assert n_splits > 1, "n_splits must be greater than 1 for noise2noise training."
        self.backbone_model_name = backbone_model_name
        self.use_gsd = use_gsd
        self.nn_config = nn_config
        self.nn_domain = nn_domain
        self.model_name = f'Noise2Noise_2DPET_{nn_domain}'
        if self.supervised:
            self.model_name += '_supervised'
        else:
            self.model_name += '_N2N'
        self.reconstruction_type = reconstruction_type
        self.reconstruction_config = reconstruction_config
        #
        self.n_splits = n_splits # n_splits means n * (n - 1) pairs will be used for noise2noise training
        #
        self.objective_type = objective_type
        self.consensus_loss = consensus_loss
        if self.consensus_loss:
            assert 'mse' in self.objective_type.lower(), "Currently consensus loss is only derived from MSE loss."
            if self.nn_domain == 'photon':
                assert self.n_splits == 2, "Currently consensus loss is only implemented for n_splits=2 in photon domain."

        if self.nn_domain == 'photon':
            projection_consistency = False
        self.projection_consistency = projection_consistency

        self.prompt_consistency = prompt_consistency

        self.prior = prior.lower() if prior is not None else None
        self.prior_weight = prior_weight
        if self.prior is not None:
            assert prior in ['tv', 'gibbs'], "Currently only 'TV' and 'Gibbs' priors are supported."
        
        self.forward_operator_type = 'radon'
        self.seed = seed

        self.num_workers = num_workers
        if self.num_workers > 1:
            multiprocessing.set_start_method('fork')

        #
        self_dict = self.__dict__.copy()
        self_dict.pop('num_workers', None) # num_workers is not relevant for model signature and caching, as it does not affect the training results. We set it to a fixed value in get_data_loader instead.
        self._id = hashlib.sha256(json.dumps(self_dict, sort_keys=True).encode()).hexdigest()
 
        #
        super(Noise2NoiseTrainer, self).__init__()
        #
        if self.projection_consistency or (self.nn_domain == 'image' and self.prompt_consistency > 0):
            self.get_pet_system_operator() # initialize forward operator for potential use in photon to image domain conversion and measurement consistency loss
        #
        self.register_model = register_model

    def get_metrics(self, metrics=[]):

        metrics.extend([
            [ 'PSNR', { 'name': 'im_psnr'} ],
            [ 'SSIM', { 'name': 'im_ssim'} ],
            [ 'PSNR', { 'name': 'val_nfpt_im_psnr'} ],
            [ 'SSIM', { 'name': 'val_nfpt_im_ssim'} ],
            [ 'PSNR', { 'name': 'val_prompt_im_psnr'} ],
            [ 'SSIM', { 'name': 'val_prompt_im_ssim'} ],
            [ 'PSNR', { 'name': 'n2n_psnr'} ],
        ])
        if self.nn_domain == 'image':
            metrics.append( [ 'SSIM', { 'name': 'n2n_ssim'} ] )
        for m in metrics:
            if m[0] == 'SSIM' or m[1] == 'PSNR':
                if self.nn_domain == 'image':
                    m[1]['bkg_val'] = 0.0 
                if 'im_' in m[1]['name']:
                    m[1]['bkg_val'] = 0.0
        if f'loss_{self.objective_type.lower()}' not in [ m[0].lower() for m in metrics ]:
            metrics.append( [ 'Mean', { 'name': f'loss_{self.objective_type.lower()}'} ] )
        #
        self.metrics = super(Noise2NoiseTrainer, self).get_metrics(metrics)
        return self.metrics

    def update_metrics(self, y_true, y_pred, metric_names=[]):
        for metric in self.metrics:
            if metric_names and metric.name not in metric_names:
                continue
            if 'loss' not in metric.name:
                metric.update_state(y_true, y_pred)

    def update_loss(self, loss):
        for metric in self.metrics:
            if 'loss' in metric.name:
                if isinstance(loss, dict):
                    metric.update_state(None, loss.get(metric.name, torch.tensor(0.0)))
                elif loss is not None:
                    metric.update_state(None, loss)

    def create_data_loader(self):

        # Get data generator for training
        self.dataset_train = SinogramGenerator(
                                         dest_path=os.path.join(self.dest_path, 'train'),
                                         length=self.dataset_train_size,
                                         seed=self.seed,
                                         **self.simulator_config
        )

        # Set a separate Generator() object for training.
        # Usefull for reproducible shuffling when num_workers > 0
        self.train_generator = torch.Generator()
        self.train_generator.manual_seed(self.seed)

        # Create DataLoader with the generator for reproducibility
        # NOTE On reboot, self.train_generator seed state is restored from previous runs in load_checkpoint()
        loader_train = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            generator=self.train_generator
            )
        
        # These parameters may be used for inference reconstruction or training backprojection later on, so we store them as trainer attributes so that they can be accessed from mlflow.
        self.scanner_radius = self.dataset_train.sinogram_simulator.scanner_radius_mm
        self.voxel_size_mm = self.dataset_train.sinogram_simulator.voxel_size_mm
        self.gaussian_PSF = self.dataset_train.sinogram_simulator.gaussian_PSF
        self.n_angles = self.dataset_train.sinogram_simulator.num_angles
        #
        self.dataset_val_seed = int(1e5) # Seed is fixed to have consistent validation sets. Changing image size or voxel size will give different results.
        if 'acquisition_time' in self.simulator_config:
            self.simulator_config.pop('acquisition_time') # ensure acquisition time is same as training set

        # Get data generator and loader for validation
        self.dataset_val = SinogramGenerator(
                                         dest_path=os.path.join(self.dest_path, 'val'),
                                         length=self.dataset_val_size,
                                         seed=self.dataset_val_seed,
                                         acquisition_time=self.dataset_train.acquisition_time, # use same acquisition time as training set
                                         **self.simulator_config
        )

        # Validation DataLoader. No shuffling, no Generator() needed for reproducibility.
        loader_val = DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        # Generate SinogramGenerator for testing on specific images
        # For instance the brain phantom and the lung phantom.
        self.dataset_val_specific = SinogramGeneratorSavedImages(
            dest_path=os.path.join(self.dest_path, 'val_specific'),
            acquisition_time=self.dataset_train.acquisition_time, # use same acquisition time as training set
            seed = self.seed,
            obj_path=(f"{os.getenv('WORKSPACE')}/data/brain_web_phantom/object/gt_web_after_scaling.hdr",
                       f"{os.getenv('WORKSPACE')}/data/lung_phantom/object/FDG_slice38.hdr"),
            att_path=(f"{os.getenv('WORKSPACE')}/data/brain_web_phantom/object/attenuat_brain_phantom.hdr",
                       f"{os.getenv('WORKSPACE')}/data/lung_phantom/object/CTAC_slice38.hdr"),
            **self.simulator_config
        )

        return loader_train, loader_val
    
    def get_pet_system_operator(self):
        self.pet_system = PetSystem(
            projector_type='parallelproj_parallel',
            projector_config={
                'num_angles': self.n_angles,
                'scanner_radius_mm': self.scanner_radius,
                'img_shape': self.image_size,
                'voxel_size_mm': self.voxel_size_mm
            },
            gaussian_PSF=self.gaussian_PSF,
            device=self.device
        )

    def get_optimizer(self, learning_rate=1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_config)
        return optimizer

    def get_signature(self):
        sample = self.dataset_train.__getitem__(0)[1]
        signature = (1, ) + tuple(sample.shape)
        return signature

    def create_model(self):
        # model expects channel-first inputs: we'll add channel dimension when calling
        ModelInheritanceHandler.set_parent(self.backbone_model_name)
        if self.use_gsd:
            model_class = Noise2NoiseGradientStepDenoiser
        else:
            model_class = Noise2NoiseBackboneModel
        model = model_class(
            domain=self.nn_domain,
            geometry={
                'n_angles': self.n_angles,
                'scanner_radius_mm': self.scanner_radius,
                'gaussian_PSF_fwhm_mm': self.gaussian_PSF,
                'voxel_size_mm': self.voxel_size_mm
             },
             image_size=self.image_size,
             reconstruction_config=self.reconstruction_config,
             reconstruction_type=self.reconstruction_type,
             nn_config=self.nn_config
        )
        model = model.to(self.device)
        #
        model.domain = self.nn_domain
        #
        return model

    def get_objective(self):
        type = self.objective_type.lower()
        if 'mse' in type:
            objective = torch.nn.MSELoss()
        elif type == 'l1':
            objective = torch.nn.L1Loss()
        elif type == 'hubert':
            objective = torch.nn.SmoothL1Loss(beta=1.0)
        elif type == 'poisson':
            objective = torch.nn.PoissonNLLLoss(log_input=False, full=False)
        elif type == 'kl_divergence':
            objective = torch.nn.KLDivLoss(log_target=True, reduction='batchmean')
        #
        self.model.loss_type = type # inform model about loss type for potential post-processing
        #
        return objective

    def log_and_reset_metrics(self, epoch):
        print(f'End of Epoch {epoch+1}, metrics: ')
        for metric in self.metrics:
            print(f'{metric.name}: {metric.result():.4f}')

            # Log metrics to MLflow
            mlflow.log_metric(metric.name, metric.result(), step=epoch)

            # Reset metric
            metric.reset_states()

    def compute_reference_metrics(self):
        """
        Compute some reference metrics on validation set before training starts.
        This is useful to compare reconstruction results against.
        We compute the reconstruction from the noise-free sinogram (nfpt) only. This metric cannot be beaten by denoising.
        We compute the reconstruction from the prompt sinogram only. This metric must be beaten by denoising.
        """
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (path, prompt, nfpt, gth, att, att_sino, corr, scale) in enumerate(self.loader_val):

                print(f'Computing reference metrics, batch {batch_idx+1}/{len(self.loader_val)} ...')

                # move data to device
                prompt = prompt.to(self.device).float()
                nfpt = nfpt.to(self.device).float()
                gth = gth.to(self.device).float()
                scale = scale.to(self.device).float()
                corr = corr.to(self.device).float()

                # reconstruction from noise-free sinogram
                recon_nfpt = self.model.reconstruction(nfpt, scale=scale, corr=corr, attenuation_map=att, **self.reconstruction_config)
                # update im_ metrics for reference
                metrics_to_update = [ m.name for m in self.metrics if 'nfpt' in m.name ]
                self.update_metrics(normalize_batch(gth), normalize_batch(recon_nfpt), metric_names=metrics_to_update)

                # reconstruction from prompt sinogram
                recon_prompt = self.model.reconstruction(prompt, scale=scale, corr=corr, attenuation_map=att, **self.reconstruction_config)
                # update im_ metrics for reference
                metrics_to_update = [ m.name for m in self.metrics if 'prompt' in m.name ]
                self.update_metrics(normalize_batch(gth), normalize_batch(recon_prompt), metric_names=metrics_to_update)

            print('Reference metrics on validation set before training:')
            for metric in self.metrics:
                if 'nfpt' in metric.name or 'prompt' in metric.name:
                    print(f'{metric.name}: {metric.result():.4f}')
                    mlflow.log_metric(metric.name, metric.result())
                    metric.reset_states()

    def pet_system_operator(self, image, attenuation_map=None, scale=None, forward_operator_type='radon'):
        """
        Apply the PET forward operator for self-supervised reconstruction.
        This will be used only if input domain is 'photon' and output domain is 'image',
        in which case we need to convert the reconstructed image back to sinogram domain for loss computation and gradient computation.
        
        :param image: Model prediction in the image domain.
        :param attenuation_map: Description
        :param scale: Description
        :param forward_operator_type: Description
        """
        #
        assert forward_operator_type.lower() in ['radon'], "Currently only 'radon' forward operator is supported for photon to image domain conversion."
        if forward_operator_type.lower() == 'radon':
            sinogram = self.pet_system.forward(
                image=image,
                attenuation_map=attenuation_map,
                scale=scale
            )
        else:
            raise NotImplementedError("Currently only 'radon' forward operator is supported for photon to image domain conversion.")
        return sinogram
    
    def compute_count_loss(self, output, target):
        """
        Compute count loss for photon domain.
        Photon domain loss can be either Poisson NLL or heteroscedastic MSE or MSE in Anscombe domain.
        """
        if self.objective_type.lower() == 'poisson':
            loss = self.objective(output, target)
        elif self.objective_type.lower() == 'mse':
            eps = 1.0
            weight = 1.0 / (target + eps)
            loss = (weight * (output - target) ** 2).mean()
        elif self.objective_type.lower() == 'l1':
            eps = 1.0
            weight = 1.0 / (target + eps)
            loss = (weight * torch.abs(output - target)).mean()
        elif self.objective_type.lower() == 'mse_anscombe':
            if not self.model.training:
                output = anscombe(output) # At inference time, rescale back to Anscombe domain
            target = anscombe(target)
            loss = self.objective(output, target)
        else:
            raise ValueError("Invalid objective type for photon domain. Supported types are 'poisson', 'mse' and 'mse_anscombe'.")
        return loss
    
    def compute_loss(self, output, target, attenuation_map=None, scale=None, corr=None, mask_im=None, mask_sino=None):

        if mask_im is not None:
            if self.nn_domain == 'image':
                output = output * mask_im
            else:
                output = output * mask_sino
        if mask_sino is not None:
            corr = corr * mask_sino
            if self.nn_domain == 'image' and not self.projection_consistency:
                target = target * mask_im
            else:
                target = target * mask_sino

        if self.nn_domain == 'photon':
            loss = self.compute_count_loss(output, target)
        elif self.nn_domain == 'image' and not self.projection_consistency:
            loss = self.objective(output, target)
        elif self.nn_domain == 'image' and self.projection_consistency:
            if attenuation_map is None:
                print("Warning: No attenuation map provided for photon to image domain conversion. Assuming no attenuation for forward operator.")

            #
            projected_output = self.pet_system_operator(
                output,
                attenuation_map=attenuation_map,
                scale=scale,
                forward_operator_type=self.forward_operator_type
            )
            #
            loss = self.compute_count_loss(projected_output + corr, target)
        #
        return loss
    
    def compute_loss_addons(self, outputs, prompt, attenuation_map=None, scale=None, corr=None, mask_im=None, mask_sino=None):
        if mask_im is not None:
            if self.nn_domain == 'image':
                outputs = [output * mask_im for output in outputs]
            else:
                outputs = [output * mask_sino for output in outputs]
        if mask_sino is not None:
            prompt = prompt * mask_sino
        #
        loss_addons = {}
        #
        # remove consistency term from loss if consensus_loss is True
        if self.consensus_loss:
            loss_addons[f'consensus_loss'] = 0.0
            for (i,j) in list(itertools.combinations(range(self.n_splits), 2)):
                output_i = outputs[i]
                output_j = outputs[j]
                #
                if self.nn_domain == 'image' and not self.projection_consistency:
                    consensus_loss_ij = (1 / self.n_splits**2) * self.compute_loss(output_i, output_j)
                elif self.nn_domain == 'photon':
                    consensus_loss_ij = (1 / self.n_splits**2) *self.compute_count_loss(output_i, output_j)
                elif self.nn_domain == 'image' and self.projection_consistency:
                    projected_i = self.pet_system_operator(
                        output_i,
                        attenuation_map=attenuation_map,
                        scale=scale,
                        forward_operator_type=self.forward_operator_type
                    )
                    projected_j = self.pet_system_operator(
                        output_j,
                        attenuation_map=attenuation_map,
                        scale=scale,
                        forward_operator_type=self.forward_operator_type
                    )
                    if mask_sino is not None:
                        projected_i = projected_i * mask_sino
                        projected_j = projected_j * mask_sino
                    #
                    consensus_loss_ij = (1 / self.n_splits**2) * self.compute_count_loss(projected_i, projected_j)
                #
                loss_addons[f'consensus_loss'] -= consensus_loss_ij
        #
        if self.prompt_consistency > 0:

            if self.nn_domain == 'photon':

                if mask_sino is not None:
                    prompt = prompt * mask_sino
                    outputs = [output * mask_sino for output in outputs]

                if self.consensus_loss:
                    z = torch.stack(outputs, dim=0) # (n_splits, B, C, H, W)
                    z = torch.sum(z, dim=0) # (B, C, H, W)
                    loss_prompt_consistency = self.prompt_consistency * self.compute_count_loss(z, prompt)
                else:
                    loss_prompt_consistency = self.prompt_consistency * sum(self.compute_count_loss(z_, prompt / self.n_splits) for z_ in outputs)

            elif self.nn_domain == 'image' and not self.projection_consistency:

                if mask_im is not None:
                    outputs = [output * mask_im for output in outputs]

                if mask_sino is not None:
                    corr = corr * mask_sino
                    prompt = prompt * mask_sino

                if self.consensus_loss:
                    z = torch.stack(outputs, dim=0) # (n_splits, B, C, H, W)
                    z = torch.mean(z, dim=0) # (B, C, H, W)

                    z_projected = self.pet_system_operator(
                        z,
                        attenuation_map=attenuation_map,
                        scale=scale,
                        forward_operator_type=self.forward_operator_type
                    )
                    loss_prompt_consistency = self.prompt_consistency * self.compute_count_loss(z_projected + corr, prompt)
                else:
                    z_projected = [ self.pet_system_operator(
                        z_,
                        attenuation_map=attenuation_map,
                        scale=scale,
                        forward_operator_type=self.forward_operator_type
                    ) for z_ in outputs ]

                    loss_prompt_consistency = self.prompt_consistency * sum(self.compute_count_loss(z_ + corr, prompt) for z_ in z_projected)

            elif self.nn_domain == 'image' and self.projection_consistency:

                if mask_im is not None:
                    outputs = [output * mask_im for output in outputs]
                if mask_sino is not None:
                    corr = corr * mask_sino
                    prompt = prompt * mask_sino

                if self.consensus_loss:

                    z = torch.stack(outputs, dim=0) # (n_splits, B, C, H, W)
                    z = torch.mean(z, dim=0) # (B, C, H, W)
                    z_projected = self.pet_system_operator(
                        z,
                        attenuation_map=attenuation_map,
                        scale=scale,
                        forward_operator_type=self.forward_operator_type
                    )
                    loss_prompt_consistency = self.prompt_consistency * self.compute_count_loss(z_projected + corr, prompt)

                else:
                    z_projected = [ self.pet_system_operator(
                        z_,
                        attenuation_map=attenuation_map,
                        scale=scale,
                        forward_operator_type=self.forward_operator_type
                    ) for z_ in outputs ]
                    loss_prompt_consistency = self.prompt_consistency * sum(self.compute_count_loss(z_ + corr, prompt) for z_ in z_projected)

            loss_addons[f'prompt_consistency_loss'] = loss_prompt_consistency

        return loss_addons
        
    def fit(self):

        # if self.initial_epoch == 0:
        #     self.compute_reference_metrics()
        
        # Remove reference metrics from trainer metrics list
        self.metrics = [ m for m in self.metrics if 'nfpt' not in m.name and 'prompt' not in m.name ]
        
        for epoch in range(self.initial_epoch, self.n_epochs):

            m_dict_train = {}
            # TRAIN
            self.model.train()
            for batch_idx, (path, prompt, nfpt, gth, att, att_sino, corr, scale) in enumerate(self.loader_train):

                # Set target for task
                if self.nn_domain == 'image':
                    target = gth
                else:
                    target = nfpt
                
                # Move data to device
                prompt = prompt.to(self.device).float()
                target = target.to(self.device).float()
                gth = gth.to(self.device).float()
                nfpt = nfpt.to(self.device).float()
                scale = scale.to(self.device).float()
                att = att.to(self.device).float()
                att_sino = att_sino.to(self.device).float()
                corr = corr.to(self.device).float()
                #
                # split prompt with multinomial statistics
                split_losses = []
                split_outputs = self.n_splits*[None, ] # to store outputs for all splits for potential consensus loss computation
                if not self.supervised:
                    splitted_prompts = self.model.split_prompt(prompt, mode='multinomial')
                else:
                    splitted_prompts = self.n_splits * [prompt / self.n_splits, ] # In supervised setting, we just use the same prompt for all splits, and divide by n_splits to keep the same scale of inputs to the model and loss.
                scale = scale / self.n_splits
                corr = corr / self.n_splits
                pairwise_permutations = list(itertools.permutations(range(self.n_splits), 2))
                #
                # Apply reconstruction if needed
                if self.nn_domain == 'photon':
                    x =None
                else:
                    x = [self.model.reconstruction(s, scale=scale, corr=corr, attenuation_map=att, **self.reconstruction_config) for s in splitted_prompts]

                if self.nn_domain == 'photon':
                    target = target / self.n_splits # In photon domain, we divide poisson parameter accordingly

                # Get mask for loss computation
                mask_im = (target > 0).float()
                mask_sino = (att_sino > 1.02).float()
                if self.nn_domain == 'image':
                    mask = mask_im
                else:
                    mask = mask_sino

                # Denoise and compute loss on all pairs
                for (i, j) in pairwise_permutations:
                    if x is None:
                        x_i, x_j = None, None
                    else:
                        x_i, x_j = x[i], x[j]
                    # Inference and loss computation for pair (i, j)
                    output_i = self.model(
                        x=x_i,
                        y=splitted_prompts[i],
                        scale=scale,
                        mask=mask,
                        attenuation_map=att,
                        corr=corr
                    )
                    if self.supervised:
                        if (self.nn_domain == 'image' and self.projection_consistency) or self.nn_domain == 'photon':
                            loss_target = nfpt / self.n_splits
                        else:
                            loss_target = self.model.reconstruction(nfpt, scale=scale, corr=corr, attenuation_map=att, **self.reconstruction_config)
                    else:
                        if self.nn_domain == 'image' and not self.projection_consistency:
                            loss_target = x_j
                        else:
                            loss_target = splitted_prompts[j]
                    #
                    loss_ij = self.compute_loss(output=output_i, target=loss_target, attenuation_map=att, corr=corr, scale=scale, mask_im=mask_im, mask_sino=mask_sino)
                    # We only update n2n_ metrics and loss here
                    metrics_to_update = [ m.name for m in self.metrics if m.name.startswith('n2n_') or m.name.startswith('loss_') ]
                    self.update_metrics(normalize_batch(target), normalize_batch(output_i), metric_names=metrics_to_update)
                    #
                    split_losses.append(loss_ij)
                    split_outputs[i] = output_i
                # global loss
                loss = sum(split_losses) / len(split_losses)  # average over all pairs
                #
                # compute loss addons
                loss_addons = self.compute_loss_addons(
                    outputs=split_outputs,
                    prompt=prompt,
                    attenuation_map=att,
                    scale=scale * self.n_splits,
                    corr=corr * self.n_splits,
                    mask_im=mask_im,
                    mask_sino=mask_sino
                )
                loss = loss + sum(loss_addons.values())
                # 
                # Update loss
                self.update_loss(loss)
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # log batch loss
                mlflow.log_metric(f'batch_loss_{self.objective_type.lower()}', loss.item(), step=epoch * len(self.loader_train) + batch_idx)
                #
                m_dict_train = {metric.name: f"{metric.result():.4f}" for metric in self.metrics if metric.name.startswith('n2n_') or metric.name.startswith('loss_')}
                print(f'Epoch [{epoch+1}/{self.n_epochs}], Train,  Step [{batch_idx+1}/{len(self.loader_train)}]', f"Metrics : {m_dict_train}")

            for metric in self.metrics:
                # reset state
                metric.reset_states()

            torch.cuda.empty_cache()

            m_dict_val = {}
            # VALIDATION
            if (epoch + 1) % self.val_freq == 0:
                # run model in evaluation mode and avoid building computation graph
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (_, prompt, nfpt, gth, att, att_sino, corr, scale) in enumerate(self.loader_val):

                        # set target for task
                        if self.nn_domain == 'image':
                            target = gth
                        else:
                            target = nfpt

                        # move data to device
                        prompt = prompt.to(self.device).float()
                        target = target.to(self.device).float()
                        gth = gth.to(self.device).float()
                        nfpt = nfpt.to(self.device).float()
                        scale = scale.to(self.device).float()
                        att = att.to(self.device).float()
                        att_sino = att_sino.to(self.device).float()
                        corr = corr.to(self.device).float()
                        #
                        # Split data with multinomial statistics. This is done to match training data distribution
                        y_splits = self.model.split_prompt(prompt, mode='multinomial')
                        #
                        scale = scale / self.n_splits
                        corr = corr / self.n_splits

                        # Apply reconstruction if needed
                        if self.nn_domain == 'image':
                            x = [self.model.reconstruction(s, scale=scale, corr=corr, attenuation_map=att, **self.reconstruction_config) for s in y_splits]
                        else:
                            x = None

                        # Create mask
                        mask_im = (target > 0).float()
                        mask_sino = (att_sino > 1.02).float()
                        if self.nn_domain == 'image':
                            mask = mask_im
                        else:
                            mask = mask_sino
                        #
                        splits_infered = [
                            self.model(
                                y=y_splits[i],
                                x=x[i] if x is not None else None,
                                scale=scale,
                                attenuation_map=att,
                                corr=corr,
                                mask=mask
                            ) for i in range(self.n_splits)
                        ]
                        #
                        if self.nn_domain == 'photon' and not self.supervised:
                            target = target / self.n_splits # In photon domain, we divide poisson parameter accordingly
                        else:
                            target = target
                        #
                        val_split_losses = []
                        for (i, j) in pairwise_permutations:
                            if x is not None:
                                x_i = x[i]
                                x_j = x[j]
                            else:
                                x_i, x_j = None, None
                            # inference on i
                            out_i = splits_infered[i]
                            # Loss computation for pair (i, j)
                            if self.supervised:
                                if (self.nn_domain == 'image' and self.projection_consistency) or self.nn_domain == 'photon':
                                    loss_target = nfpt / self.n_splits
                                else:
                                    loss_target = self.model.reconstruction(nfpt, scale=scale, corr=corr, attenuation_map=att, **self.reconstruction_config)
                            else:
                                if self.nn_domain == 'image' and not self.projection_consistency:
                                    loss_target = x_j
                                else:
                                    loss_target = y_splits[j]
                            val_loss = self.compute_loss(output=out_i, target=loss_target, attenuation_map=att, corr=corr, scale=scale, mask_im=mask_im, mask_sino=mask_sino)
                            val_split_losses.append(val_loss)
                            # Update n2n_ and loss metrics for validation
                            metrics_to_update = [ m.name for m in self.metrics if m.name.startswith('n2n_') or m.name.startswith('loss_') ]
                            self.update_metrics(normalize_batch(target), normalize_batch(out_i), metric_names=metrics_to_update)
                        #
                        # compute loss addons
                        val_loss_addons = self.compute_loss_addons(
                            outputs=splits_infered,
                            prompt=prompt,
                            attenuation_map=att,
                            scale=scale * self.n_splits,
                            corr=corr * self.n_splits,
                            mask_im=mask_im,
                            mask_sino=mask_sino
                        )
                        val_loss = val_loss + sum(val_loss_addons.values())
                        #
                        self.update_loss(val_loss)
                        # Apply reconstruction if needed and average outputs
                        if self.nn_domain == 'photon':
                            output = [ self.model.reconstruction(s, scale=scale, corr=corr, attenuation_map=att, **self.reconstruction_config) for s in splits_infered ]  # (B, C, H, W)
                            output = torch.stack(output, dim=0)  # (n_splits, B, C, H, W)
                            output = torch.mean(output, dim=0)  # (B, C, H, W)
                        else:
                            splits_infered = torch.stack(splits_infered, dim=0)  # (n_splits, B, C, H, W)
                            output = torch.mean(splits_infered, dim=0)  # (B, C, H, W)
                        # Update im_ metrics for validation
                        metrics_to_update = [ m.name for m in self.metrics if m.name.startswith('im_') ]
                        self.update_metrics(normalize_batch(gth), normalize_batch(output), metric_names=metrics_to_update)
                        #
                        m_dict_val = {f'val_{metric.name}': metric.result() for metric in self.metrics}
                        print(f'Epoch [{epoch+1}/{self.n_epochs}], Validation, Step [{batch_idx+1}/{len(self.loader_val)}]', f'Metrics : {m_dict_val}')

                # print and reset metrics
                for metric in self.metrics:
                    metric.reset_states()

            # perform evaluation on brain phantom and log results as artifact for visual inspection of model performance evolution during training
            for batch_idx, (path, prompt, nfpt, gth, att, att_sino, corr, scale) in enumerate(self.dataset_val_specific):

                if batch_idx == 0:
                    phantom_name = "brain_web_phantom"
                elif batch_idx == 1:
                    phantom_name = "lung_phantom"

                print(f'Inference on {phantom_name} for visual inspection of model performance evolution during training, batch {batch_idx+1}/{len(self.dataset_val_specific)} ...')


                # move data to device
                gth = gth.to(self.device).float().unsqueeze(0) # add batch dimension
                prompt = prompt.to(self.device).float().unsqueeze(0) # add batch dimension
                nfpt = nfpt.to(self.device).float().unsqueeze(0) # add batch dimension
                scale = torch.tensor(scale).to(self.device).float().unsqueeze(0) # add batch dimension
                att = att.to(self.device).float().unsqueeze(0) # add batch dimension
                att_sino = att_sino.to(self.device).float().unsqueeze(0) # add batch dimension
                corr = corr.to(self.device).float().unsqueeze(0) # add batch dimension

                # Denoised reconstruction from prompt
                self.model.image_size = gth.shape[-2:] # update model image size to match phantom image size
                #
                recon_noise2noise = self.model.forward_inference(prompt, scale=scale, corr=corr, attenuation_map=att, monte_carlo_steps=1, split=True, mask=(gth > 0)) # (B, C, H, W)
                recon_noise2noise = recon_noise2noise.to('cpu').squeeze().detach().numpy().astype(np.float32)
                gth = gth.to('cpu').float().squeeze().detach().numpy().astype(np.float32)
                mask = gth > 0
                recon_noise2noise = recon_noise2noise * mask

                # Compute metrics

                PSNR_denoised = PSNR(I=gth, K=recon_noise2noise, mask=mask)
                SSIM_denoised = SSIM(img1=gth, img2=recon_noise2noise, mask=mask)
                metrics = {
                    'psnr': PSNR_denoised.item(),
                    'ssim': SSIM_denoised.item()
                }
                if phantom_name == "brain_web_phantom":
                    # bias variance
                    white_matter_mask = get_white_matter_mask(gth)
                    bias_white_matter = (torch.abs(torch.mean(torch.tensor(recon_noise2noise)[white_matter_mask]) - torch.tensor(gth)[white_matter_mask][0]) / torch.tensor(gth)[white_matter_mask][0]).item()
                    expectation_squared_white_matter = torch.mean(torch.tensor(recon_noise2noise)[white_matter_mask] ** 2).item()
                    variance_white_matter = (torch.sqrt(expectation_squared_white_matter - (torch.mean(torch.tensor(recon_noise2noise)[white_matter_mask]) ** 2)) / torch.mean(torch.tensor(recon_noise2noise)[white_matter_mask])).item()
                    metrics['bias_white_matter'] = bias_white_matter
                    metrics['variance_white_matter'] = variance_white_matter
                #
                reconstructions = {
                    'denoised': recon_noise2noise
                }
                
                if epoch == 0:
                    for input_type, input in zip(['nfpt', 'prompt'], [nfpt, prompt]):
                        recon_input = self.model.reconstruction(input, scale=scale, corr=corr, attenuation_map=att, **self.reconstruction_config).to('cpu').squeeze().detach().numpy().astype(np.float32)
                        recon_input = recon_input
                        PSNR_input = PSNR(I=gth, K=recon_input, mask=mask)
                        SSIM_input = SSIM(img1=gth, img2=recon_input, mask=mask)
                        metrics[f'psnr_{input_type}'] = PSNR_input.item()
                        metrics[f'ssim_{input_type}'] = SSIM_input.item()
                        reconstructions[f'{input_type}'] = recon_input * mask

                for input_type, recon in reconstructions.items():
                    #log raw numpy array as artifact
                    tmp_save_path = os.path.join(self.dest_path, f'reconstruction_{input_type}_raw/epoch_{epoch+1}.npy')
                    os.makedirs(os.path.dirname(tmp_save_path), exist_ok=True)
                    np.save(tmp_save_path, recon)
                    mlflow.log_artifact(tmp_save_path, artifact_path=phantom_name)
                    os.remove(tmp_save_path)
                    # log figure with metrics as annotations
                    if phantom_name == "brain_web_phantom":
                        format_kwargs = {
                            'magnification': 2,
                            'vmin': 0.0,
                            'vmax': 230.0,
                            'roi_mask': [[24, 79], [31, 86]],
                            'shift': True
                        }
                    elif phantom_name == "lung_phantom":
                        format_kwargs = {
                            'magnification': 3,
                            'vmin': 0.0,
                            'vmax': 80.0,
                            'roi_mask': [[42, 115], [50, 123]],
                            'shift': (None, 5)
                        }
                    fig = format_phantom_figure(
                        recon,
                        annotations={
                            'ssim': metrics.get(f'ssim_{input_type}', metrics.get('ssim', None)),
                            'psnr': metrics.get(f'psnr_{input_type}', metrics.get('psnr', None)),
                        },
                        **format_kwargs
                    )
                    mlflow.log_figure(fig, f'{phantom_name}/reconstruction_{input_type}/epoch_{epoch+1}.png')
                #
                mlflow.log_dict(metrics, f'{phantom_name}/metrics_epoch/{epoch+1}.json')
                # mlflow.log_figure(fig, f'{phantom_name}/reconstruction/epoch_{epoch+1}.png')

                if phantom_name == "brain_web_phantom":
                    # plot paretto front evolution for bias and variance on brain phantom during training
                    white_matter_bias = []
                    white_matter_variance = []
                    for epoch_ in range(0, epoch+1):
                        artifact_path = f'{phantom_name}/metrics_epoch/{epoch_+1}.json'
                        metrics = mlflow.artifacts.load_dict(f"{mlflow.active_run().info.artifact_uri}/{artifact_path}")
                        white_matter_bias.append(metrics.get('bias_white_matter', None))
                        white_matter_variance.append(metrics.get('variance_white_matter', None))

                    with plt.style.context('ggplot'):
                        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                        ax.plot(white_matter_variance, white_matter_bias, marker='o', color='gray')
                        for epoch_, (bias, var) in enumerate(zip(white_matter_bias, white_matter_variance), start=1):
                            ax.annotate(f'E{epoch_}', (var, bias), fontsize=8, ha='left')
                        ax.set_ylabel('Bias in white matter')
                        ax.set_xlabel('Variance in white matter')
                        ax.set_title('Bias-Variance Evolution in White Matter during Training')
                        plt.tight_layout()
                        plt.close(fig)
                        mlflow.log_figure(fig, f'{phantom_name}/bias_variance_evolution.png')

            # # metric monitoring
            m_dict = {**m_dict_train, **m_dict_val}
            # monitored_metrics = [ m for m in list(m_dict.keys()) if (m.startswith('val_loss_') and 'reg_' not in m) or m.startswith('val_im_') ]
            # for metric_name in monitored_metrics:
            #     if 'loss' in metric_name:
            #         mode = 'min'
            #     else:
            #         mode = 'max'
            #     self.mlflow_metric_monitoring(epoch, metric_name, m_dict[metric_name], mode=mode)

            # 
            self.model.image_size = self.image_size # update model image size to training image size

            # log metrics
            for metric_name, metric_value in m_dict.items():
                mlflow.log_metric(metric_name, metric_value, step=epoch+1)

            # log reboot model as artifact
            self.mlflow_log_checkpoint_as_artifact(epoch, artifact_path="reboot_model")

            torch.cuda.empty_cache()

        # log final model
        if self.register_model:
            registered_model_name =self.model_name
        else:
            registered_model_name = None
        mlflow.pytorch.log_model(self.model, artifact_path="final_model", registered_model_name=registered_model_name)

        # # log final best models
        # for metric_name in monitored_metrics:
        #     # retrieved best model checkpoint from artifact
        #     self.load_checkpoint(artifact_path=f"best_model_{metric_name}")
        #     mlflow.pytorch.log_model(self.model, artifact_path=f"best_model_{metric_name}", registered_model_name=f"{self.model_name}_{metric_name}")