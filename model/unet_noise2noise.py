import os
from pytorcher.models import *

from pytorcher.pet_system import PetSystem

import torch
from torch import nn

import hashlib


from pytorcher.utils import tensor_hash, iradon

class ModelInheritanceHandler:

    parent = nn.Module  # default parent class
    _registered_subclasses = []  # track subclasses to update when parent changes
    
    @staticmethod
    def register_subclass(cls):
        """Register a subclass to be updated when parent class changes."""
        ModelInheritanceHandler._registered_subclasses.append(cls)
        return cls
    
    @staticmethod
    def set_parent(parent_class_name):
        """Change the parent class at runtime and update all registered subclasses."""
        if parent_class_name in globals():
            parent_class = globals()[parent_class_name]
            ModelInheritanceHandler.parent = parent_class
            # Update all registered subclasses to inherit from the new parent
            for subclass in ModelInheritanceHandler._registered_subclasses:
                subclass.__bases__ = (parent_class,)
        else:
            # Try dynamic import if not found in globals
            try:
                import pytorcher.models as models_module
                if hasattr(models_module, parent_class_name):
                    parent_class = getattr(models_module, parent_class_name)
                    ModelInheritanceHandler.parent = parent_class
                    for subclass in ModelInheritanceHandler._registered_subclasses:
                        subclass.__bases__ = (parent_class,)
                else:
                    raise ValueError(f"Parent class {parent_class_name} not found in pytorcher.models or global scope.")
            except ImportError:
                raise ValueError(f"Parent class {parent_class_name} not found and could not import from pytorcher.models.")

@ModelInheritanceHandler.register_subclass
class Noise2NoiseBackboneModel(nn.Module):

    """
    U-Net model for Noise2Noise.
    If outputs are in the photon/Poisson domain using 'mse_anscombe' loss, apply the rescale stage at inference time.
    """

    def __init__(self,
                 domain='image',
                 geometry={},
                 reconstruction_type='fbp',
                 reconstruction_config={},
                 image_size=(160,160),
                 n_splits=2,
                 nn_config={}):
        # Store the current parent class name for pickle support
        self._parent_class_name = None
        if ModelInheritanceHandler.parent != nn.Module:
            # Store the name of the custom parent class
            self._parent_class_name = ModelInheritanceHandler.parent.__name__
        #
        self.domain = domain
        #
        self.n_angles = geometry.get('n_angles', 300)
        self.scanner_radius = geometry.get('scanner_radius_mm', 300)
        self.gaussian_PSF = geometry.get('gaussian_PSF_fwhm_mm', 4.0)
        self.voxel_size_mm = geometry.get('voxel_size_mm', (2.0, 2.0))
        #
        #
        self.image_size = image_size
        self.n_splits = n_splits
        self.reconstruction_type = reconstruction_type
        self.reconstruction_config = reconstruction_config
        assert self.reconstruction_type.lower() in ['fbp', 'mlem'], "Currently only FBP and MLEM are supported."
        #
        # must call nn.Module init before assigning any nn.Module attributes such as done in init_pet_system_operator and get_reconstruction_operator
        super(Noise2NoiseBackboneModel, self).__init__(**nn_config)
    
    def __getstate__(self):
        """Prepare model for pickling by storing the parent class name."""
        state = self.__dict__.copy()
        # Store parent class name for restoration on unpickling
        state['_parent_class_name'] = ModelInheritanceHandler.parent.__name__
        return state
    
    def __setstate__(self, state):
        """Restore model from pickle and apply the correct parent class."""
        parent_class_name = state.pop('_parent_class_name', None)
        self.__dict__.update(state)
        # Restore the parent class if it was customized
        if parent_class_name and parent_class_name != 'Module':
            ModelInheritanceHandler.set_parent(parent_class_name)
        
    def get_pet_system_operator(self):

        if not hasattr(self, 'pet_system_operator'):
            geometry = {
                'num_angles':self.n_angles,
                'scanner_radius_mm':self.scanner_radius,
                'voxel_size_mm':self.voxel_size_mm,
                'img_shape': self.image_size,
            }
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
            pet_system_operator = PetSystem(
                projector_type='parallelproj_parallel',
                projector_config=geometry,
                gaussian_PSF=self.gaussian_PSF,
                device=device
            )

        return pet_system_operator

    def reconstruction(self, y, scale=None, attenuation_map=None, corr=None, mode='fbp', **kwargs):

        if mode == 'fbp':
            # update scale if corr is provided
            if corr is not None and scale is not None:
                count_ratio = torch.sum(corr, dim=[1,2,3]) / (torch.sum(y, dim=[1,2,3]))  # (B,)
                scale = scale * count_ratio  # (B,)

            if corr is not None:
                y = torch.clamp(y - corr, min=0)  # (B, C, H, W)

        pet_system_operator = self.get_pet_system_operator()
        #
        if mode == 'fbp':
            x_recon = pet_system_operator.fbp(y, scale=scale, **kwargs) # (B, C, H, W)

        elif mode == 'mlem':
            x_recon = pet_system_operator.mlem(y, corr=corr, attenuation_map=attenuation_map, scale=scale, **kwargs) # (B, C, H, W)
        else:
            raise ValueError(f"Unknown reconstruction mode: {mode}")

        
        return x_recon


    def split_prompt(self, prompt, mode='multinomial', consistent=True, seed=None):
        """
        Split prompt sinogram into n_splits sinograms with multinomial statistics.

        :param prompt: (B, C, H, W) sinogram batch (non-negative integer counts)
        :param consistent: if True, each sample gets its own deterministic seed
        :param seed: global seed used only when consistent=False
        :return: list of n_splits tensors, each (B, C, H, W)
        """
        B = prompt.shape[0]
        device = prompt.device

        if mode != 'multinomial':
            raise ValueError(f'Unknown split mode: {mode}')

        # Create per-sample generators
        generators = []

        if consistent:
            # Deterministic seed per sample based on its content
            for b in range(B):
                g = torch.Generator(device=device)
                s = tensor_hash(prompt[b], format='int') % (2**63 - 1)
                g.manual_seed(s)
                generators.append(g)
        else:
            # One shared generator (e.g. inference-time reproducibility)
            g = torch.Generator(device=device)
            if seed is not None:
                g.manual_seed(seed)
            generators = [g] * B

        # Multinomial splitting via sequential binomials
        remaining = prompt.clone()
        splits = []

        for i in range(self.n_splits - 1):
            p = 1.0 / (self.n_splits - i)

            split_i = torch.empty_like(prompt)

            # We loop over batch for RNG correctness, but each draw is fully vectorized over (C,H,W)
            for b in range(B):
                split_i[b] = torch.binomial(
                    count=remaining[b],
                    prob=torch.full_like(remaining[b], p, dtype=torch.float32),
                    generator=generators[b]
                )

            splits.append(split_i)
            remaining = remaining - split_i

        splits.append(remaining)  # last split gets leftovers

        return splits
    
    def forward_inference(self, y, scale, corr=None, seed=None, attenuation_map=None, mask=None, monte_carlo_steps=1, split=True):
        """
        Forward pass through the Noise2Noise U-Net model with input splitting and output aggregation.
        The splitting process has some randomness; set seed for reproducibility.
        Use monte_carlo_steps > 1 for multiple stochastic passes and average the results.
        :param y: (B, C, H, W) input sinogram tensor
        :param attenuation_map: (B, 1, H, W) attenuation map tensor, used for adjoint computation.
        :param scale: (B,) scale factor to be applied to sinogram before reconstruction.
        :param seed: random seed for splitting
        :param attenuation_map: (B, 1, H, W) attenuation map tensor, used for adjoint computation.
        :param mask: (B, 1, H, W) mask tensor to apply to the output
        :param monte_carlo_steps: number of stochastic passes to average
        :return: (B, C, H, W) output image tensor
        """
        if seed is not None:
            torch.manual_seed(seed)

        assert len(scale) == y.shape[0], "Scale must have the same batch size as input y"

        # Stack scale accordingly
        if split:
            scale = (scale / self.n_splits)#.repeat(self.n_splits) # Dividing the number of counts by n_splits is equivalent to dividing scale factor by n_splits
            scale = scale.repeat(self.n_splits)  # (B * n_splits,)

        if attenuation_map is not None and split:
            attenuation_map = attenuation_map.repeat_interleave(repeats=self.n_splits, dim=0)  # (B * n_splits, 1, H, W)

        if corr is not None and split:
            corr = (corr / self.n_splits).repeat_interleave(repeats=self.n_splits, dim=0)  # (B * n_splits, C, H, W)

        outputs = torch.zeros((y.shape[0], y.shape[1], self.image_size[0], self.image_size[1]), device=y.device) # (B, C, H, W)
        for i in range(monte_carlo_steps):
            # Split input sinogram
            if split:
                splitted_prompts = self.split_prompt(y, mode='multinomial', consistent=False)  # list of (B, C, H, W)
            else:
                splitted_prompts = [y, ]

            splitted_prompts = torch.cat(splitted_prompts, dim=0)  # (B * n_splits, C, H, W)

            # Denoise
            splits_denoised = self.forward(y=splitted_prompts, scale=scale, corr=corr, attenuation_map=attenuation_map)  # (B * n_splits, C, H, W)

            # Apply reconstruction if needed and average outputs
            if self.domain == 'photon':
                output = self.reconstruction(splits_denoised, scale=scale, corr=corr, attenuation_map=attenuation_map, mode=self.reconstruction_type, **self.reconstruction_config)
            else:
                output = splits_denoised
            
            # Expand splits to have (n_splits, B, C, H, W)
            output = torch.chunk(output, self.n_splits, dim=0)  # list of (B, C, H, W)
            output = torch.stack(output, dim=0)  # (n_splits, B, C, H, W)
            output = torch.mean(output, dim=0)  # (B, C, H, W)

            output *= mask

            outputs += output

            # torch.cuda.empty_cache()
            # time.sleep(5)

        # Average over monte carlo steps
        outputs = outputs / monte_carlo_steps # (B, C, H, W)
        return outputs
    

    def forward(self, y, x=None, scale=None, mask=None, corr=None, attenuation_map=None):
        """
        :param y: (B, C, H, W) input sinogram tensor
        :param x: (B, C, H, W) input image tensor to be used when domain is 'image'. If None, reconstruction will be applied to y to get x.
        :param corr: (B, C, H, W) correction sinogram tensor to be subtracted from input sinogram before reconstruction. This can be used to remove estimated scatter or randoms from the input. If None, no correction will be applied.
        :param attenuation_map: (B, C, H, W) attenuation map to be used for the reconstruction operator. If None, no attenuation will be applied.
        :param scale: (B,) scale factor to be applied to sinogram before reconstruction. This is typically acquisition_time * np.log(2) / half_life, but can be set to 1 if the input sinogram has already been scaled accordingly. If None, no scaling will be applied.
        """

        if self.domain == 'image' and x is None:
            x = self.reconstruction(y, scale=scale, corr=corr, attenuation_map=attenuation_map, mode=self.reconstruction_type, **self.reconstruction_config)  # (B, C, H, W)
        elif self.domain == 'photon':
            x = None
        #

        if self.domain == 'photon':
            y = torch.log1p(y)  # log(1+x) is a variance-stabilizing transform for Poisson data that can be more stable than Anscombe for low counts

        if x is not None:

            if hasattr(self, 'use_noise_level_map') and self.use_noise_level_map and self.domain == 'image':
                pet_system_operator = self.get_pet_system_operator()
                inverse_variance_map = pet_system_operator.forward_adjoint(y, attenuation_map=attenuation_map, scale=scale)
                # Compute sensitivity map as backprojection of ones
                sensitivity_map = pet_system_operator.forward_adjoint(torch.ones_like(y), attenuation_map=attenuation_map, scale=scale)
                inverse_variance_map = inverse_variance_map/sensitivity_map
                x = torch.cat([x, inverse_variance_map], dim=1)  # (B, C+2, H, W)
 
            output = super().forward(x)

        else:
            if self.domain == 'photon':
                y = torch.log1p(y)
            output = super().forward(y)
            if self.domain == 'photon':
                output = torch.expm1(output)  # inverse of log1p
        # 
        # Anscombe inverse transform to convert back to original Poisson scale
        if hasattr(self, 'loss_type') and self.loss_type == 'mse_anscombe' and not self.training:
            output = ( (output / 2) ** 2 ) - (3 / 8)
            output = torch.clamp(output, min=0.0)
        #
        if self.domain == 'image' and (output.shape[-2], output.shape[-1]) != self.image_size:
            output = torch.nn.functional.interpolate(output, size=self.image_size, mode='bilinear', align_corners=False)
        #
        if mask is not None:
            output = output * mask
        return output

class Noise2NoiseGradientStepDenoiser(Noise2NoiseBackboneModel):
    """
    Update forward pass to include gradient step denoising
    as introduced by Hureault et al. in "GRADIENT STEP DENOISER FOR CONVERGENT PLUG-AND-PLAY
    DOI : 10.48550/arXiv.2110.03220
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.domain == 'image', "Gradient step denoiser is only applicable for image domain models."

    def forward(self, y, x=None, scale=None, mask=None, corr=None, attenuation_map=None, return_g=False):

        with torch.enable_grad():

            if x is None:
                # It is mandatory to initialize x for gradient tracking.
                x = self.reconstruction(y, scale=scale, corr=corr, attenuation_map=attenuation_map, **self.reconstruction_config)  # (B, C, H, W)
                
            x.requires_grad_(True)
            N_x = super().forward(y, x=x, scale=scale, mask=mask, corr=corr, attenuation_map=attenuation_map)

            g = 0.5 * torch.sum((N_x - x) ** 2, dim=[1,2,3])  # (B,)
            g_sum = g.sum()

            if return_g:
                return g_sum
            grad_g = torch.autograd.grad(g_sum, x, create_graph=self.training)[0]  # (B, C, H, W)

            # Update x with gradient step
            x = x - grad_g

        if mask is not None:
            x = x * mask

        # # softplus
        # x = torch.nn.functional.softplus(x)

        return x