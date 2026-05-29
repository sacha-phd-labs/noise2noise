from pytorcher.models import UNet

from pytorcher.pet_system import PetSystem

import torch
from torch import nn

import hashlib


from pytorcher.utils import tensor_hash, iradon


class UNetNoise2NoisePET(UNet):

    """
    U-Net model for Noise2Noise.
    If outputs are in the photon/Poisson domain using 'mse_anscombe' loss, apply the rescale stage at inference time.
    """

    def __init__(self, *args,
                 input_domain='image', output_domain='image',
                 physics='backward_pet_radon', 
                 physics_mode='pre_inverse',
                 sinogram_size=(300, 300),
                 geometry={},
                 reconstruction_type='fbp',
                 reconstruction_config={},
                 image_size=(160,160),
                 n_splits=2,
                 **kwargs):
        #
        self.input_domain = input_domain
        self.output_domain = output_domain
        #
        self.n_angles = geometry.get('n_angles', 300)
        self.scanner_radius = geometry.get('scanner_radius_mm', 300)
        self.gaussian_PSF = geometry.get('gaussian_PSF_fwhm_mm', 4.0)
        self.voxel_size_mm = geometry.get('voxel_size_mm', 2.0)
        #
        #
        self.image_size = image_size
        self.sinogram_size = sinogram_size
        self.n_splits = n_splits
        self.reconstruction_type = reconstruction_type
        self.reconstruction_config = reconstruction_config
        assert self.reconstruction_type.lower() in ['fbp', 'mlem'], "Currently only FBP and MLEM are supported."
        #
        # must call nn.Module init before assigning any nn.Module attributes such as done in init_pet_system_operator and get_reconstruction_operator
        super(UNetNoise2NoisePET, self).__init__(*args, **kwargs)
        
    def get_pet_system_operator(self):

        if not hasattr(self, 'pet_system_operator'):
            geometry = {
                'num_angles':self.n_angles,
                'scanner_radius_mm':self.scanner_radius,
                'voxel_size_mm':self.voxel_size_mm
            }
            self.pet_system_operator = PetSystem(
                projector_type='parallelproj_parallel',
                projector_config=geometry,
                gaussian_PSF=self.gaussian_PSF,
                device=next(self.parameters()).device
            )

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
            splits_denoised = self.forward(y=splitted_prompts, scale=scale.repeat(self.n_splits), mask=mask, corr=corr, attenuation_map=attenuation_map)  # (B * n_splits, C, H, W)

            # Expand splits to have (n_splits, B, C, H, W)
            splits_denoised = torch.chunk(splits_denoised, self.n_splits, dim=0)  # list of (B, C, H, W)

            # Apply reconstruction if needed and average outputs
            if self.unet_output_domain == 'photon':
                output = [ self.reconstruction(split_denoised_, scale=scale, corr=corr, attenuation_map=attenuation_map, mode=self.reconstruction_type, **self.reconstruction_config) for split_denoised_ in splits_denoised ]  # (B, C, H, W)
            else:
                splits_denoised = torch.stack(splits_denoised, dim=0)  # (n_splits, B, C, H, W)
                output = torch.mean(splits_denoised, dim=0)  # (B, C, H, W)

            outputs += output

            # torch.cuda.empty_cache()
            # time.sleep(5)

        # Average over monte carlo steps
        outputs = outputs / monte_carlo_steps # (B, C, H, W)
        return outputs
    
    def del_unpickable_attributes(self):
        # Remove attributes that cannot be pickled (e.g. for MLFlow model registering)
        if hasattr(self, 'forward_pet_radon_operator'):
            del self.forward_pet_radon_operator

    def forward(self, x, scale=None, mask=None, corr=None, attenuation_map=None):
        """
        :param x: either a batch of sinograms (B, C, H, W) if input_domain is 'photon', or a batch of images if input_domain is 'image'.
        :param x_domain: 'photon' or 'image', only needed if the domain of x is different from self.input_domain and we need to apply the reconstruction operator. If None, it will be inferred from the shape of x and self.input_domain.
        :param attenuation_map: (B, C, H, W) attenuation map to be used for the reconstruction operator. If None, no attenuation will be applied.
        :param scale: (B,) scale factor to be applied to sinogram before reconstruction. This is typically acquisition_time * np.log(2) / half_life, but can be set to 1 if the input sinogram has already been scaled accordingly. If None, no scaling will be applied.
        """

        if self.input_domain == 'photon' and self.output_domain == 'image':
            x = self.reconstruction(x, scale=scale, corr=corr, attenuation_map=attenuation_map, mode=self.reconstruction_type, **self.reconstruction_config)  # (B, C, H, W)
        #

        if self.unet_output_domain == self.unet_input_domain == 'photon':
            x = torch.log1p(x)  # log(1+x) is a variance-stabilizing transform for Poisson data that can be more stable than Anscombe for low counts

        output = super().forward(x)

        if self.unet_output_domain == 'photon' and self.unet_input_domain == 'photon':
            output = torch.expm1(output)  # inverse of log1p
        # 
        # Anscombe inverse transform to convert back to original Poisson scale
        if hasattr(self, 'loss_type') and self.loss_type == 'mse_anscombe' and not self.training:
            output = ( (output / 2) ** 2 ) - (3 / 8)
            output = torch.clamp(output, min=0.0)
        #
        if self.output_domain == 'image' and (output.shape[-2], output.shape[-1]) != self.image_size:
            output = torch.nn.functional.interpolate(output, size=self.image_size, mode='bilinear', align_corners=False)
        #
        if mask is not None:
            output = output * mask
        return output