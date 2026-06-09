import pytest, sys
import torch

from pathlib import Path

# Add the noise2noise package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model.unet_noise2noise import Noise2NoisePETModel

class TestUNetNoise2Noise:

    @pytest.mark.parametrize("reconstruction_type", ['fbp', 'mlem'])
    def test_unet_noise2noise(self, reconstruction_type):
        n_angles = 100
        scanner_radius = 100
        voxel_size_mm = (2.0, 2.0)
        sinogram_size = (n_angles, int(2 * scanner_radius / voxel_size_mm[0]))  # (n_angles, n_detectors)
        image_size = (50, 50)
        model = Noise2NoisePETModel(
            reconstruction_type=reconstruction_type,
            image_size=image_size,
            geometry={
                "n_angles": n_angles,
                "scanner_radius_mm": scanner_radius,
                "voxel_size_mm": voxel_size_mm,
            }
        )

        # Create dummy inputs
        sino = torch.randn(2, 1, *sinogram_size)  # (B, C, H, W)
        scale = torch.ones(2)  # (B,)
        corr = torch.ones(2, 1, *sinogram_size)  # (B, 1, H, W)
        att = torch.ones(2, 1, *image_size)  # (B, 1, H, W)

        # Test reconstruction
        output = model.reconstruction(sino, scale=scale, corr=corr, attenuation_map=att, mode=reconstruction_type)
        assert output.shape == (2, 1, *image_size)

if __name__ == "__main__":
    TestUNetNoise2Noise().test_unet_noise2noise(reconstruction_type='fbp')
    TestUNetNoise2Noise().test_unet_noise2noise(reconstruction_type='mlem')