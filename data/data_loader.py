import os
import random
import json, hashlib

import numpy as np
import torch
from torch.utils.data import Dataset

from pet_simulator import PetSystem
from phantom_simulation import Phantom2DPetGenerator

from tools.image.castor import read_castor_binary_file

class SinogramGenerator(Dataset):

    def __init__(
            self,
            dest_path='./',
            length=10,
            image_size=(160,160),
            voxel_size=(2,2,2),
            n_angles=300,
            scanner_radius=300,
            volume_activity=1e3,  # in kBq/ml this is a reasonable pre-computed value for toy simulator
            nb_counts=1e6,
            half_life=109.8*60,
            acquisition_time=None,
            scatter_component=0.36,
            random_component=0.50,
            scatter_sigma=2.0,
            gaussian_PSF=4, # in mm
            seed=None):
        self.dest_path = dest_path
        if not os.path.exists(self.dest_path):
            os.makedirs(self.dest_path)
        self.length = length
        #
        if seed is None:
            self.seed = random.randint(0, 1e32)
        self.seed = seed
        #
        # Simulation parameters
        self.nb_counts = nb_counts
        self.half_life = half_life
        self.acquisition_time = acquisition_time
        self.scatter_component = scatter_component
        self.random_component = random_component
        self.scatter_sigma = scatter_sigma
        self.gaussian_PSF = gaussian_PSF
        #
        self.hashcode = self.get_generator_hashcode()
        #
        self.phantom_generator = Phantom2DPetGenerator(shape=image_size, voxel_size=voxel_size, volume_activity=volume_activity)
        #
        self.sinogram_simulator = PetSystem(
            projector_type='parallelproj_parallel',
            projector_config={
                'scanner_radius_mm': scanner_radius,
                'num_angles': n_angles,
                'img_shape': image_size,
                'voxel_size_mm': voxel_size[:2],
            },
            scatter_component=scatter_component,
            scatter_sigma=scatter_sigma,
            random_component=random_component,
            gaussian_PSF=gaussian_PSF,
            half_life=half_life,
            seed=seed
        )

        #
        if self.acquisition_time is None:
            self.acquisition_time = self.set_acquisition_time(n_samples=100)

    def __len__(self):
        return self.length
    

    def get_generator_hashcode(self):
        dict_to_hash = self.__dict__.copy()
        # remove non hashable items
        dict_to_hash.pop('length', None)
        serialized = json.dumps(dict_to_hash, sort_keys=True, separators=(",", ":"))
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:8]

    def set_acquisition_time(self, n_samples=100):
        """
        Compute the projection for several samples and set the acquisition time accordingly to the target average number of counts.
        param n_samples: number of samples to simulate.
        param half_life: half life of the isotope in seconds. Default is 109.8*60 for F-18.
        """
        print(f"Setting acquisition time over {n_samples} samples to reach {self.nb_counts} counts on average...")
        counts_list = []
        for idx in range(min(n_samples, self.length)):
            # Set phantom generator seed to match sample index
            self.phantom_generator.set_seed(self.seed + idx)
            #
            obj_path, att_path = self.generate_phantom(idx)
            #
            obj = read_castor_binary_file(obj_path).squeeze()
            att = read_castor_binary_file(att_path).squeeze()
            # Simulate true counts
            _ , _, _, noise_free_prompt, _ = self.sinogram_simulator.get_nfpt(obj, att)
            # Get total counts in the noise-free prompt sinogram
            counts = noise_free_prompt.sum()  # in counts
            #
            counts_list.append(counts)

        # get best estimate of counts using histogram average
        hist = np.histogram(counts_list, bins=30)
        max_bin_idx = np.argmax(hist[0])
        bin_edges = hist[1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        avg_counts = bin_centers[max_bin_idx]
        #
        half_life = self.sinogram_simulator.half_life
        acquisition_time = (self.nb_counts / avg_counts) * half_life / np.log(2)
        print(f"Estimated acquisition time: {acquisition_time:.2f} seconds to reach {self.nb_counts} counts on average.")
        # cleanup
        return acquisition_time
    
    def generate_phantom(self, idx):
        # Set Seed
        self.phantom_generator.set_seed(self.seed + idx)
        # Create unique hashcode for data sample with idx and dataset generator hashcode
        data_hashcode = str(idx) + '_' + self.hashcode
        #
        dest_path = os.path.join(self.dest_path, f'data_{data_hashcode}')
        if not os.path.exists(f'{dest_path}/object/object.img'):
            # Generate phantom
            obj_path, att_path = self.phantom_generator.run(os.path.join(self.dest_path, f'data_{data_hashcode}', f'object'))
        else:
            obj_path = f'{dest_path}/object/object'
            att_path = f'{dest_path}/object/object_att'
        #
        return obj_path, att_path

    def simulate_sinogram(self, idx):

        # Set seeds
        torch.manual_seed(self.seed + idx)
        torch.cuda.manual_seed_all(self.seed + idx)
        self.phantom_generator.set_seed(self.seed + idx)
        # Create unique hashcode for data sample with idx and dataset generator hashcode
        data_hashcode = str(idx) + '_' + self.hashcode
        #
        dest_path = os.path.join(self.dest_path, f'data_{data_hashcode}')
        if not os.path.exists(f'{dest_path}/simu/simu_nfpt.s.hdr'):
            # Generate phantom
            obj_path, att_path = self.generate_phantom(idx)
            # Simulate sinogram
            self.sinogram_simulator.run(img_path=obj_path, img_att_path=att_path, dest_path=dest_path, acquisition_time=self.acquisition_time)
        #
        data_prompt, prompt_metadata = read_castor_binary_file(f'{dest_path}/simu/simu_pt.s.hdr', return_metadata=True)
        scale_factor = float(prompt_metadata.get('scale_factor', 1.0)) # Used for reconstruction
        data_prompt = torch.from_numpy(data_prompt)
        #
        data_nfpt = read_castor_binary_file(f'{dest_path}/simu/simu_nfpt.s.hdr')
        data_nfpt = torch.from_numpy(data_nfpt)
        #
        if os.path.exists(f'{dest_path}/object/object.hdr'):
            data_gth = read_castor_binary_file(f'{dest_path}/object/object.hdr')
            data_gth = torch.from_numpy(data_gth)
        else:
            data_gth = None
        #
        if os.path.exists(f'{dest_path}/object/object_att.hdr'):
            data_att = read_castor_binary_file(f'{dest_path}/object/object_att.hdr')
            data_att = torch.from_numpy(data_att)
        else:
            data_att = None
        #
        if os.path.exists(f'{dest_path}/simu/simu_att.s.hdr'):
            data_att_sino = read_castor_binary_file(f'{dest_path}/simu/simu_att.s.hdr')
            data_att_sino = torch.from_numpy(data_att_sino)
        else:
            data_att_sino = None
        #
        randoms = read_castor_binary_file(f'{dest_path}/simu/simu_rd.s.hdr')
        scatter = read_castor_binary_file(f'{dest_path}/simu/simu_sc.s.hdr')
        corr = randoms + scatter
        data_corr = torch.from_numpy(corr)
        #
        return dest_path, data_prompt, data_nfpt, data_gth, data_att, data_att_sino, data_corr, scale_factor


    def generate_sample(self, idx):

        # Simulate sinogram
        dest_path, prompt, nfpt, gth, att, att_sino, data_corr, scale_factor = self.simulate_sinogram(idx)

        return dest_path, prompt, nfpt, gth, att, att_sino, data_corr, scale_factor

    def __getitem__(self, idx):
        """
        Output shape : (1, H, W)
        """

        dest_path, prompt, nfpt, gth, att, att_sino, data_corr, scale_factor = self.generate_sample(idx)
        return dest_path, prompt, nfpt, gth, att, att_sino, data_corr, scale_factor

class SinogramGeneratorSavedImages(SinogramGenerator):
    """
    Used to generate sinogram from pre-saved object images, for testing purposes.
    """

    def __init__(self, obj_path, att_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_path = obj_path
        self.att_path = att_path
        if not isinstance(obj_path, (list, tuple)):
            self.obj_path = (obj_path,)
        if not isinstance(att_path, (list, tuple)):
            self.att_path = (att_path,)
        #
        assert len(self.obj_path) == len(self.att_path), "obj_path and att_path should have the same length"
        #
        for idx in range(len(self)):
            assert os.path.exists(self.obj_path[idx]), f"Object path {self.obj_path[idx]} does not exist"
            assert os.path.exists(self.att_path[idx]), f"Attenuation map path {self.att_path[idx]} does not exist"
        #
        assert hasattr(self, "nb_counts"), "nb_counts attribute should be defined in the parent class"

    def __len__(self):
        return len(self.obj_path)

    def generate_phantom(self, idx):
        return self.obj_path[idx], self.att_path[idx]
    
    def get_sinogram_simulator(self, idx):
        """Simulation parameters may change as test images may have different sizes and voxel sizes."""
        data_gth, metadata = read_castor_binary_file(self.obj_path[idx], return_metadata=True)
        data_gth = data_gth.squeeze()
        voxel_size = (float(metadata.get('scaling factor (mm/pixel) [1]', 2.0)), float(metadata.get('scaling factor (mm/pixel) [2]', 2.0)))
        if data_gth.shape != self.sinogram_simulator.proj.img_shape or voxel_size != self.sinogram_simulator.proj.voxel_size_mm:
            # Re-initialize sinogram simulator with new image size and voxel size
            self.sinogram_simulator = PetSystem(
                projector_type='parallelproj_parallel',
                projector_config={
                    'scanner_radius_mm': self.sinogram_simulator.scanner_radius_mm,
                    'num_angles': self.sinogram_simulator.num_angles,
                    'img_shape': data_gth.shape,
                    'voxel_size_mm': voxel_size,
                },
                scatter_component=self.scatter_component,
                scatter_sigma=self.scatter_sigma,
                random_component=self.random_component,
                gaussian_PSF=self.gaussian_PSF,
                half_life=self.half_life,
                seed=self.seed
            )
    
    def set_acquisition_time(self, idx):
        # Read Image ground truth
        data_gth = read_castor_binary_file(self.obj_path[idx]).squeeze()
        data_att = read_castor_binary_file(self.att_path[idx]).squeeze()
        # Simulate true counts
        _ , _, _, noise_free_prompt, _ = self.sinogram_simulator.get_nfpt(data_gth, data_att)
        # Get total counts in the noise-free prompt sinogram
        counts = noise_free_prompt.sum()  # in counts
        # Compute acquisition time to reach desired counts (nb_counts)
        half_life = self.sinogram_simulator.half_life
        self.acquisition_time = (self.nb_counts / counts) * half_life / np.log(2)
        return self.acquisition_time
    
    def simulate_sinogram(self, idx):

        self.get_sinogram_simulator(idx)

        self.set_acquisition_time(idx)
        #
        dest_path, data_prompt, data_nfpt, data_gth, data_att, data_att_sino, data_corr, scale_factor = super().simulate_sinogram(idx)
        #
        data_gth = read_castor_binary_file(self.obj_path[idx])
        data_gth = torch.from_numpy(data_gth)
        #
        data_att = read_castor_binary_file(self.att_path[idx])
        data_att = torch.from_numpy(data_att)
        #
        return dest_path, data_prompt, data_nfpt, data_gth, data_att, data_att_sino, data_corr, scale_factor


class SinogramGeneratorReconstructionTest(SinogramGenerator):

    """
    This generator is quite different in the sense that it is aimed to evaluate post-denoising reconstruction.
    Therefore, it only generates one noisy sinogram and the clean ground truth image (not clean sinogram).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def generate_sample(self, idx):

        # Simulate sinogram
        dest_path, data_nfpt = self.simulate_sinogram(idx)
        
        # Read Image ground truth
        with open(f'{dest_path}/object/object.img', 'rb') as f:
            object_gth = torch.frombuffer(f.read(), dtype=torch.float32)
            object_gth = object_gth.reshape(self.image_size)

        # Generate 1 Poisson noisy version
        data_noisy = torch.poisson(data_nfpt)

        return dest_path, data_noisy, data_nfpt, object_gth

    def __getitem__(self, idx):

        dest_path, noisy, sinogram_clean, image_clean = self.generate_sample(idx)
        return dest_path, noisy, sinogram_clean, image_clean        

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt


    dest_path = os.path.join(os.getenv("WORKSPACE"), "data", "test")
    dataset = SinogramGenerator(dest_path=dest_path, length=2, seed=42, scanner_radius=300)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, (dest_path, prompt, nfpt, gth, att, att_sino, corr, scale) in enumerate(loader):
        pass

    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(prompt.squeeze(), cmap='gray')
    ax[0].set_title('Prompt')
    ax[1].imshow(nfpt.squeeze(), cmap='gray')
    ax[1].set_title('Noisy 2')
    ax[2].imshow(gth.squeeze(), cmap='gray')
    ax[2].set_title('Clean')
    plt.show()
