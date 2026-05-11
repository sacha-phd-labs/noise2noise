import matplotlib.pyplot as plt
import numpy as np

from tools.image.castor import read_castor_binary_file
    
def get_brain_figure(img, magnification_factor=2, vmin=0, vmax=230, annotations={}):
    """
    Format the brain image in a matplotlib figure for visualization
    """
    img = img.squeeze()
    print(img.max(), img.min())

    # Invert intensities
    img = vmax - img + vmin
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin) * 255

    # img_min = img.min()
    # img_max = img.max()
    # img = img_max - img + img_min  # Invert intensities while preserving the original min and max values
    # img = (img - img_min) / (img_max - img_min) * 255  # Normalize to [0, 255]
    
    img = img.astype(np.uint8)

    # Flip the image to match the orientation of the original data
    img = np.flip(img, axis=0)
    img = np.flip(img, axis=1)

    # Crop the image to focus on the brain region
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    crop_size = 55  # Adjust this size as needed
    img = img[center_y - crop_size:center_y + crop_size, center_x - crop_size:center_x + crop_size]

    # Shift the brain to the left and to the bottom
    shift_x = -img.shape[1] // 10
    shift_y = img.shape[0] // 100
    img = np.roll(img, shift_x, axis=1)
    img = np.roll(img, shift_y, axis=0)

    # Get ROI square mask around tumour
    x_min, x_max = 25, 32,
    y_min, y_max = 77, 84
    # Add some padding to the ROI
    padding_x = (x_max - x_min) * 2
    padding_y = (y_max - y_min) * 2
    padding = max(padding_x, padding_y) // 2  # Adjust padding as
    x_min = max(x_min - padding, 0)
    x_max = min(x_max + padding, img.shape[1] - 1)
    y_min = max(y_min - padding, 0)
    y_max = min(y_max + padding, img.shape[0] - 1)
    roi_mask = np.zeros_like(img, dtype=bool)
    roi_mask[y_min:y_max+1, x_min:x_max+1] = True

    # Get frontier mask around tumour
    frontier_mask = np.zeros_like(img, dtype=bool)
    frontier_mask[y_min-1:y_max+2, x_min-1:x_max+2] = True
    frontier_mask = np.logical_xor(frontier_mask, roi_mask)

    # Convert to RGB scale
    img_rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img_rgb[..., 0] = img  # Red channel
    img_rgb[..., 1] = img  # Green channel
    img_rgb[..., 2] = img  # Blue channel
    img = img_rgb

    # Highlight the ROI frontierin red
    img[frontier_mask] = [105, 98, 201]  # Red color for ROI

    # Get magnified view of the tumour region
    size_ROI = np.sqrt(np.logical_or(frontier_mask, roi_mask).sum()).astype(int)
    mask_ROI = np.logical_or(frontier_mask, roi_mask)
    magnified_img = np.zeros((size_ROI, size_ROI, 3), dtype=np.uint8)
    magnified_img[..., 0] = img[mask_ROI, 0].reshape(size_ROI, size_ROI)  # Red channel
    magnified_img[..., 1] = img[mask_ROI, 1].reshape(size_ROI, size_ROI)  # Green channel
    magnified_img[..., 2] = img[mask_ROI, 2].reshape(size_ROI, size_ROI)  # Blue channel
    magnified_img = np.kron(magnified_img, np.ones((magnification_factor, magnification_factor, 1), dtype=np.uint8)) 

    # Display the magnified version in the top left corner of the original image
    img[0:size_ROI*magnification_factor, img.shape[1]-size_ROI*magnification_factor:img.shape[1]] = magnified_img

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.axis('off')

    # set font to serif latex
    plt.rcParams["font.family"] = "serif"

    # add annotations if provided
    content = ""
    for key, value in annotations.items():
        key = key.upper()
        content += f"{key}: {value:2.2f}"
        if key == "PSNR":
            content += " dB"
        content += "\n"
    content = content.strip()  # Remove the last newline
    ax.text(0.99, 0.01, content, transform=ax.transAxes, fontsize=18, verticalalignment='bottom', horizontalalignment='right', color='black', )#bbox=dict(facecolor='black'))
    return fig

if __name__ == "__main__":

    import os
    
    brain_data_path = os.path.join(os.getenv('WORKSPACE'), 'data/brain_web_phantom/object/gt_web_after_scaling')
    brain_data = read_castor_binary_file(brain_data_path)


    fig = get_brain_figure(brain_data, magnification_factor=2, annotations={'SSIM': 0.987654321, 'PSNR': 25.675651985})
    plt.savefig(os.path.join(os.getenv('WORKSPACE'), 'data/brain_web_phantom/object/brain_figure.png'), bbox_inches='tight')