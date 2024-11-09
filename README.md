# FEGAN
FEGAN(Feature Enhanced GAN-based Super Resolution for Thermal Image)

The code was only tested on a single NVIDIA RTX A5000 graphics processing unit, Ubuntu 18.04 operating system, TensorFlow-GPU version 2.10.0 

Trained utilizing thermal images sourced from the Teledyne FLIR ADAS dataset.

For this study, both the ground truth thermal images and the training and testing thermal images underwent resizing to dimensions of 256 × 256 and 64 × 64, respectively.

## How to use:
1. Download the thermal image dataset
2. Resize the original thermal images to 256 × 256 as the HR image
3. Downscaling the HR image to 64 × 64 as the LR image (some downsampling techniques like Gaussian pyramid construction or Laplacian pyramid construction can be used. Ensuring get more realistic low-resolution images can help improve model performance.)
4. Using the FE module to enhance the LR image
5. Adjust the parameters in the FE module to satisfy your requirements
6. Using the enhanced LR image and the HR image as the dataset for the FEGAN training
7. Split the dataset into Training, Validation and Testing parts
8. Train the FEGAN model

## Citing FEGAN:
L. Zhu, R. Wu, B. -G. Lee, L. Nkenyereye, W. -Y. Chung and G. Xu, "FEGAN: A Feature-Oriented Enhanced GAN for Enhancing Thermal Image Super-Resolution," in IEEE Signal Processing Letters, vol. 31, pp. 541-545, 2024, doi: 10.1109/LSP.2024.3356751.
keywords: {Training;Feature extraction;Iron;Image reconstruction;Image edge detection;Imaging;Measurement;Generative adversarial network (GAN);super-resolution;feature enhancement;thermal imaging},

@ARTICLE{10411026,
  author={Zhu, Linzhen and Wu, Renjie and Lee, Boon-Giin and Nkenyereye, Lionel and Chung, Wan-Young and Xu, Gen},
  journal={IEEE Signal Processing Letters}, 
  title={FEGAN: A Feature-Oriented Enhanced GAN for Enhancing Thermal Image Super-Resolution}, 
  year={2024},
  volume={31},
  number={},
  pages={541-545},
  keywords={Training;Feature extraction;Iron;Image reconstruction;Image edge detection;Imaging;Measurement;Generative adversarial network (GAN);super-resolution;feature enhancement;thermal imaging},
  doi={10.1109/LSP.2024.3356751}}
