"""
FEGAN

This implementation performs super-resolution on thermal images using a GAN architecture.
Input images are resized to:
- Ground truth HR images: 256 × 256 pixels
- LR images: 64 × 64 pixels

The model architecture consists of:
- A generator with residual blocks and upsampling
- A discriminator following SRGAN architecture
- VGG19 for feature extraction and perceptual loss
"""

# Configuration parameters dictionary
params = {
    # Dataset parameters
    'n_samples': 10000,        # Number of training images to use
    'batch_size': 32,          # Batch size for training
    'test_split': 0.33,        # Fraction of data used for testing
    'random_seed': 42,         # Random seed for reproducibility
    
    # Model parameters
    'num_res_blocks': 24,      # Number of residual blocks in generator
    'init_filters': 64,        # Initial number of filters in conv layers
    'kernel_size': 3,          # Kernel size for most conv layers
    'large_kernel': 9,         # Kernel size for first and last conv layers
    'leaky_alpha': 0.2,        # Slope for LeakyReLU
    'bn_momentum': 0.5,        # Momentum for batch normalization
    
    # Training parameters
    'epochs': 150,             # Number of training epochs
    'version': 1,              # Model version identifier
    'learning_rate': 0.0001,   # Learning rate for Adam optimizer
    'beta_1': 0.9,            # Beta 1 for Adam optimizer
    'beta_2': 0.999,          # Beta 2 for Adam optimizer
    'epsilon': 1e-08,         # Epsilon for Adam optimizer
    'decay': 0.000004,        # Learning rate decay
    
    # Loss weights
    'adv_loss_weight': 1.5e-3, # Weight for adversarial loss
    'content_loss_weight': 0.95, # Weight for content (VGG) loss
    
    # Image dimensions
    'hr_size': 256,           # Size of high-resolution images
    'lr_size': 64,            # Size of low-resolution images
    'channels': 3,            # Number of color channels
    
    # Paths
    'lr_path': "",       # Path to low-resolution images
    'hr_path': "",        # Path to high-resolution images
    'model_save_path': "gen{}/gen_e_{}.h5"  # Model save path format
}

# Add these to the params dictionary at the beginning of the file
params.update({
    # Discriminator parameters
    'disc_init_filters': 32,   # Initial number of filters in discriminator
    'disc_bn_momentum': 0.8,   # Batch normalization momentum for discriminator
    
    # VGG parameters
    'vgg_layer': 10,          # Which VGG layer to use for feature extraction
    
    # Model saving parameters
    'save_frequency': 1,      # How often to save the model (in epochs)
    
    # Visualization parameters
    'plot_size': (16, 8),     # Figure size for plots
    
    # Additional paths
    'test_image_path': "",    # Path to test image (LR)
    'test_image_hr_path': "" # Path to test image (HR)
})

import os
import cv2
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras import layers, Model
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Conv2D, PReLU, BatchNormalization, Flatten
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add
from tqdm import tqdm
from keras import backend as K

# Configure GPU and optimizer settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
keras.optimizers.Adam(lr=params['learning_rate'], 
                     beta_1=params['beta_1'], 
                     beta_2=params['beta_2'], 
                     epsilon=params['epsilon'], 
                     decay=params['decay'])


# Define blocks to build the generator
def res_block(ip):
    """
    Implements a residual block with two convolutional layers and skip connection.
    
    Args:
        ip: Input tensor
    Returns:
        Tensor after residual block processing
    """
    res_model = Conv2D(64, (3, 3), padding="same")(ip)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    res_model = PReLU(shared_axes=[1, 2])(res_model)

    res_model = Conv2D(64, (3, 3), padding="same")(res_model)
    res_model = BatchNormalization(momentum=0.5)(res_model)

    return add([ip, res_model])


def upscale_block(ip):
    """
    Implements an upscaling block that doubles the spatial dimensions.
    
    Args:
        ip: Input tensor
    Returns:
        Upscaled tensor with 2x spatial dimensions
    """
    up_model = Conv2D(256, (3, 3), padding="same")(ip)
    up_model = UpSampling2D(size=2)(up_model)
    up_model = PReLU(shared_axes=[1, 2])(up_model)

    return up_model


# Generator model
def create_gen(gen_ip, num_res_block):
    """
    Creates the generator model with residual blocks and upsampling layers.
    
    Args:
        gen_ip: Input tensor for the generator
        num_res_block: Number of residual blocks to use
    Returns:
        Keras Model instance representing the generator
    """
    layers = Conv2D(64, (9, 9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1, 2])(layers)

    temp = layers

    for i in range(num_res_block):
        layers = res_block(layers)

    layers = Conv2D(64, (3, 3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = add([layers, temp])

    layers = upscale_block(layers)
    layers = upscale_block(layers)

    op = Conv2D(3, (9, 9), padding="same")(layers)

    return Model(inputs=gen_ip, outputs=op)


# Descriminator block that will be used to construct the discriminator
def discriminator_block(ip, filters, strides=1, bn=True):
    """
    Basic building block for the discriminator network.
    
    Args:
        ip: Input tensor
        filters: Number of filters in convolution
        strides: Stride length for convolution
        bn: Boolean flag for batch normalization
    Returns:
        Processed tensor after convolution, optional batch norm, and LeakyReLU
    """
    disc_model = Conv2D(filters, (3, 3), strides=strides, padding="same")(ip)

    if bn:
        disc_model = BatchNormalization(momentum=0.8)(disc_model)

    disc_model = LeakyReLU(alpha=0.2)(disc_model)

    return disc_model


# Descriminartor, as described in the original paper
def create_disc(disc_ip):
    """
    Creates the discriminator model following SRGAN architecture.
    Uses multiple discriminator blocks with increasing filter sizes.
    
    Args:
        disc_ip: Input tensor for discriminator
    Returns:
        Keras Model instance representing the discriminator
    """
    df = params['disc_init_filters']

    d1 = discriminator_block(disc_ip, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df * 2)
    d4 = discriminator_block(d3, df * 2, strides=2)
    d5 = discriminator_block(d4, df * 4)
    d6 = discriminator_block(d5, df * 4, strides=2)
    d7 = discriminator_block(d6, df * 8)
    d8 = discriminator_block(d7, df * 8, strides=2)
    d9 = discriminator_block(d8, df * 16)
    d10 = discriminator_block(d9, df * 16, strides=2)

    d10_5 = Flatten()(d10)
    d11 = Dense(df * 16)(d10_5)
    d12 = LeakyReLU(alpha=0.2)(d11)
    validity = Dense(1, activation='sigmoid')(d12)

    return Model(disc_ip, validity)


def charbonnier(y_true, y_pred):
    """
    Implements Charbonnier loss function, a differentiable variant of L1 loss.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    Returns:
        Computed Charbonnier loss
    """
    epsilon = 1e-3

    error = y_true - y_pred

    p = K.sqrt(K.square(error) + K.square(epsilon))

    return K.mean(p)

from keras.applications import VGG19


def build_vgg(hr_shape):
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)

    return Model(inputs=vgg.inputs, outputs=vgg.layers[params['vgg_layer']].output)


# Combined model
def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)

    gen_features = vgg(gen_img)

    disc_model.trainable = False
    validity = disc_model(gen_img)

    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])

"""
Data Loading and Preprocessing Section
"""

# Load training images
lr_list = os.listdir("lr_LT")[:params['n_samples']]  # Load first n low-resolution images

# Load and preprocess low-resolution images
lr_images = []
for img in lr_list:
    img_lr = cv2.imread("lr_LT/" + img, cv2.IMREAD_COLOR)
    img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB color space
    lr_images.append(img_lr)

# Load and preprocess high-resolution images
hr_list = os.listdir("hr_L")[:params['n_samples']]
hr_images = []
for img in hr_list:
    img_hr = cv2.imread("hr_L/" + img, cv2.IMREAD_COLOR)
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB color space
    hr_images.append(img_hr)

# Convert lists to numpy arrays for efficient processing
lr_images = np.array(lr_images)
hr_images = np.array(hr_images)

# Visualize random image pair for verification
image_number = random.randint(0, len(lr_images) - 1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Low Resolution Input (64x64)')
plt.imshow(np.reshape(lr_images[image_number], (64, 64, 3)))
plt.subplot(122)
plt.title('High Resolution Target (256x256)')
plt.imshow(np.reshape(hr_images[image_number], (256, 256, 3)))
plt.show()

# Normalize pixel values to range [0, 1]
lr_images = lr_images / 255.
hr_images = hr_images / 255.

# Split dataset into training and testing sets
lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images,
                                                        test_size=params['test_split'], random_state=params['random_seed'])

"""
Model Creation and Compilation Section
"""

# Define input shapes for both streams
hr_shape = (hr_train.shape[1], hr_train.shape[2], hr_train.shape[3])  # 256x256x3
lr_shape = (lr_train.shape[1], lr_train.shape[2], lr_train.shape[3])  # 64x64x3

# Create input layers
lr_ip = Input(shape=lr_shape)
hr_ip = Input(shape=hr_shape)

# Initialize generator with 24 residual blocks
generator = create_gen(lr_ip, num_res_block=params['num_res_blocks'])
generator.summary()

# Initialize and compile discriminator
discriminator = create_disc(hr_ip)
discriminator.compile(loss="binary_crossentropy", 
                     optimizer="adam", 
                     metrics=['accuracy'])
discriminator.summary()

# Initialize VGG19 for feature extraction
vgg = build_vgg((256, 256, 3))
print(vgg.summary())
vgg.trainable = False  # Freeze VGG19 weights

# Create and compile the combined GAN model
gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)

# Compile GAN with combined loss:
# 1. Adversarial loss (binary_crossentropy)
# 2. Content loss (MSE between VGG feature maps)
gan_model.compile(loss=["binary_crossentropy", "mse"], 
                 loss_weights=[params['adv_loss_weight'], params['content_loss_weight']],  # Weight factors for each loss
                 optimizer="adam")
gan_model.summary()

"""
Training Data Preparation
"""

# Create batches of training data
train_lr_batches = []
train_hr_batches = []
for it in range(int(hr_train.shape[0] / params['batch_size'])):
    start_idx = it * params['batch_size']
    end_idx = start_idx + params['batch_size']
    train_hr_batches.append(hr_train[start_idx:end_idx])
    train_lr_batches.append(lr_train[start_idx:end_idx])

"""
Training Loop Section
"""

# Main training loop over epochs
for e in range(params['epochs']):
    # Initialize labels for GAN training
    fake_label = np.zeros((params['batch_size'], 1))  # Labels for generated (fake) images
    real_label = np.ones((params['batch_size'], 1))   # Labels for real images

    # Lists to track losses during training
    g_losses = []  # Generator losses
    d_losses = []  # Discriminator losses

    # Training loop over batches with progress bar
    for b in tqdm(range(len(train_hr_batches))):
        # Get current batch of images
        lr_imgs = train_lr_batches[b]  # Low-resolution input images
        hr_imgs = train_hr_batches[b]  # High-resolution target images

        # Generate fake (super-resolved) images
        fake_imgs = generator.predict_on_batch(lr_imgs)

        # Train discriminator on both real and fake images
        discriminator.trainable = True
        d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_label)   # Loss on generated images
        d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)    # Loss on real images

        # Train generator (with discriminator weights frozen)
        discriminator.trainable = False
        
        # Calculate average discriminator loss
        d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)

        # Extract VGG features for perceptual loss
        image_features = vgg.predict(hr_imgs)

        # Train generator through combined model
        # Uses both adversarial and content (VGG) losses
        g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], 
                                               [real_label, image_features])

        # Store losses for averaging
        d_losses.append(d_loss)
        g_losses.append(g_loss)

    # Calculate average losses for current epoch
    g_losses = np.array(g_losses)
    d_losses = np.array(d_losses)
    g_loss = np.sum(g_losses, axis=0) / len(g_losses)
    d_loss = np.sum(d_losses, axis=0) / len(d_losses)

    # Print progress report
    print("epoch:", e + 1, "g_loss:", g_loss, "d_loss:", d_loss)

    # Save generator model periodically
    if (e + 1) % params['save_frequency'] == 0:
        generator.save(params['model_save_path'].format(params['version'], e + 1))

"""
Model Testing and Evaluation Section
"""

# Load the trained generator model
from keras.models import load_model
from numpy.random import randint

model_path = params['model_save_path'].format(params['version'], params['epochs'])
generator = load_model(model_path, compile=False)

# Test on random sample from test set
[X1, X2] = [lr_test, hr_test]
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]

# Generate super-resolved image
gen_image = generator.predict(src_image)

# Visualize results
plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('LR Image')
plt.imshow(src_image[0, :, :, :])
plt.subplot(232)
plt.title('Superresolution')
plt.imshow(gen_image[0, :, :, :])
plt.subplot(233)
plt.title('Orig. HR image')
plt.imshow(tar_image[0, :, :, :])
plt.show()

"""
Additional Test Case Section
"""

# Test on external image
sreeni_lr = cv2.imread(params['test_image_path'])
sreeni_hr = cv2.imread(params['test_image_hr_path'])

# Preprocess test images
sreeni_lr = cv2.cvtColor(sreeni_lr, cv2.COLOR_BGR2RGB)  # Convert color space
sreeni_hr = cv2.cvtColor(sreeni_hr, cv2.COLOR_BGR2RGB)

# Normalize pixel values
sreeni_lr = sreeni_lr / 255.
sreeni_hr = sreeni_hr / 255.

# Add batch dimension
sreeni_lr = np.expand_dims(sreeni_lr, axis=0)
sreeni_hr = np.expand_dims(sreeni_hr, axis=0)

# Generate super-resolved image
generated_sreeni_hr = generator.predict(sreeni_lr)

# Visualize results
plt.figure(figsize=params['plot_size'])
plt.subplot(231)
plt.title('LR Image')
plt.imshow(sreeni_lr[0, :, :, :])
plt.subplot(232)
plt.title('Superresolution')
plt.imshow(generated_sreeni_hr[0, :, :, :])
plt.subplot(233)
plt.title('Orig. HR image')
plt.imshow(sreeni_hr[0, :, :, :])
plt.show()
