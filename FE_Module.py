'''
FE Module: Deployed Feature Enhancement Mechanism, which is a part ofthe paper FEGAN,
the details like equations and parameters are discussed in the paper.
'''
import cv2
import numpy as np
import os
from tqdm import tqdm  # For progress bar

class FEMechanism:
    def __init__(self, weight_multiplier=2, noise_threshold=15, integration_factor=0.3):
        """
        Initialize the Feature Enhancement mechanism
        
        Args:
            weight_multiplier (float): μL value for edge enhancement (default: 2)
            noise_threshold (float): Δnoise value for thresholding (default: 15)
            integration_factor (float): αi value for feature integration (default: 0.3)
        """
        self.weight_multiplier = weight_multiplier
        self.noise_threshold = noise_threshold
        self.integration_factor = integration_factor
        
        # Laplacian kernel for feature extraction
        self.laplacian_kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)

    def feature_extraction(self, image):
        """
        FX Module: Extract features using Laplacian operator
        
        Args:
            image (numpy.ndarray): Input LR thermal image
            
        Returns:
            numpy.ndarray: Image with extracted features
        """
        # Convert image to float32 for processing
        image_float = image.astype(np.float32)
        
        # Apply Laplacian convolution
        features = cv2.filter2D(image_float, -1, self.laplacian_kernel)
        
        return features

    def edge_enhancement(self, features):
        """
        EE Module: Enhance edges using weighted multiplication and thresholding
        
        Args:
            features (numpy.ndarray): Features extracted from FX module
            
        Returns:
            numpy.ndarray: Enhanced edge features
        """
        # Multiply features by weight multiplier (Equation 1)
        S_ij = self.weight_multiplier * features
        
        # Apply thresholding criteria (Equation 2)
        I_E = np.zeros_like(S_ij)
        
        # Apply conditions from Equation 2
        mask1 = (S_ij >= self.noise_threshold) & (S_ij <= 255)
        mask2 = S_ij > 255
        
        I_E[mask1] = S_ij[mask1]
        I_E[mask2] = 255
        
        return I_E

    def feature_integration(self, original_image, enhanced_features):
        """
        FI Module: Combine original image with enhanced features
        
        Args:
            original_image (numpy.ndarray): Original LR thermal image
            enhanced_features (numpy.ndarray): Enhanced features from EE module
            
        Returns:
            numpy.ndarray: Final enhanced thermal image
        """
        # Apply Equation 3
        enhanced_image = original_image - self.integration_factor * enhanced_features
        
        # Ensure pixel values are within valid range [0, 255]
        enhanced_image = np.clip(enhanced_image, 0, 255)
        
        return enhanced_image.astype(np.uint8)

    def enhance_image(self, image):
        """
        Apply the complete FE mechanism to an input image
        
        Args:
            image (numpy.ndarray): Input LR thermal image
            
        Returns:
            numpy.ndarray: Enhanced thermal image
            dict: Dictionary containing intermediate results
        """
        # Store intermediate results
        results = {}
        
        # 1. Feature Extraction (FX)
        features = self.feature_extraction(image)
        results['features'] = features
        
        # 2. Edge Enhancement (EE)
        enhanced_edges = self.edge_enhancement(features)
        results['enhanced_edges'] = enhanced_edges
        
        # 3. Feature Integration (FI)
        final_image = self.feature_integration(image, enhanced_edges)
        results['final_image'] = final_image
        
        return final_image, results

def process_thermal_image(image_path, save_results=False):
    """
    Process a thermal image using the FE mechanism
    
    Args:
        image_path (str): Path to the input thermal image
        save_results (bool): Whether to save intermediate results
        may be used for testing different parameters

    Returns:
        numpy.ndarray: Enhanced thermal LR image
    """
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Create FE mechanism instance
    fe_mechanism = FEMechanism()
    
    # Process the image
    enhanced_image, results = fe_mechanism.enhance_image(image)
    
    # Save results if requested
    if save_results:
        cv2.imwrite('original.png', image)
        cv2.imwrite('features.png', results['features'])
        cv2.imwrite('enhanced_edges.png', results['enhanced_edges'])
        cv2.imwrite('enhanced_final.png', enhanced_image)
    
    return enhanced_image

def process_directory(input_dir, output_dir=None, save_intermediate=False):
    """
    Process all images in a directory using the FE mechanism
    
    Args:
        input_dir (str): Path to directory containing input thermal images
        output_dir (str, optional): Path to save enhanced images. If None, 
                                  will create 'enhanced_' + input_dir
        save_intermediate (bool): Whether to save intermediate results for each image
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = 'enhanced_' + os.path.basename(input_dir.rstrip('/'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        raise ValueError(f"No valid image files found in {input_dir}")
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Construct full input path
            input_path = os.path.join(input_dir, image_file)
            
            # Construct output path
            output_path = os.path.join(output_dir, f"enhanced_{image_file}")
            
            # Process the image
            enhanced_image = process_thermal_image(input_path, 
                                                save_results=save_intermediate)
            
            # Save the enhanced image
            cv2.imwrite(output_path, enhanced_image)
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

# Example usage
if __name__ == "__main__":
    try:
        # Process all images in the lr_ directory
        input_directory = "lr_"
        output_directory = "enhanced_lr"
        
        process_directory(input_directory, 
                        output_directory, 
                        save_intermediate=False)
        
        print("Image enhancement completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
