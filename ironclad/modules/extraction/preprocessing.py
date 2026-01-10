import torch
from torchvision import transforms
from PIL import Image


class Preprocessing:
    """
    Preprocessing pipeline for images before feeding them to the model.

    This class provides a default preprocessing transformation pipeline.

    Attributes:
        image_size (int): The target size (height and width) for the image.
        preprocess_transform (transforms.Compose): The default transform pipeline.
        device (str): The device used for tensor operations ('cuda' or 'cpu').
    """

    def __init__(self, image_size: int = 160):
        """
        Initialize the Preprocessing pipeline.

        Parameters:
            image_size (int): The target size for the image. Default is 160.
        """
        self.image_size = image_size  # E.g., 160 or 224 depending on model requirements
        self.preprocess_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def process(self, probe: Image.Image) -> torch.Tensor:
        """
        Preprocess the input image and prepare it for model inference.

        The image is resized, normalized, and converted to a tensor, then moved to the appropriate device.

        Parameters:
            probe (PIL.Image): The input image.

        Returns:
            torch.Tensor: The preprocessed image tensor with an added batch dimension.
        """
        tensor = self.preprocess_transform(probe)
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor


# Example usage
if __name__ == "__main__":
    image_size = 160
    image_path = "simclr_resources/probe/Alan_Ball/Alan_Ball_0002.jpg"
    preprocessing = Preprocessing(image_size=image_size)
    
    # Open the image
    probe_image = Image.open(image_path)
    
    # Process image through the default pipeline
    processed_tensor = preprocessing.process(probe_image)
    print("Processed tensor shape:", processed_tensor.shape)
