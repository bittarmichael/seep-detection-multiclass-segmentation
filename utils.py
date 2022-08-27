import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import torch

class ImageHelperFunctions:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def image_properties(image) -> None:
        image = np.array(image)
        print(f"image type: {type(image)}")
        print(f"image shape: {image.shape}")
        
    @staticmethod
    def preview_images(list_images) -> None:
        fig, axs = plt.subplots(1, len(list_images), figsize=(10,10))
        for i, image in enumerate(list_images):
            axs[i].imshow(image)
            axs[i].axis('off')
        plt.show()
    
    @staticmethod
    def count_classes_in_arr(arr, show=False):
        _arr = np.array(arr)
        counts = np.unique(_arr, return_counts=True)
        print(f"classes: {counts[0]}")
        print(f"counts: {counts[1]}")
        
        if show:
            plt.bar(counts[0][1:], counts[1][1:])
            plt.show()
        return counts[0], counts[1]
        
    @staticmethod
    def find_mu_and_std(loader):
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        for data, _ in loader:
            channels_sum += torch.mean(data, dim = [0, 2, 3])
            channels_squared_sum += torch.mean(data ** 2, dim = [0, 2, 3])
            num_batches += 1
            
        mean = channels_sum / num_batches
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
        
        return mean, std
    
    @staticmethod
    def list_files_in_dir(_dir, ext):
        return [f"{_dir}{x}" for x in os.listdir(_dir) if ext in x]
    
    @staticmethod
    def read_image(image_path):
        image = Image.open(image_path)       
        return image
    
    @staticmethod
    def read_mask(mask_path):
        mask = Image.open(mask_path)
        return mask
    
    
class ImageTransformFunctions:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def mask_to_onehot(mask):
        classes = [0, 1, 2, 3, 4, 5, 6, 7]
        mask = np.asanyarray(mask)
        
        _mask = [mask == i for i in classes]
        mask = np.array(_mask).astype(np.uint8)
        return mask
    
    @staticmethod
    def onehot_to_mask(onehot):
        array = np.argmax(onehot, axis=0)
        array = array.astype(np.uint8)
        return Image.fromarray(array, 'L')

    @staticmethod
    def palette():
        '''
        Returns palette (used in 'P' mode of image representation)
        '''
        colors_dict = {
            0: [0, 0, 0],       # 0 = background: black
            1: [255, 0, 0],     # 1 = seep class 1: red
            2: [255, 127, 0],   # 2: orange 
            3: [255, 255, 0],   # 3: yellow
            4: [0, 255, 0],     # 4: green
            5: [0, 0, 255],     # 5: blue
            6: [46, 43, 95],    # 6: dark blue
            7: [139, 0, 255],   # 7: purple
            }
        palette = []
        for i in np.arange(256):
            if i in colors_dict:
                palette.extend(colors_dict[i])
            else:
                palette.extend([0, 0, 0])
        return palette
    
    @staticmethod
    def mask_to_palette(mask):
        """
        Converts mask to P-mode image
        """
        mask = mask.convert('P')
        mask.putpalette(ImageTransformFunctions.palette())
        return mask
    
    @staticmethod
    def normalize(image, mu, std):
        im = np.array(image)
        im = (im - mu) / std
        return im
    
    @staticmethod
    def inv_normalize(image, mu, std):
        im = np.squeeze(image)
        im = im * 65535
        im = im * std + mu
        
        im = im.astype('uint16')
        im = Image.fromarray(im, 'I;16')
        return im
    
    @staticmethod
    def augment(image, mask):
        rotate = transforms.RandomRotation(180)
        angle = rotate.get_params(rotate.degrees)
        
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
        
        if np.random.uniform() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            
        return image, mask
    
    @staticmethod
    def PILtoTensor(pil_image):
        convert = transforms.ToTensor()
        return convert(pil_image)
            
        