import os
from PIL import Image
import random


angle = [90, 180, 270] 
def oversample_images(directory):
    # Get list of image files in the directory
    images = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]
    index = 0
    while True:
        index+=1
        # Select a random image from the directory
        image_path = os.path.join(directory, random.choice(images))
        
        # Open the image
        img = Image.open(image_path)
        
        # Rotate the image by a random angle
        rotated_img = img.rotate(random.choice(angle))
        
        # Save the new image
        new_image_path = os.path.join(directory, f'new_{len(images)}.jpg')
        rotated_img.save(new_image_path)
        
        # Add the new image to the list of images
        images.append(new_image_path)
    
        if index == 500:
            break 
oversample_images('D://Khoa/Cancer_Recognition/Types/healthy/')
