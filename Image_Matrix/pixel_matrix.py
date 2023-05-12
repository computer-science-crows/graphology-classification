import os
from PIL import Image
import numpy as np

def get_pixel_matrix(img_path):
    img = Image.open(img_path)
    img_gray = img.convert('L')
    return img_gray

def images_pixel_matrix():
    input_folder_path = "D:/UniVerSiDaD/IV Ano/Machine Learning/Project/Graphology classification/graphology-classification/Images/photo-cropped"
    folders_names = os.listdir(input_folder_path)
    print(folders_names)

    output_images_path = "D:/UniVerSiDaD/IV Ano/Machine Learning/Project/Graphology classification/graphology-classification/Image_Matrix/Img_npy1"
    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)
        print("Directorio creado: ", output_images_path)

    count = 62
    for folder_name in folders_names:

        folder_image_path = input_folder_path + "/" + folder_name
        folders_images_names = os.listdir(folder_image_path)
        print(folder_image_path)

        for i, folder_image_name in enumerate(folders_images_names):
            images_paths = folder_image_path + '/' + folder_image_name
            images_names = os.listdir(images_paths)

            for j, image_name in enumerate(images_names):

                image_path = images_paths + '/' + image_name
                pixel_matrix = get_pixel_matrix(image_path)

                if not os.path.exists(f'D:/UniVerSiDaD/IV Ano/Machine Learning/Project/Graphology classification/graphology-classification/Image_Matrix/Img_npy1/{count}'):
                    os.makedirs(f'D:/UniVerSiDaD/IV Ano/Machine Learning/Project/Graphology classification/graphology-classification/Image_Matrix/Img_npy1/{count}')
                
                np.save(f'D:/UniVerSiDaD/IV Ano/Machine Learning/Project/Graphology classification/graphology-classification/Image_Matrix/Img_npy1/{count}/{j}.npy', pixel_matrix)

            count += 1

images_pixel_matrix()