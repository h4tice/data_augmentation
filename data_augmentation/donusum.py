import os
import cv2
from rotate import yoloRotatebbox
from helpers import cvFormattoYolo

def rotate_dataset(data_dir, output_dir, angle):
    """
    Tüm veri kümesini belirli bir açıya göre döndürüp yeni dosyalara kaydeder.
    
    Args:
        data_dir (str): Görüntü ve etiket dosyalarının bulunduğu ana dizin.
        output_dir (str): Döndürülmüş görüntü ve etiketlerin kaydedileceği dizin.
        angle (int): Döndürme açısı (derece cinsinden).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # İşlenecek görüntü dosya uzantıları
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(valid_extensions):
            image_name = os.path.join(data_dir, filename.split('.')[0])
            image_ext = '.' + filename.split('.')[-1]

            # yoloRotatebbox sınıfını kullanarak döndürme işlemini gerçekleştir
            rotator = yoloRotatebbox(image_name, image_ext, angle)
            
            # Etiketleri ve görüntüyü döndür 
            rotated_bbox = rotator.rotateYolobbox()
            rotated_image = rotator.rotate_image()
            
            # Yeni dosya adlarını belirle
            rotated_image_filename = os.path.join(output_dir, filename.split('.')[0] + f"aci_{angle}{image_ext}")
            rotated_label_filename = os.path.join(output_dir, filename.split('.')[0] + f"aci_{angle}.txt")
            
            # Döndürülmüş görüntüyü kaydet
            cv2.imwrite(rotated_image_filename, rotated_image)
            
            # Döndürülmüş etiketleri kaydet
            with open(rotated_label_filename, 'w') as fout:
                for bbox in rotated_bbox:
                    fout.write(' '.join(map(str, cvFormattoYolo(bbox, rotated_image.shape[0], rotated_image.shape[1]))) + '\n')

# Kullanım örneği
if __name__ == "__main__":
    rotate_dataset(
        data_dir='/valid/images2',       # Orijinal veri kümesinin bulunduğu dizin
        output_dir='/valid/toplu/',      # Döndürülmüş veri kümesinin kaydedileceği dizin
        angle=330                                   # Döndürme açısı (örneğin, 90 derece)
    )
