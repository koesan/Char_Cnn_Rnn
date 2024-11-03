import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

# GoogleNet modelini yükle ve özellik çıkarma modunu ayarla
model = models.googlenet(pretrained=True, transform_input=True)
model = torch.nn.Sequential(*list(model.children())[:-2])  # Son iki katmanı çıkar
model.eval()

# Görüntü ön işleme
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #  transforms.CenterCrop(image_size) kaggledeki bir projedek ullanılıyor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) kagledeki bir projede kullanılan değerler
])

def create_crops(image):
    crops = []
    width, height = image.size
    crop_size = min(width, height)
    
    # Orta kırpma
    crops.append(transforms.CenterCrop(crop_size)(image))
    
    # Köşe kırpmalar
    crops.append(image.crop((0, 0, crop_size, crop_size)))  # Sol üst
    crops.append(image.crop((width - crop_size, 0, width, crop_size)))  # Sağ üst
    crops.append(image.crop((0, height - crop_size, crop_size, height)))  # Sol alt
    crops.append(image.crop((width - crop_size, height - crop_size, width, height)))  # Sağ alt
    
    # Yatay çevirme ve aynı kırpmaları uygula
    flipped_image = transforms.RandomHorizontalFlip(p=1.0)(image)
    crops.append(transforms.CenterCrop(crop_size)(flipped_image))
    crops.append(flipped_image.crop((0, 0, crop_size, crop_size)))
    crops.append(flipped_image.crop((width - crop_size, 0, width, crop_size)))
    crops.append(flipped_image.crop((0, height - crop_size, crop_size, height)))
    crops.append(flipped_image.crop((width - crop_size, height - crop_size, width, height)))
    
    return crops

def process_all_classes(data_folder):
    for class_folder_name in sorted(os.listdir(data_folder)):  # Klasörleri sıralı şekilde al
        class_folder = os.path.join(data_folder, class_folder_name)
        if os.path.isdir(class_folder):  # Sadece klasörleri işlemek için
            features_list = []
            for image_file in sorted(os.listdir(class_folder)):  # Dosyaları sıralı şekilde al
                if image_file.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(class_folder, image_file)
                    image = Image.open(image_path)
                    crops = create_crops(image)
                    print(image_path)
                    for crop in crops:
                        crop_tensor = preprocess(crop).unsqueeze(0)
                        with torch.no_grad():
                            feature = model(crop_tensor).squeeze().numpy()
                        features_list.append(feature)
            
            # Özellikleri 60 x 10 x 1024 formatında kaydet
            num_images = len(features_list) // 10  # Toplam görüntü sayısı
            features_tensor = torch.tensor(features_list).view(num_images, 10, 1024)  # 60 x 10 x 1024 formatında yeniden şekillendir
            features_tensor = features_tensor.permute(0, 2, 1)  # 60 x 1024 x 10 formatına çevir
            
            # Çıkış dosyasını sınıf adına göre adlandır
            output_file = os.path.join(data_folder, f"{class_folder_name}.t7")
            torch.save(features_tensor, output_file)
            print(f"{class_folder_name} için özellikler kaydedildi: {output_file}")

# Örnek kullanım
data_folder = '/home/koesan/Videolar/text_to_image_project/dataset (copy)/images'  # Ana klasör yolunu belirleyin
process_all_classes(data_folder)
