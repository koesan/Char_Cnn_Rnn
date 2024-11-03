import os
import torch
import numpy as np
import char_cnn_rnn_pytorch.char_cnn_rnn as ccr

# Modeli yükleme
model_path = "/home/koesan/Videolar/text_to_image_project/char_cnn_rnn_pytorch/ckpt/sje_cub_c10_hybrid/sje_cub_c10_hybrid_0.00070_True_trainval_2024_10_26_07_13_57.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ccr.char_cnn_rnn("birds", "cvpr")
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Metin açıklamaları için embedding hesaplama
def get_text_embedding(text):
    with torch.no_grad():
        embedding = model(text)  # Metin embedding'ini hesaplama
    return embedding.cpu().numpy()

# Ana veri setini kaydetme fonksiyonu
def save_cub_icml_dataset(text_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for class_folder in sorted(os.listdir(text_dir)):
        class_text_path = os.path.join(text_dir, class_folder)
        
        if os.path.isdir(class_text_path):
            for txt_file in sorted(os.listdir(class_text_path)):
                if txt_file.endswith('.txt'):
                    txt_file_path = os.path.join(class_text_path, txt_file)
                    
                    with open(txt_file_path, 'r', encoding='utf-8') as file:
                        text = file.read().strip()  # Açıklama metnini oku
                        embedding = get_text_embedding(text)
                        
                        # Çıktı dosyası ismini .txt dosyasının adıyla aynı yap
                        output_file_name = os.path.splitext(txt_file)[0] + ".t7"
                        output_file_path = os.path.join(output_dir, class_folder, output_file_name)
                        
                        # Sınıf klasörünü oluştur
                        os.makedirs(os.path.join(output_dir, class_folder), exist_ok=True)
                        
                        # Embedding'i .t7 dosyası olarak kaydet
                        torch.save(torch.tensor(embedding), output_file_path)
                        print(f"{txt_file} dosyası için embedding {output_file_name} olarak kaydedildi.")

# Kullanım örneği
text_folder = '/home/koesan/Videolar/text_to_image_project/dataset/text'
output_folder = '/home/koesan/Videolar/text_to_image_project/dataset/embeddings'

save_cub_icml_dataset(text_folder, output_folder)
