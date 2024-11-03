import os
import torch
from char_cnn_rnn_pytorch.char_cnn_rnn import char_cnn_rnn, prepare_text

# Modeli Tanımla ve Ağırlıkları Yükle
def load_model(model_path, dataset='birds', model_type='cvpr'):
    model = char_cnn_rnn(dataset=dataset, model_type=model_type)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Metin Satırlarını Embedding'e Dönüştürme
def get_text_embeddings(model, text_lines):
    embeddings = []
    for line in text_lines:
        txt_tensor = prepare_text(line).unsqueeze(0).float()
        with torch.no_grad():
            embedding = model(txt_tensor)
        embeddings.append(embedding)
    return torch.vstack(embeddings)  # (n, 1024) boyutunda embedding döner

# Ana İşlev: Alt klasörlerdeki tüm .txt dosyalarını işleyip .t7 dosyalarını kaydet
def process_text_files_in_directory(root_dir, model, save_format='t7'):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_path = os.path.join(subdir, file)

                # .txt dosyasındaki her satırı oku
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text_lines = f.readlines()

                # (n, 1024) boyutunda metin embedding'leri
                txt_embeddings = get_text_embeddings(model, text_lines)

                # Verileri kaydet
                output_path = os.path.join(subdir, file.replace('.txt', f'.{save_format}'))
                torch.save(txt_embeddings, output_path)
                print(f"{file} dosyasından embedding oluşturuldu ve {output_path} olarak kaydedildi.")

# Model dosyasının yolu
model_path = '/home/koesan/Videolar/text_to_image_project/char_cnn_rnn_pytorch/ckpt/sje_cub_c10_hybrid/sje_cub_c10_hybrid_0.00070_True_trainval_2024_10_28_18_01_47.pth'

# Modeli Yükle
model = load_model(model_path, dataset='birds', model_type='cvpr')

# Ana klasör yolunu belirle
root_dir = '/home/koesan/Videolar/text_to_image_project/dataset/text'

# Tüm txt dosyalarını işle
process_text_files_in_directory(root_dir, model)
