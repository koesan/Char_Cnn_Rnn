import os
import torch

def str_to_labelvec(string, max_str_len):
    string = string.lower()
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~+-=<>()[]{} "
    alpha_to_num = {k: v + 1 for v, k in enumerate(alphabet)}
    labels = torch.zeros(max_str_len).long()
    max_i = min(max_str_len, len(string))
    for i in range(max_i):
        labels[i] = alpha_to_num.get(string[i], alpha_to_num[' ']) 

    return labels

def process_txt_files(data_folder, max_str_len=201):

    for subfolder_name in sorted(os.listdir(data_folder)):
        subfolder_path = os.path.join(data_folder, subfolder_name)
        if os.path.isdir(subfolder_path):  # Sadece klasörleri işle
            all_labels = []
            
            for filename in sorted(os.listdir(subfolder_path)):
                if filename.endswith('.txt'):
                    file_path = os.path.join(subfolder_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        labels_for_file = []
                        
                        for line in lines:
                            label_vec = str_to_labelvec(line.strip(), max_str_len)
                            labels_for_file.append(label_vec)

                        all_labels.append(labels_for_file)
            
            # 60 (txt dosyası) x 201 (karakter sayısı) x 10 (satır sayısı) şeklinde tensor oluştur
            all_labels_tensor = torch.zeros(len(all_labels), max_str_len, 10).long()
            
            for i, file_labels in enumerate(all_labels):
                for j, label_vec in enumerate(file_labels[:10]):  # Sadece ilk 10 satırı al
                    all_labels_tensor[i, :, j] = label_vec
            
            # Çıkış dosyasını alt klasör adına göre adlandır ve ana klasöre kaydet
            output_file = os.path.join(data_folder, f"{subfolder_name}.t7")
            torch.save(all_labels_tensor, output_file)
            print(f"{subfolder_name} için tensor dosyası kaydedildi: {output_file}")

data_folder = '/home/koesan/Videolar/text_to_image_project/dataset (copy)/text'  # Ana klasör yolunu güncelleyin
process_txt_files(data_folder)
