# **Char-Cnn-Rnn**

Bu projenin amacı, kendi veri setimizi kullanarak Char-CNN-RNN modelini geliştirmek ve verilerimizi gömmektir (embedding).

***

## **Gereksinimler**

pip3 install pytorch==2.4.0 torchvision==0.19.0 pillow==10.4.0 tqdm==4.66.5 

***

## **kullanım**

Öncelikle txt ve img verileri bununduğu bir veri seti oluşturmalısınız veri setinin yapısı aşağıdaki formatta olamlıdır(Resimler .jpg .png ,jpeg olabilir).

***

### **Veri Seti Klasör Yapısı**

```
dataset/
├── text/
│   ├── class1/
│   │   ├── dosya1.txt
│   │   ├── dosya2.txt
│   │   └── ...
│   ├── class2/
│   │   ├── dosya1.txt
│   │   ├── dosya2.txt
│   │   └── ...
│   ├── class3/
│   │   ├── dosya1.txt
│   │   ├── dosya2.txt
│   │   └── ...
└── images/
    ├── class1/
    │   ├── dosya1.jpg
    │   ├── dosya1.jpg
    │   └── ...
    ├── class2/
    │   ├── dosya1.jpg
    │   ├── dosya1.jpg
    │   └── ...
    ├── class3/
    │   ├── dosya1.jpg
    │   ├── dosya1.jpg
    │   └── ...
```

### **Görsel verilerin hazırlanması**

Görüntü verileri, **[Learning Deep Representations of Fine-grained Visual Descriptions Paper](https://arxiv.org/pdf/1605.05395)** makalesinin 5. bölümünde belirtildiği gibi hazırlandı. Öncelikle, her görsel toplamda 10 parçaya ayrılacak. Bu parçalar, her görselin sol üst, sol alt, sağ üst, sağ alt ve orta kısımlarının kırpılmasıyla elde edilecektir. Ardından, görsel yatay çevrilmesiyle aynı işlemler tekrar edilerek toplamda 10 görsel elde edilecektir.

Elde edilen görseller, GoogleNet kullanılarak 1024 boyutunda özellik vektörlerine dönüştürülecektir. Bu özellik çıkarımı sürecinde, her görsel için kırpılan parçalar üzerinde işlem yapılacak ve sonuçta elde edilen çıktılar, `60 (görsel sayısı) x 1024 (özellik vektörü) x 10 (parça sayısı)` boyutunda `.t7` formatında kaydedilecektir. Her sınıf için bir tane `.t7` dosyası oluşturulacaktır.

Görsel ön işleme için `img2t7.py` dosyasındaki 72. satıra, veri setinizdeki image dosyanızın yolunu girin ve ardından çalıştırın.

```
python3 img2t7.py
```

### **Metin Verilerinin Hazırlanması**

Metin verileri, **[Learning Deep Representations of Fine-grained Visual Descriptions Paper](https://arxiv.org/pdf/1605.05395)** makalesinin 5. bölümünde belirtildiği gibi hazırlandı. İlk olarak, her bir `.txt` dosyası satır satır okunacak ve her bir `.txt` dosyasında toplamda 10 satır bulunması gerekmektedir. Ardından, okunan satırlar 201 boyutundaki karakterlere ayrılacak şekilde işlenecektir. Eğer bir satır 201 karakterden daha uzun ise, fazla karakterler silinecek; eğer 201 karakterden daha kısa ise, eksik olan kısımlar sıfırlar ile doldurulacaktır.

Her bir karaktere bir sayısal değer atanacak şekilde işlem yapılacak ve bu sayede karakter verileri sayısal formata dönüştürülecektir. Son olarak, işlenen tüm `.txt` dosyaları ve satırları bir araya getirilerek, `60 (txt sayısı) x 201 (karakter sayısı) x 10 (satır sayısı)` boyutlarında tek bir `.t7` dosyasına kaydedilecektir. Her bir sınıf için de ayrı bir `.t7` dosyası oluşturulacaktır.

Metin ön işleme için `txt2t7.py` dosyasındaki 49. satıra, veri setinizdeki text dosyalanızın yolunu girin ve ardından çalıştırın.

```
python3 txt2t7.py
```

### **modelin eğitimi**

Görsel ve metin dosyaları hazırlandıktan sonra, her bir sınıf için elinizde bir tane `.t7` dosyası olması gerekiyor.

Klasör yapısının aşağıdaki gibi olması gerekmektedir:

```
dataset/
├── text/
│   ├── class1/
│   │   ├── dosya1.txt
│   │   ├── dosya2.txt
│   │   └── ...
│   ├── class2/
│   │   ├── dosya1.txt
│   │   ├── dosya2.txt
│   │   └── ...
│   ├── class3/
│   │   ├── dosya1.txt
│   │   ├── dosya2.txt
│   │   └── ...
│   ├── class1.t7
│   ├── class2.t7
│   ├── class3.t7
└── images/
    ├── class1/
    │   ├── dosya1.jpg
    │   ├── dosya1.jpg
    │   └── ...
    ├── class2/
    │   ├── dosya1.jpg
    │   ├── dosya1.jpg
    │   └── ...
    ├── class3/
    │   ├── dosya1.jpg
    │   ├── dosya1.jpg
    │   └── ...
    ├── class1.t7
    ├── class2.t7
    ├── class3.t7
```

Modeli eğitmek için aşağıdaki kodu proje klasörünün içinde çalıştırın. `data_dir` kısmına veri setinizin yolunu girin.

```
python3 sje_train.py --seed 123 --use_gpu True --dataset birds --model_type cvpr --data_dir "file path" --train_split trainval --learning_rate 0.0007 --symmetric True --epochs 200 --checkpoint_dir ckpt --save_file sje_cub_c10_hybrid
```

Eğitim bittikten sonra, modeliniz `ckpt` klasörünün içinde olacaktır. Eğittiğiniz modeli test edebilmek için aşağıdaki kodu çalıştırın. `data_dir` kısmına veri setinin adresini ve `model_path` kısmına eğittiğiniz modelin adresini ekleyin.

```
python3 sje_eval.py --seed 123 --use_gpu True --dataset birds --model_type cvpr --data_dir "file path" --eval_split test --num_txts_eval 0 --print_class_stats True --batch_size 40 --model_path "file path"
```

***

## **Değerlendirme**

![Screenshot from 2024-10-26 07-28-30](https://github.com/user-attachments/assets/202b9301-cbef-4afb-8772-26e1252a59ff)


***

## **kaynaklar**

[Char-CNN-RNN for PyTorch GitHub](https://github.com/martinduartemore/char_cnn_rnn_pytorch/tree/master)

[char-CNN-RNN GitHub](https://github.com/1o0ko/char-CNN-RNN)

[charCnnRnn_embedding GitHub](https://github.com/ramidzamzam/charCnnRnn_embedding/tree/main)

[cvpr2016 GitHub](https://github.com/reedscot/cvpr2016)

[icml2016 GitHub](https://github.com/reedscot/icml2016)

[Generative Adversarial Text to Image Synthesis Paper](https://arxiv.org/abs/1605.05396)

[Learning Deep Representations of Fine-grained Visual Descriptions Paper](https://arxiv.org/pdf/1605.05395)

