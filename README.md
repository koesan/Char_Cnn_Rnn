# Char-Cnn-Rnn

Bu projenin amacı, kendi veri setimizi kullanarak Char-CNN-RNN modelini geliştirmek ve verilerimizi gömmektir (embedding).

## Gereksinimler



## kullanım:

Öncelikle txt ve img verileri bununduğu bir veri seti oluşturmalısınız veri setinin yapısı aşağıdaki formatta olamlıdır(Resimler .jpg .png ,jpeg olabilir).

### Veri Seti Klasör Yapısı

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

### Görsel verilerin hazırlanması

Görüntü verileri, **[Learning Deep Representations of Fine-grained Visual Descriptions Paper](https://arxiv.org/pdf/1605.05395)** makalesinin 5. bölümünde belirtildiği gibi hazırlandı. Öncelikle, her görsel toplamda 10 parçaya ayrılacak. Bu parçalar, her görselin sol üst, sol alt, sağ üst, sağ alt ve orta kısımlarının kırpılmasıyla elde edilecektir. Ardından, görsel yatay çevrilmesiyle aynı işlemler tekrar edilerek toplamda 10 görsel elde edilecektir.

Elde edilen görseller, GoogleNet kullanılarak 1024 boyutunda özellik vektörlerine dönüştürülecektir. Bu özellik çıkarımı sürecinde, her görsel için kırpılan parçalar üzerinde işlem yapılacak ve sonuçta elde edilen çıktılar, `60 (görsel sayısı) x 1024 (özellik vektörü) x 10 (parça sayısı)` boyutunda `.t7` formatında kaydedilecektir. Her sınıf için bir tane `.t7` dosyası oluşturulacaktır.

### Metin Verilerinin Hazırlanması

Metin verileri, **[Learning Deep Representations of Fine-grained Visual Descriptions Paper](https://arxiv.org/pdf/1605.05395)** makalesinin 5. bölümünde belirtildiği gibi hazırlandı. İlk olarak, her bir `.txt` dosyası satır satır okunacak ve her bir `.txt` dosyasında toplamda 10 satır bulunması gerekmektedir. Ardından, okunan satırlar 201 boyutundaki karakterlere ayrılacak şekilde işlenecektir. Eğer bir satır 201 karakterden daha uzun ise, fazla karakterler silinecek; eğer 201 karakterden daha kısa ise, eksik olan kısımlar sıfırlar ile doldurulacaktır.

Her bir karaktere bir sayısal değer atanacak şekilde işlem yapılacak ve bu sayede karakter verileri sayısal formata dönüştürülecektir. Son olarak, işlenen tüm `.txt` dosyaları ve satırları bir araya getirilerek, `60 (txt sayısı) x 201 (karakter sayısı) x 10 (satır sayısı)` boyutlarında tek bir `.t7` dosyasına kaydedilecektir. Her bir sınıf için de ayrı bir `.t7` dosyası oluşturulacaktır.

### modelin eğitimi

görsel ve metin dosyaları hazırlandıktan sora görsel ve metin verileri için elinizde her bir sınıf için 1 tane t7 dosyası olması gerekiyor.

klasör yapısı aşağıdaki gibi olması gerkiyor.
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

ardından modeli eğitmek için aşağıdaki kodu proje klasörünün içinde iken çalıştırın(data_dir kısmına veri setinizin adresini girin)
```
python3 sje_train.py --seed 123 --use_gpu True --dataset birds --model_type cvpr --data_dir "file path" --train_split trainval --learning_rate 0.0007 --symmetric True --epochs 200 --checkpoint_dir ckpt --save_file sje_cub_c10_hybrid
```

kaynaklar:

[Char-CNN-RNN for PyTorch GitHub](https://github.com/martinduartemore/char_cnn_rnn_pytorch/tree/master)

[char-CNN-RNN GitHub](https://github.com/1o0ko/char-CNN-RNN)

[charCnnRnn_embedding GitHub](https://github.com/ramidzamzam/charCnnRnn_embedding/tree/main)

[cvpr2016 GitHub](https://github.com/reedscot/cvpr2016)

[icml2016 GitHub](https://github.com/reedscot/icml2016)

[Generative Adversarial Text to Image Synthesis Paper](https://arxiv.org/abs/1605.05396)

[Learning Deep Representations of Fine-grained Visual Descriptions Paper](https://arxiv.org/pdf/1605.05395)

