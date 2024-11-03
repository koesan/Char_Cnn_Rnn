
# **Char-CNN-RNN**

Bu projenin amacı, kendi veri setimizi kullanarak Char-CNN-RNN modelini geliştirmek ve metinsel verilerimizi gömmektir.

Model, text-image eşleme ve text-to-image gibi ayrıntılı açıklama gerektiren görevlerde oldukça başarılıdır.

### Neden Char-CNN-RNN?
Char-CNN-RNN modeli, görsel olarak benzer nesneleri tanımak için yüksek detay ve hassasiyet gerektiren durumlarda kullanılır (örneğin, farklı kuş türleri). Karakter seviyesindeki CNN ve RNN yapısı, açıklamalardaki karakter sıralamasını öğrenerek modeli yazım hatalarına karşı dayanıklı hale getirir. Bu özellikleriyle sıfırdan öğrenme (zero-shot learning) senaryolarında başarılıdır.

### Nasıl Çalışır?
Char-CNN-RNN modeli, metin ve görselleri ortak bir özellik alanında kodlayarak iki modalite arasında güçlü bir eşleşme sağlar. Bu sayede, daha önce görmediği görseller için sadece metin açıklamalarına dayanarak doğru tahminler yapabilir.

### Geleneksel Gömme Modellerine Göre Avantajları
Word2Vec veya GloVe gibi kelime bazlı gömme modellerinden farklı olarak, Char-CNN-RNN:
- Karakterleri doğrudan işler, bu sayede kelime formlarındaki varyasyonlara dayanıklıdır.
- Bağlamı dinamik olarak öğrenir, bu da benzer sınıfları ayırt etme yeteneğini güçlendirir.
- Yazım hatalarına karşı dayanıklıdır ve sabit bir kelime haznesine bağlı kalmadan daha esnek bir şekilde çalışır.

---

## **Gereksinimler**

```
pip3 install pytorch==2.4.0 torchvision==0.19.0 pillow==10.4.0 tqdm==4.66.5 
```

---

## **Kullanım**

Öncelikle `txt` ve `img` verilerinin bulunduğu bir veri seti oluşturmalısınız. Veri setinin yapısı aşağıdaki formatta olmalıdır. Veri seti tek sınıflı ve çok sınıflı olabilir ve resimler `.jpg`, `.png`, `.jpeg` olabilir.

Projede kullanılan veri seti [TXT dosyaları text_c10 klasöründe](https://drive.google.com/file/d/0B0ywwgffWnLLZW9uVHNjb2JmNlE/view?resourcekey=0-8y2UVmBHAlG26HafWYNoFQ), [İMG dosyaları ımage klasöründe]
---

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
    │   ├── dosya2.jpg
    │   └── ...
    ├── class2/
    │   ├── dosya1.jpg
    │   ├── dosya2.jpg
    │   └── ...
    ├── class3/
    │   ├── dosya1.jpg
    │   ├── dosya2.jpg
    │   └── ...
```

### **Görsel Verilerin Hazırlanması**

Görüntü verileri, **[Learning Deep Representations of Fine-grained Visual Descriptions Paper](https://arxiv.org/pdf/1605.05395)** makalesinin 5. bölümünde belirtildiği gibi hazırlandı. 

Öncelikle, her görsel toplamda 10 parçaya ayrılacak. Bu parçalar, her görselin sol üst, sol alt, sağ üst, sağ alt ve orta kısımlarının kırpılmasıyla elde edilecektir. Ardından, görsel yatay çevrilmesiyle aynı işlemler tekrar edilerek toplamda 10 görsel elde edilir.

Elde edilen görseller, GoogleNet kullanılarak 1024 boyutunda özellik vektörlerine dönüştürülür. Bu özellik çıkarımı sürecinde, her görsel için kırpılan parçalar üzerinde işlem yapılacak ve sonuçta elde edilen çıktılar, 60 (görsel sayısı) x 1024 (özellik vektörü) x 10 (parça sayısı) boyutunda `.t7` formatında kaydedilir. Her sınıf için bir tane `.t7` dosyası oluşturulacak.

Görsel ön işleme için `img2t7.py` dosyasındaki 72. satıra, veri setinizdeki `image` dosyanızın yolunu girin ve ardından çalıştırın.

```bash
python3 img2t7.py
```

### **Metin Verilerinin Hazırlanması**

Metin verileri, **[Learning Deep Representations of Fine-grained Visual Descriptions Paper](https://arxiv.org/pdf/1605.05395)** makalesinin 5. bölümünde belirtildiği gibi hazırlandı. 

İlk olarak, her bir `.txt` dosyası satır satır okunacak ve her bir `.txt` dosyasında toplamda 10 satır bulunması gerekmektedir. Ardından, okunan satırlar 201 boyutundaki karakterlere ayrılacak şekilde işlenir. Eğer bir satır 201 karakterden daha uzun ise, fazla karakterler silinir; eğer 201 karakterden daha kısa ise, eksik olan kısımlar sıfırlarla doldurulur.

Her bir karaktere bir sayısal değer atanacak şekilde işlem yapılır ve bu sayede karakter verileri sayısal formata dönüştürülür. Son olarak, işlenen tüm `.txt` dosyaları ve satırları bir araya getirilerek, 60 (txt sayısı) x 201 (karakter sayısı) x 10 (satır sayısı) boyutlarında tek bir `.t7` dosyasına kaydedilir. Her bir sınıf için de ayrı bir `.t7` dosyası oluşturulacak.

Metin ön işleme için `txt2t7.py` dosyasındaki 49. satıra, veri setinizdeki `text` dosyalarının yolunu girin ve ardından çalıştırın.

```bash
python3 txt2t7.py
```

### **Modelin Eğitimi**

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
    │   ├── dosya2.jpg
    │   └── ...
    ├── class2/
    │   ├── dosya1.jpg
    │   ├── dosya2.jpg
    │   └── ...
    ├── class3/
    │   ├── dosya1.jpg
    │   ├── dosya2.jpg
    │   └── ...
    ├── class1.t7
    ├── class2.t7
    ├── class3.t7
```

Modeli eğitmek için aşağıdaki kodu proje klasörünün içinde çalıştırın. `data_dir` kısmına veri setinizin yolunu girin.

Eğer çok sınıflı bir model eğitecekseniz `sje_train.py` dosyasının 41. satırında `MultimodalDataset` olmalı. Tek bir sınıf varsa 41. satırda `SinglemodalDataset` olarak değiştirin.

```bash
python3 sje_train.py --seed 123 --use_gpu True --dataset birds --model_type cvpr --data_dir "file path" --train_split trainval --learning_rate 0.0007 --symmetric True --epochs 200 --checkpoint_dir ckpt --save_file sje_cub_c10_hybrid
```

Eğitim bittikten sonra, modeliniz `ckpt` klasörünün içinde olacaktır. Eğittiğiniz modeli test edebilmek için aşağıdaki kodu çalıştırın. `data_dir` kısmına veri setinin adresini ve `model_path` kısmına eğittiğiniz modelin adresini ekleyin.

```bash
python3 sje_eval.py --seed 123 --use_gpu True --dataset birds --model_type cvpr --data_dir "file path" --eval_split test --num_txts_eval 0 --print_class_stats True --batch_size 40 --model_path "file path"
```
---

![Screenshot from 2024-10-26 07-28-30](https://github.com/user-attachments/assets/202b9301-cbef-4afb-8772-26e1252a59ff)

---

## ** Metin gömmeleri oluşturmak**

** Önceden eğitilmiş modelleri indirmek için **
[cvpr](https://github.com/reedscot/cvpr2016)
[icml](https://github.com/reedscot/icml2016)

Eğitilmiş model ile metinleri gömmek için `Text_embedding.py` dosyasının 42. satırında yer alan `model_path` kısmına modelin dosya yolunu ekleyin. Ardından `root_dir` kısmına gömmek isteidğiniz txt dosyalarının yolunu girin.

```bash
python3 Text_embedding.py
```

***

## **Kaynaklar**

- [Char-CNN-RNN for PyTorch GitHub](https://github.com/martinduartemore/char_cnn_rnn_pytorch/tree/master)
- [char-CNN-RNN GitHub](https://github.com/1o0ko/char-CNN-RNN)
- [charCnnRnn_embedding GitHub](https://github.com/ramidzamzam/charCnnRnn_embedding/tree/main)
- [cvpr2016 GitHub](https://github.com/reedscot/cvpr2016)
- [icml2016 GitHub](https://github.com/reedscot/icml2016)
- [Generative Adversarial Text to Image Synthesis Paper](https://arxiv.org/abs/1605.05396)
- [Learning Deep Representations of Fine-grained Visual Descriptions Paper](https://arxiv.org/pdf/1605.05395)
