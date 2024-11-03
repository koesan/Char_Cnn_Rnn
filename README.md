# Char-Cnn-Rnn

Bu projenin amacı, kendi veri setimizi kullanarak Char-CNN-RNN modelini geliştirmek ve verilerimizi gömmektir (embedding).

## Gereksinimler



## kullanım:

Öncelikle txt ve img verileri bununduğu bir veri seti oluşturmalısınız veri setinin yapısı aşağıdaki formatta olamlıdır(Resimler .jpg .png ,jpeg olabilir).

### Veri Seti Klasör Yapısı

```
dataset/
├── text/
│   ├── dosya1.txt
│   ├── dosya2.txt
│   ├── dosya3.txt
│   └── ...
└── images/
    ├── resim1.jpg
    ├── resim2.jpg
    ├── resim3.jpg
    └── ...
```

### görsel verilerin hazırlanması

Görüntü verileri, Learning Deep Representations of Fine-grained Visual Descriptions makalesinin 5. bölümünde belirtildiği gibi hazırlandı. Öncelikle görseller toplam 10 (görselin sol üst, sol alt, sağ üst, sağ alt, orta kısmı kırpılır arıdndan görsel çevrilerek aynı şekilde 5 tane daha görsel kırpılır sonuç olarak 10 tane görsel elde edilecek) parçaya ayrılacaktır. Ardından, tüm görseller GoogleNet kullanılarak her görsel, 1024 boyutunda özellik vektörlerine dönüştürülecektir. Elde edilen çıktılar, 60 (görsel sayısı) x 1024 (özellik vektörü) x 10 (parça sayısı) boyutunda .t7 formatında kaydedilecektir. Her sınıf için birtane t7 dosyası oluşturulacak.



kaynaklar:

[Char-CNN-RNN for PyTorch GitHub](https://github.com/martinduartemore/char_cnn_rnn_pytorch/tree/master)

[char-CNN-RNN GitHub](https://github.com/1o0ko/char-CNN-RNN)

[charCnnRnn_embedding GitHub](https://github.com/ramidzamzam/charCnnRnn_embedding/tree/main)

[cvpr2016 GitHub](https://github.com/reedscot/cvpr2016)

[icml2016 GitHub](https://github.com/reedscot/icml2016)

[Generative Adversarial Text to Image Synthesis Paper](https://arxiv.org/abs/1605.05396)

[Learning Deep Representations of Fine-grained Visual Descriptions Paper](https://arxiv.org/pdf/1605.05395)

