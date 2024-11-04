
# **Char-CNN-RNN**

This project aims to develop the Char-CNN-RNN model using our own dataset and to embed our textual data.

The model is quite successful in tasks that require detailed descriptions like text-image matching and text-to-image.

### Why Char-CNN-RNN?
The Char-CNN-RNN model is used in situations requiring high detail and accuracy for recognizing visually similar objects (e.g., different bird species). The character-level CNN and RNN structure makes the model resistant to typos by learning the character sequence in descriptions. With these features, it is successful in zero-shot learning scenarios.

### How Does It Work?
The Char-CNN-RNN model encodes text and images in a common feature space, creating a strong match between the two modalities. This allows the model to make correct predictions based on text descriptions, even for images it has never seen before.

### Advantages Over Traditional Embedding Models
Unlike word-based embedding models like Word2Vec or GloVe, Char-CNN-RNN:
- Processes characters directly, making it resilient to variations in word forms.
- Learns context dynamically, enhancing its ability to distinguish similar classes.
- Is resistant to typos and works flexibly without relying on a fixed vocabulary.

---

## **Requirements**

```bash
pip3 install pytorch==2.4.0 torchvision==0.19.0 pillow==10.4.0 tqdm==4.66.5 
```

Dataset used in the project:

[TXT files](https://drive.google.com/file/d/0B0ywwgffWnLLZW9uVHNjb2JmNlE/view?resourcekey=0-8y2UVmBHAlG26HafWYNoFQ)

[IMG files](https://data.caltech.edu/records/65de6-vp158)

---

## **Usage**

First, you need to create a dataset with `txt` and `img` files. The dataset structure should be in the following format. 

> [!NOTE]
> The dataset can be single-class or multi-class, and images can be `.jpg`, `.png`, or `.jpeg`.
> 

---

### **Dataset Folder Structure**

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

### **Preparing Image Data**

Image data was prepared as stated in Section 5 of the **[Learning Deep Representations of Fine-grained Visual Descriptions Paper](https://arxiv.org/pdf/1605.05395)**.

Each image will be divided into 10 parts by cropping the top left, bottom left, top right, bottom right, and center parts. This process is repeated with a horizontal flip, resulting in 10 images.

These images are then converted to feature vectors of 1024 dimensions using GoogleNet. Each cropped part of the image will be processed, resulting in a `.t7` file of size 60 (number of images) x 1024 (feature vector) x 10 (number of parts) for each class.

To preprocess images, enter the path of your `image` folder on line 72 in the `img2t7.py` file, and then run it.

```bash
python3 img2t7.py
```

### **Preparing Text Data**

Text data was prepared as specified in Section 5 of the **[Learning Deep Representations of Fine-grained Visual Descriptions Paper](https://arxiv.org/pdf/1605.05395)**.

Each `.txt` file will be read line by line, with each containing 10 lines. These lines are processed to contain 201 characters each. Longer lines are truncated, and shorter lines are zero-padded.

Each character is assigned a numerical value, converting character data into numerical form. All `.txt` files are combined into a single `.t7` file of size 60 (number of txt files) x 201 (character count) x 10 (line count) for each class.

To preprocess text files, enter the path of your `text` folder on line 49 in the `txt2t7.py` file, and then run it.

```bash
python3 txt2t7.py
```

### **Training the Model**

Once image and text files are prepared, each class should have a `.t7` file.

The folder structure should be as follows:

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

To train the model, run the code in the project folder. Enter the path of your dataset in `data_dir`.

For a multi-class model, line 41 in the `sje_train.py` file should read `MultimodalDataset`. For a single class, change line 41 to `SinglemodalDataset`.

```bash
python3 sje_train.py --seed 123 --use_gpu True --dataset birds --model_type cvpr --data_dir "file path" --train_split trainval --learning_rate 0.0007 --symmetric True --epochs 200 --checkpoint_dir ckpt --save_file sje_cub_c10_hybrid
```

After training, your model will be in the `ckpt` folder. To test it, run the code below, adding the dataset path to `data_dir` and the trained model path to `model_path`.

```bash
python3 sje_eval.py --seed 123 --use_gpu True --dataset birds --model_type cvpr --data_dir "file path" --eval_split test --num_txts_eval 0 --print_class_stats True --batch_size 40 --model_path "file path"
```

---

## **Creating Text Embeddings**

> [!To download pre-trained models:]
> 
> [cvpr](https://github.com/reedscot/cvpr2016)
> 
> [icml](https://github.com/reedscot/icml2016)

To embed text with the trained model, enter the model path in `model_path` on line 42 of the `Text_embedding.py` file. Then, enter the path of the txt files you want to embed in `root_dir`.

```bash
python3 Text_embedding.py
```

***

## **References**

- [Char-CNN-RNN for PyTorch GitHub](https://github.com/martinduartemore/char_cnn_rnn_pytorch/tree/master)
- [char-CNN-RNN GitHub](https://github.com/1o0ko/char-CNN-RNN)
- [charCnnRnn_embedding GitHub](https://github.com/ramidzamzam/charCnnRnn_embedding/tree/main)
- [cvpr2016 GitHub](https://github.com/reedscot/cvpr2016)
- [icml2016 GitHub](https://github.com/reedscot/icml2016)
- [Generative Adversarial Text to Image Synthesis Paper](https://arxiv.org/abs/1605.05396)
- [Learning Deep Representations of Fine-grained Visual Descriptions Paper](https://arxiv.org/pdf/1605.05395)
