import os
import torch
from torch.utils.data import Dataset
from char_cnn_rnn.char_cnn_rnn import labelvec_to_onehot

class SinglemodalDataset(Dataset):

    def __init__(self, data_dir, cls):
        super().__init__()
        self.cls = cls
        
        # Load images and texts for the specific class
        path_imgs = os.path.join(data_dir, "image/001.Black_footed_Albatross" + '.t7')
        path_txts = os.path.join(data_dir, "text/001.Black_footed_Albatross" + '.t7')
        
        self.cls_imgs = torch.Tensor(torch.load(path_imgs))
        self.cls_txts = torch.LongTensor(torch.load(path_txts))

        self.num_instances = self.cls_imgs.size(0)

    def __len__(self):
        return self.num_instances

    def __getitem__(self, index):
        # Get a specific instance from the class
        id_txt = torch.randint(self.cls_txts.size(2), (1,))
        id_view = torch.randint(self.cls_imgs.size(2), (1,))

        img = self.cls_imgs[index, :, id_view].squeeze()
        txt = self.cls_txts[index, :, id_txt].squeeze()
        txt = labelvec_to_onehot(txt)

        return {'img': img, 'txt': txt}
