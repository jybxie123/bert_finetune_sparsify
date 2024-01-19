from torch.utils.data import Dataset
import torch
    
class SelfDefDataset(Dataset):
    def __init__(self, dataset_dict):
        self.input_ids = torch.tensor(dataset_dict['input_ids'])
        self.attention_mask = torch.tensor(dataset_dict['attention_mask'])
        self.labels = torch.tensor(dataset_dict['label'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], 'labels': self.labels[idx]}
        return item