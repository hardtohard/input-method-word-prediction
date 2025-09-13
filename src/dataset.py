from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import config
#Dataset
class InputMethodDataset(Dataset):
    def __init__(self,data_path):
        self.data=pd.read_json(data_path,lines=True,orient='records').to_dict('records')
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        input_tensor=torch.tensor(self.data[index]['input'],dtype=torch.long)
        target_tensor=torch.tensor(self.data[index]['target'],dtype=torch.long)
        return input_tensor,target_tensor
#2.获取Dataloader方法
#之所以这样写是为了不仅获取训练集还可以获取测试集
def get_dataloader(train=True):
    data_path=config.PROCESSED_DIR/('index_train.jsonl' if train else 'index_test.jsonl')
    dataset=InputMethodDataset(data_path)
    return DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True)
if __name__ == '__main__':
    train_dataloader=get_dataloader()
    print(f'train batch个数:{len(train_dataloader)}')
    test_dataloader=get_dataloader(train=False)
    print(f'test batch个数:{len(test_dataloader)}')
    for inputs,targets in train_dataloader:
        print(inputs.shape)  #[batch_size,seq_len]
        print(targets.shape) #[batch_size]
        #print(inputs)
        break