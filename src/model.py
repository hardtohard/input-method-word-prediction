from torch import nn
import config

class InputMethodModel(nn.Module):
    #因为引用到了nn.module继承父类的一些属性所以初始化要引用父类初始化
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=config.EMBEDDING_DIM)
        self.rnn=nn.RNN(input_size=config.EMBEDDING_DIM,hidden_size=config.HIDDEN_SIZE,batch_first=True)
        self.linear=nn.Linear(in_features=config.HIDDEN_SIZE,out_features=vocab_size)
    def forward(self,x):
        #x.shape:[batch_size,seq_len]
        embed = self.embedding(x)
        #embed.shape:[batch_size,seq_len,embedding_dim]
        #hidden可以用_占位
        output,hidden=self.rnn(embed)
        #output.shape: [batch_size,seq_len,hidden_dim]
        last_hidden=output[:,-1,:]
        #这里有点难last_hidden.shape[batch_size,hidden_dim]

        #此处可以备选方案：利用hidden
        #last_hidden=hidden.squeeze(0)
        #具体参数看官方api调用：hidden.shape:[1,batch_size,hidden_dim]
        #复用output这个变量
        output=self.linear(last_hidden)
        #output.shape:[batch_size,vocab_size]
        return output
