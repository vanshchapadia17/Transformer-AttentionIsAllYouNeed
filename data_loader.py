import torch

#hyper parameteres
batch_size=64 #sequences processed in parallel
block_size=256 #maximum context length
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#loadand tokenize data

with open ('C:\\Users\\DELL\\Desktop\\Transformer\\data\\train.csv','r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text))) #this gives us that what are the total unique charaters in our data
vocab_size = len(chars) #total unique characters in our data = 66

string_to_integer = {ch:i for i,ch in enumerate(chars)} #mapping from character to integer
integer_to_string = {i:ch for i,ch in enumerate(chars)} #mapping from integer to character

encode = lambda s: [string_to_integer[c] for c in s] #encoder: take a string, output a list of integers
decode = lambda l: ''.join([integer_to_string[i] for i in l]) #decoder: take a list of integers, output a string



#train and validate data split
with open('C:\\Users\\DELL\\Desktop\\Transformer\\data\\validation.csv','r', encoding='utf-8') as f:
    val_text = f.read()

#this converts data into pytorch tensors
train_data = torch.tensor(encode(text), dtype=torch.long) 
val_data = torch.tensor(encode(val_text), dtype=torch.long)


##function for preparing tarining data
def get_batch(split):

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y 

"""
xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)


for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f'when input is {context.tolist()} the target: {target.item()}')
"""