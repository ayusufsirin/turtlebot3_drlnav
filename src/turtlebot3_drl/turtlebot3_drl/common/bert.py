# pip install transformers
# import os
# os.environ['REQUESTS_CA_BUNDLE'] = ''

from transformers import AutoTokenizer, AutoModel
from torch import nn

# Load a pre-trained language model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def encode_text(text_prompt):
    inputs = tokenizer(text_prompt, return_tensors="pt")
    outputs = model(**inputs)
    # Use mean pooling to get a single vector
    embeddings = outputs.last_hidden_state.mean(dim=1)
    # Convert to a Python list of floats
    mp = nn.MaxPool1d(24)
    embeddings_red = mp(embeddings)
    return embeddings_red.squeeze().tolist()

if __name__ == '__main__':
    text_prompt = "Go to Wall1"
    embedding = encode_text(text_prompt)
    print(embedding)
    # print(dir(embedding.shape))
    # print(embedding.shape)
    # print(embedding.flatten())
    # print(list(embedding.flatten()))
    # print(embedding.flatten().shape)
    #
    # a = []
    # a += list(embedding.flatten())
    #
    # print(type(a[0]))

    # print(a)