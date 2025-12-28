import requests
import re
import random
import time
from collections import Counter

urls = [
    "https://ka.wikipedia.org/wiki/საქართველო",
    "https://ka.wikipedia.org/wiki/თბილისი",
    "https://ka.wikipedia.org/wiki/ვეფხისტყაოსანი",
    "https://ka.wikipedia.org/wiki/ილია_ჭავჭავაძე",
    "https://ka.wikipedia.org/wiki/ქართული_ენა",
]

def fetch_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            return r.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return ""

raw_text = ""
for url in urls:
    raw_text += fetch_text(url) + " "
    time.sleep(1)

def clean_and_tokenize(html_text):
    text = re.sub(r'<[^>]+>', ' ', html_text)
    words = re.findall(r'[ა-ჰ]+', text)
    return words

all_words = clean_and_tokenize(raw_text)
unique_words = list(set(word for word in all_words if len(word) > 2))

# Fallback if scraping fails
if len(unique_words) < 10:
    print("Scraping failed or returned few words. Using fallback list.")
    unique_words = [
        "გამარჯობა", "საქართველო", "თბილისი", "ქუთაისი", "ბათუმი", 
        "რუსთავი", "გორი", "ზუგდიდი", "ფოთი", "ხაშური", 
        "სამტრედია", "სენაკი", "თელავი", "ახალციხე", "ქობულეთი",
        "ოზურგეთი", "კასპი", "ჭიათურა", "წყალტუბო", "საგარეჯო",
        "გარდაბანი", "ბორჯომი", "ტყიბული", "ხონი", "ბოლნისი",
        "ახალქალაქი", "გურჯაანი", "მცხეთა", "ყვარელი", "ახმეტა",
        "კარელი", "ლანჩხუთი", "დუშეთი", "საჩხერე", "დედოფლისწყარო",
        "ლაგოდეხი", "ნინოწმინდა", "აბაშა", "წნორი", "ვალე",
        "ჯვარი", "თეთრიწყარო", "სიღნაღი", "ბაღდათი", "ვანი"
    ]

print(f"Total words found: {len(all_words)}")
print(f"Unique vocabulary size: {len(unique_words)}")
print("Examples:", unique_words[:10])

# Cell
georgian_chars = sorted(list(set(" ".join(unique_words))))

def inject_noise(word, error_rate=0.7):
    if random.random() > error_rate:
        return word
    
    chars = list(word)
    if not chars: return word
    
    op = random.choice(['insert', 'delete', 'replace', 'swap'])
    idx = random.randint(0, len(chars) - 1)
    
    if op == 'insert':
        chars.insert(idx, random.choice(georgian_chars))
    elif op == 'delete':
        if len(chars) > 1:
            chars.pop(idx)
    elif op == 'replace':
        chars[idx] = random.choice(georgian_chars)
    elif op == 'swap':
        if idx < len(chars) - 1:
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
            
    return "".join(chars)

dataset_pairs = []

for word in unique_words:
    dataset_pairs.append((word, word))
    for _ in range(3):
        corrupted = inject_noise(word, error_rate=1.0)
        dataset_pairs.append((corrupted, word))

random.shuffle(dataset_pairs)
print(f"Total training pairs: {len(dataset_pairs)}")

# Cell
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

SOS_token = 0
EOS_token = 1
PAD_token = 2

class Vocabulary:
    def __init__(self):
        self.char2index = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2}
        self.index2char = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>"}
        self.n_chars = 3

    def add_word(self, word):
        for char in word:
            if char not in self.char2index:
                self.char2index[char] = self.n_chars
                self.index2char[self.n_chars] = char
                self.n_chars += 1

vocab = Vocabulary()
for _, word in dataset_pairs:
    vocab.add_word(word)
for word, _ in dataset_pairs:
    vocab.add_word(word)

MAX_LENGTH = max(len(p[0]) for p in dataset_pairs) + 5

class SpellDataset(Dataset):
    def __init__(self, pairs, vocab):
        self.pairs = pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_word, target_word = self.pairs[idx]
        return self.tensorFromWord(input_word), self.tensorFromWord(target_word)

    def tensorFromWord(self, word):
        indexes = [self.vocab.char2index[char] for char in word]
        indexes.append(EOS_token)
        if len(indexes) < MAX_LENGTH:
            indexes += [PAD_token] * (MAX_LENGTH - len(indexes))
        else:
            indexes = indexes[:MAX_LENGTH-1] + [EOS_token]
        return torch.tensor(indexes, dtype=torch.long)

split_idx = int(0.9 * len(dataset_pairs))
train_pairs = dataset_pairs[:split_idx]
val_pairs = dataset_pairs[split_idx:]

train_dataset = SpellDataset(train_pairs, vocab)
val_dataset = SpellDataset(val_pairs, vocab)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cell
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        output = self.dropout(self.embedding(input))
        output = torch.nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.out.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        _, hidden = self.encoder(src)
        input = trg[:, 0].unsqueeze(1)
        for t in range(1, trg_len):
            output, hidden = self.decoder.gru(self.decoder.dropout(self.decoder.embedding(input)), hidden)
            prediction = self.decoder.out(output)

            outputs[:, t, :] = prediction.squeeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(2)
            input = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

# Cell
HIDDEN_SIZE = 256
enc = EncoderRNN(vocab.n_chars, HIDDEN_SIZE).to(device)
dec = DecoderRNN(HIDDEN_SIZE, vocab.n_chars).to(device)
model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

N_EPOCHS = 15
CLIP = 1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_loader, criterion)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'models/best_model.pt')
    
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}')

# Cell
import pickle

save_data = {
    'char2index': vocab.char2index,
    'index2char': vocab.index2char,
    'hidden_size': HIDDEN_SIZE,
    'n_chars': vocab.n_chars,
    'max_length': MAX_LENGTH
}

with open('models/model_config.pkl', 'wb') as f:
    pickle.dump(save_data, f)