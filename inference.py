import torch
import torch.nn as nn
import pickle

with open('models/model_config.pkl', 'rb') as f:
    config = pickle.load(f)

char2index = config['char2index']
index2char = config['index2char']
HIDDEN_SIZE = config['hidden_size']
N_CHARS = config['n_chars']
MAX_LENGTH = config['max_length']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        pass

enc = EncoderRNN(N_CHARS, HIDDEN_SIZE).to(device)
dec = DecoderRNN(HIDDEN_SIZE, N_CHARS).to(device)
model = Seq2Seq(enc, dec, device).to(device)

model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
model.eval()

# Cell
def correct_word(word: str, model_path: str = None) -> str:
    indexes = []
    for char in word:
        if char in char2index:
            indexes.append(char2index[char])
            
    indexes.append(1)
    tensor_in = torch.tensor(indexes, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, hidden = model.encoder(tensor_in)
        decoder_input = torch.tensor([[0]], device=device)
        decoded_chars = []
        
        for _ in range(MAX_LENGTH):
            output, hidden = model.decoder(decoder_input, hidden)
            top1 = output.argmax(2)
            token_idx = top1.item()
            
            if token_idx == 1:
                break
            if token_idx == 0:
                continue
            if token_idx == 2:
                break
                
            decoded_chars.append(index2char[token_idx])
            decoder_input = top1
            
    return "".join(decoded_chars)

test_words = [
    "გამარჰონა",
    "თბილისი",
    "პროგამა",
    "საქარველო",
    "უნივერსიტეტი"
]

for tw in test_words:
    print(f"Input: {tw:15} -> Output: {correct_word(tw)}")

# Cell
demo_set = [
    ("შკოლა", "სკოლა"),
    ("მასწავლბელი", "მასწავლებელი"),
    ("ისტრია", "ისტორია"),
    ("გეოგრფია", "გეოგრაფია"),
    ("ლიტერატრა", "ლიტერატურა"),
    ("ქალაქ", "ქალაქი"),
    ("მანქან", "მანქანა"),
    ("კომპიუტრი", "კომპიუტერი"),
    ("ტელეფონი", "ტელეფონი"),
    ("ინტერნტი", "ინტერნეტი"),
    ("წიგნი", "წიგნი"),
    ("რვეული", "რვეული"),
    ("კალამი", "კალამი"),
    ("დაფა", "დაფა"),
    ("მაგია", "მაგიდა"),
    ("სკამ", "სკამი"),
    ("ფანჯრა", "ფანჯარა"),
    ("კარი", "კარი"),
    ("იატკი", "იატაკი"),
    ("ჭერი", "ჭერი")
]

for inp, target in demo_set:
    pred = correct_word(inp)
    status = "✅" if pred == target else f"❌ (Expected {target})"
    print(f"In: {inp:15} | Out: {pred:15} | {status}")