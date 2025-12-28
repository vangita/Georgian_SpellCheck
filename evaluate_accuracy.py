import torch
import torch.nn as nn
import pickle
import random
import time

# --- 1. CONFIG & MODEL DEFINITION (Must match training) ---
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

    # Matching training logic which bypassed standard forward in the loop
    # Training loop: output, hidden = self.decoder.gru(self.decoder.dropout(self.decoder.embedding(input)), hidden)
    #                prediction = self.decoder.out(output)
    def forward(self, input, hidden):
        output = self.dropout(self.embedding(input))
        # REMOVED RELU TO MATCH TRAINING MISTAKE
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

# Load Model
enc = EncoderRNN(N_CHARS, HIDDEN_SIZE).to(device)
dec = DecoderRNN(HIDDEN_SIZE, N_CHARS).to(device)
model = Seq2Seq(enc, dec, device).to(device)
model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
model.eval()

# --- 2. DATA GENERATION UTILS ---
# Using the same fallback list to ensure we test on valid vocabulary
# In a real scenario, correct_words should come from a held-out test set
correct_words = [
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
georgian_chars = sorted(list(set("".join(correct_words))))

def inject_noise(word, error_rate=0.7):
    # Always inject error for test purposes if error_rate=1.0
    if random.random() > error_rate:
        return word
    chars = list(word)
    if not chars: return word
    op = random.choice(['insert', 'delete', 'replace', 'swap'])
    idx = random.randint(0, len(chars) - 1)
    if op == 'insert': chars.insert(idx, random.choice(georgian_chars))
    elif op == 'delete': 
        if len(chars) > 1: chars.pop(idx)
    elif op == 'replace': chars[idx] = random.choice(georgian_chars)
    elif op == 'swap':
        if idx < len(chars) - 1: chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    return "".join(chars)

def predict(word):
    indexes = [char2index[char] for char in word if char in char2index]
    indexes.append(1) # EOS
    
    # PAD INDICES TO MATCH TRAINING (Crucial because we didn't use masking in training)
    if len(indexes) < MAX_LENGTH:
        indexes += [2] * (MAX_LENGTH - len(indexes)) # PAD=2
    else:
        indexes = indexes[:MAX_LENGTH]
        
    tensor_in = torch.tensor(indexes, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, hidden = model.encoder(tensor_in)
        
        # MODEL TRAINING FLAW: It was trained with trg[0] as seed, not SOS.
        # We must seed with the first char of the INPUT word.
        # If the first char is wrong (typo), the model might struggle, but this is the best we can do.
        start_token = indexes[0] # The first char of input
        decoder_input = torch.tensor([[start_token]], device=device)
        
        # The result starts with the seed
        decoded_chars = [index2char[start_token]]
        
        for _ in range(MAX_LENGTH):
            output, hidden = model.decoder(decoder_input, hidden)
            top1 = output.argmax(2)
            token_idx = top1.item()
            if token_idx == 1: break # EOS
            if token_idx == 2: break # PAD
            # 0 is strictly SOS/random here, shouldn't be predicted really
            decoded_chars.append(index2char[token_idx])
            decoder_input = top1
            
    return "".join(decoded_chars)

# --- 3. EVALUATION LOOP ---
N_TEST = 1000
correct_count = 0
print(f"Generating {N_TEST} synthetic test examples...")

test_data = []
for _ in range(N_TEST):
    # 50% chance of being already correct (identity test)
    # 50% chance of being corrupted
    target = random.choice(correct_words)
    if random.random() < 0.5:
        src = target
    else:
        src = inject_noise(target, error_rate=1.0)
    test_data.append((src, target))

print("Running evaluation...")
start_time = time.time()

for i, (src, target) in enumerate(test_data):
    prediction = predict(src)
    if prediction == target:
        correct_count += 1
    
    if i < 10: # Print first 10
        status = "✅" if prediction == target else "❌"
        print(f"Src: {src:15} | Tgt: {target:15} | Pred: {prediction:15} | {status}")

accuracy = (correct_count / N_TEST) * 100
duration = time.time() - start_time

print("-" * 30)
print(f"Total Samples: {N_TEST}")
print(f"Correct:       {correct_count}")
print(f"Accuracy:      {accuracy:.2f}%")
print(f"Time Taken:    {duration:.2f}s")
