import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

# Constants
VOCAB = ['a', 'b', 'c', 'd', 'e', 'X']
VOCAB_SIZE = len(VOCAB)
MAX_SEQ_LENGTH = 10
MIN_SEQ_LENGTH = 5
PAD_IDX = VOCAB_SIZE  # Using VOCAB_SIZE as padding index
BATCH_SIZE = 32
NUM_EPOCHS = 50

# Create letter to index and index to letter mappings
letter2idx = {letter: idx for idx, letter in enumerate(VOCAB)}
letter2idx['<pad>'] = PAD_IDX
idx2letter = {idx: letter for letter, idx in letter2idx.items()}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=MAX_SEQ_LENGTH):
        super().__init__()

        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LetterSortingTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size + 1, d_model)  # +1 for padding
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model * 4,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, vocab_size + 1)  # +1 for padding

    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer(src, src_mask)
        return self.output_layer(output)


class LetterSortingDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.data = []

        for _ in range(num_samples):
            # Random sequence length between MIN_SEQ_LENGTH and MAX_SEQ_LENGTH
            seq_len = random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)

            # Generate random sequence of letters
            sequence = random.choices(VOCAB, k=seq_len)

            # Sort the sequence for target
            target = sorted(sequence)

            # Convert to indices and pad
            src_indices = [letter2idx[c] for c in sequence]
            tgt_indices = [letter2idx[c] for c in target]

            # Pad sequences to MAX_SEQ_LENGTH
            src_indices += [PAD_IDX] * (MAX_SEQ_LENGTH - len(src_indices))
            tgt_indices += [PAD_IDX] * (MAX_SEQ_LENGTH - len(tgt_indices))

            self.data.append((torch.tensor(src_indices), torch.tensor(tgt_indices)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = LetterSortingTransformer(VOCAB_SIZE).to(device)

    # Create datasets and dataloaders
    train_dataset = LetterSortingDataset(num_samples=1000)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src)

            loss = criterion(output.view(-1, VOCAB_SIZE + 1), tgt.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}')

    return model


def sort_sequence(model, sequence):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Convert input sequence to indices
    src_indices = [letter2idx[c] for c in sequence]
    src_indices += [PAD_IDX] * (MAX_SEQ_LENGTH - len(src_indices))
    src = torch.tensor(src_indices).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(src)
        predictions = output.argmax(dim=-1)

    # Convert predictions back to letters and remove padding
    result = [idx2letter[idx.item()] for idx in predictions[0]]
    result = [c for c in result if c != '<pad>']

    return result


# Example usage
if __name__ == "__main__":
    # Train the model
    model = train_model()

    # Test the model
    test_sequences = [
        ['b', 'a', 'c', 'e', 'd'],
        ['e', 'd', 'c', 'b', 'a'],
        ['a', 'b', 'c', 'd', 'e'],
        ['a', 'a', 'e', 'e', 'a', 'b'],
        ['e', 'e', 'b', 'b', 'b', 'a', 'b', 'c']
    ]

    for seq in test_sequences:
        sorted_seq = sort_sequence(model, seq)
        print(f"Input: {seq}")
        print(f"Output: {sorted_seq}\n")