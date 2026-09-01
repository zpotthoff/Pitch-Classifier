import torch
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd

from src.NeuralNetwork import PitchNeuralNetwork
from src.PitchDataset import PitchDataset

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    train_df = pd.read_csv('data/training.csv')
    validation_df = pd.read_csv('data/validation.csv')
    test_df = pd.read_csv('data/testing.csv')

    training_dataset = PitchDataset(train_df.drop(columns='pitch_type'), train_df['pitch_type'])
    validation_dataset = PitchDataset(validation_df.drop(columns='pitch_type'), validation_df['pitch_type'])
    testing_dataset = PitchDataset(test_df.drop(columns='pitch_type'), test_df['pitch_type'])

    training_loader = DataLoader(
        training_dataset,
        batch_size=256,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=256,
        shuffle=False
    )

    # minus 1 so we don't count pitch_type
    # eventually get this stuff not hardcoded
    model = PitchNeuralNetwork(num_inputs=train_df.shape[1] - 1, num_pitches=19).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.001
    )

    for epoch in range(20):
        model.train()

        for X_batch, y_batch in training_loader:
            optimizer.zero_grad()
            logits = model(X_batch.float().to(device))
            loss = criterion(logits, y_batch.to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in validation_loader:
                logits = model(X_batch.float().to(device))
                predictions = logits.argmax(dim=1)
                correct += (predictions == y_batch.to(device)).sum().item()
                total += y_batch.size(0)

        print(f"Epoch {epoch} --- Loss: {loss.item():.4f} --- Accuracy: {correct/total:.4f}")


if __name__ == "__main__":
    main()
