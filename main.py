import torch
from src.NeuralNetwork import PitchNeuralNetwork
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    df = pd.read_csv('data.csv')
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['pitch_type'])
    df = df.drop('pitch_type')

    model = PitchNeuralNetwork(len(df.iloc[0]), len(encoder.classes_)).to(device)
    print(model)


if __name__ == "__main__":
    main()
