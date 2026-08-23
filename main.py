import torch
from src.NeuralNetwork import NeuralNetwork

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)


if __name__ == "__main__":
    main()
