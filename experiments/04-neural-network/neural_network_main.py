
import torch

EPOCHS, LEARNING, SEED = 500, 0.01, 42
START, TARGET = 50.0, 1250.0


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def training(x: torch.Tensor, y: torch.Tensor, model: NeuralNetwork) -> None:                       
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        # Forward pass
        prediction = model(x)

        # Calculate loss
        loss = torch.nn.functional.mse_loss(prediction, y)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:03d} | Loss: {loss.item():.6f}")


def print_export(x: torch.Tensor, model: NeuralNetwork) -> None:
    model.eval()

    with torch.no_grad():
        prediction = model(x)

    print(f"\nFinal prediction: {prediction.item():.4f}")


def main() -> None:
    torch.manual_seed(SEED)

    x = torch.tensor([[START]])
    y = torch.tensor([[TARGET]])

    model = NeuralNetwork()

    training(x, y, model)
    print_export(x, model)


if __name__ == "__main__":
    main()
