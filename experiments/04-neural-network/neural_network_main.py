
import torch

EPOCHS, LEARNING, SEED = 500, 0.05, 42

START = [[1.0], [2.0], [3.0], [4.0], [5.0]]
END = [[4.0], [8.0], [12.0], [16.0], [20.0]]

#TEST_INPUT = [10.0]
TEST_INPUT = [[6.0], [7.0], [10.0], [100.0], [1250.0]]

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
        predictions = model(x)

    print("\nModel Evaluation")
    print(f"{'Input':>10} {'Expected':>12} {'Predicted':>12} {'Error':>12}")

    for input_value, prediction in zip(x, predictions):
        actual = input_value.item()
        expected = actual * 4
        predicted = prediction.item()
        error = abs(expected - predicted)

        print(
            f"{actual:10.2f} "
            f"{expected:12.2f} "
            f"{predicted:12.4f} "
            f"{error:12.4f}"
        )


def main() -> None:
    torch.manual_seed(SEED)

    x = torch.tensor(START)
    y = torch.tensor(END)

    model = NeuralNetwork()

    training(x, y, model)
    test_x = torch.tensor(TEST_INPUT)
    print_export(test_x, model)


if __name__ == "__main__":
    main()
