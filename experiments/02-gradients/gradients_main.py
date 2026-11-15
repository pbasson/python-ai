
import torch

START, END, TARGET = 3.0, 2.0, 12.0

def multiplication(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b

def calculate_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target) ** 2

def print_export(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.tensor:
    print("Prediction:", x.item())
    print("Loss:", y.item())
    print("Weight gradient:", z.grad.item())

def main() -> None:
    x = torch.tensor(START)
    w = torch.tensor(END, requires_grad=True)
    target = torch.tensor(TARGET)

    # Forward pass
    prediction = multiplication(w, x)

    # Calculate error
    loss = calculate_loss(prediction, target)

    # Calculate gradients of the loss
    loss.backward()

    print_export(prediction, loss, w)


if __name__ == "__main__":
    main()
