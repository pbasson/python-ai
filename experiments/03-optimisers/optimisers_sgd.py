import torch

EPOCHS, TRAINABLE, LEARNING = 20, 2.0, 0.01
START, TARGET = 3.0, 12.0

def training( x: torch.Tensor, y: torch.Tensor, w: torch.nn.Parameter):
    optimizer = torch.optim.SGD([w], lr=LEARNING)

    for epoch in range(EPOCHS):
        # 1. Reset previous gradients
        optimizer.zero_grad()

        # 2. Forward pass
        prediction = w * x

        # 3. Calculate loss
        loss = (prediction - y) ** 2

        # 4. Calculate gradients
        loss.backward()

        # 5. Update weight
        optimizer.step()

        print(f"Epoch {epoch + 1:02d} | "
            f"Weight: {w.item():.4f} | "
            f"Loss: {loss.item():.4f}"
        )

def print_export(x: torch.Tensor, w: torch.nn.Parameter):
    print(f"\nFinal weight: {w.item():.4f}")
    print(f"Final prediction: {(w * x).item():.4f}")

def main() -> None:
    x = torch.tensor(START)
    y = torch.tensor(TARGET)

    # Trainable parameter
    w = torch.nn.Parameter(torch.tensor(TRAINABLE))

    training(x, y, w)

    print_export(x, w)


if __name__ == "__main__":
    main()
