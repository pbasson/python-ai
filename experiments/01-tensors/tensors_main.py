import torch


def addition(x, y):
    return x + y

def multiplication(x, y):
    return x * y

def print_export(x: torch.tensor ):
    print("Shape:", x.shape)
    print("Dimensions:", x.ndim)
    print("Data type:", x.dtype)

def main() -> None:
    # Create two tensors
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    # Perform calculations
    addition_result = addition(a, b)
    multiplication_result = multiplication(a, b)

    print("Addition:", addition_result)
    print("Multiplication:", multiplication_result)

    # Examine tensor properties
    print_export(a)


if __name__ == "__main__":
    main()