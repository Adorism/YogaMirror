import torch
import torch.onnx
import torchvision


def main():
    dummy_input = torch.randn(1, 3, 224, 224)
    model = torch.load('pytorch_model.pt')
    torch.onnx.export(model, dummy_input, "YogaPose.onnx")


if __name__ == '__main__':
    main()
