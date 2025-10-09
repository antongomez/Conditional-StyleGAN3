import torch
import torch.nn as nn

N1 = 5    # dimension de entrada
D1 = 2          # decimacion, por defecto 2
H1 = 32      # lado patches entrada, por defecto 28 (sizex=sizey)
# dimension de salida (seleccionada), por defecto 16
N2 = 16
# lado patches salida (calculada), por defecto 28 (sizex=sizey)
H2 = int(H1/D1)

N3 = 32
D2 = 2          # decimacion, por defecto 2
H3 = int(H2/D2)  # lado patches salida

N4 = H3*H3*N3   # dimension de entrada
N5 = 512    # dimension de salida

class CNN21(nn.Module):
    def __init__(self, N1=5, N2=16, N3=32, N4=8*8*32, N5=512, D1=2, D2=2):
        super(CNN21, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(N1, N2, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(N2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=D1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(N2, N3, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(N3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=D2))
        self.fc = nn.Linear(N4, N5)

    def forward(self, img, c, update_emas=False, return_features=False, return_latents=False, **block_kwargs):
        out = self.layer1(img)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out, None
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier = CNN21().to(device)

    image_size = 32
    num_channels = 5
    batch_size = 32

    img = torch.randn(batch_size, num_channels, image_size, image_size).to(device)
    print(f"Input image size: {img.size()}")  # torch.Size([32, 5, 32, 32])
    output = classifier(img)
    print(f"Output size: {output.size()}")  # torch.Size([32, 10])

