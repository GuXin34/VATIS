import torch
import torch.nn as nn
import torch.nn.functional as F

class VQVAE(nn.Module):
    def __init__(self, input_channels, num_embeddings, embedding_dim, hidden_dim):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(input_channels, hidden_dim)
        self.codebook = Codebook(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, input_channels)

    def forward(self, x):
        z = self.encoder(x)
        quantized, indices = self.codebook(z)
        x_recon = self.decoder(quantized)
        return x_recon, quantized, indices

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x


class Codebook(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Codebook, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        # Flatten the input tensor for quantization
        flattened = x.view(-1, self.embedding_dim)

        # Calculate distances between the input vectors and the embeddings
        distances = torch.norm(flattened.unsqueeze(1) - self.embedding.weight, dim=2, p=2)

        # Find the closest embeddings and their indices
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices).view(x.shape)

        return quantized, indices

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_channels):
        super(Decoder, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(embedding_dim, hidden_dim * 8, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(hidden_dim * 2, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x
