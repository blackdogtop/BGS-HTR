import torch
from dataloader import validation_loader, FilePaths

"""validate"""
model = torch.load(FilePaths.savedModel, map_location=torch.device('cpu'))
model.eval()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

correct = 0
total = 0

with torch.no_grad():
    # Iterate over the validation set
    for data in validation_loader:
        images, labels = data

        if torch.cuda.is_available():
            images = images.to(device)
            labels = labels.to(device)

        outputs = model(images)

        # torch.max is an argmax operation
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

if __name__ == '__main__':
    print('Accuracy of the network on the validation images: %d %%' % (100 * correct / total))