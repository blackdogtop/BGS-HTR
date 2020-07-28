from dataloader import FilePaths, test_loader, test_data_df
import torch
import cv2

"""测试"""
model = torch.load(FilePaths.savedModel, map_location=torch.device('cpu'))
model.eval()

results = []
with torch.no_grad():
    # Iterate over the test set
    for data in test_loader:
        images, labels = data

        outputs = model(images)

        # torch.max is an argmax operation
        _, predicted = torch.max(outputs.data, 1)

        for p in predicted:
            results.append(p.item())
# print(results)

for i, path in enumerate(test_data_df['path']):
    image = cv2.imread(path)
    print(results[i])
    cv2.imshow('test',image)
    cv2.waitKey(0)