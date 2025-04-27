import torch


checkpoint = torch.load('/home/myid/bs83243/mastersProject/energy_constraint_ensemble/valFile/resnet18.pth')
predictionVectors = checkpoint['predictionVectors']
predictionVectorsList.append(torch.nn.functional.softmax(predictionVectors, dim=-1).cpu()) # 
labelVectors = prediction['labelVectors']
labelVectorsList.append(labelVectors.cpu())
# tmpAccList.append(calAccuracy(predictionVectors, labelVectors)[0].cpu())

print(predictionVectors.size())

# print(checkpoint.keys())  # This will show the stored data
