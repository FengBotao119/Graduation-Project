import torch
from torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms

def inference(model,pic):
    model.eval()
    transform1 = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    pic= transform1(pic).unsqueeze(0)
    result=model(Variable(pic))
    result_npy=result.data.numpy()
    return result_npy

model = make_model()