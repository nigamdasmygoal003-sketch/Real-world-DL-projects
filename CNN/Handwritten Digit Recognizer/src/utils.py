import torch

def predict(model,image):
    model.eval()
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output,dim=1)
        pred = torch.argmax(probs,dim=1).item()
        confidence = probs[0][pred].item()
        
    return pred,confidence
    