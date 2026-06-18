import torch
class Evaluator:
    def __init__(self,loader,device): self.loader=loader; self.device=device
    def evaluate(self,model):
        model.to(self.device).eval(); loss_fn=torch.nn.CrossEntropyLoss(reduction='sum'); loss=0; correct=0; n=0
        with torch.no_grad():
            for x,y in self.loader:
                x,y=x.to(self.device),y.to(self.device); logits=model(x); loss+=float(loss_fn(logits,y)); correct+=int((logits.argmax(1)==y).sum()); n+=len(x)
        return loss/max(1,n), correct/max(1,n)
