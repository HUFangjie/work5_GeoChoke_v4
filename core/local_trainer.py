import torch
class LocalTrainer:
    def __init__(self,epochs:int,lr:float,device:str): self.epochs=epochs; self.lr=lr; self.device=device
    def train(self, model, loader):
        model.to(self.device); model.train(); opt=torch.optim.SGD(model.parameters(),lr=self.lr); loss_fn=torch.nn.CrossEntropyLoss(); total=0; n=0
        for _ in range(self.epochs):
            for x,y in loader:
                x,y=x.to(self.device),y.to(self.device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step(); total+=float(loss.item())*len(x); n+=len(x)
        return total/max(1,n)
