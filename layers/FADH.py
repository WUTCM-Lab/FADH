import mindspore

from .FADH_Modules import *
from .crossformer2 import *
from .resnet import resnet18
import copy


    
class BaseModel(nn.Cell):
    def __init__(self, hash_bit =64,label=13,opt={}):
        super(BaseModel, self).__init__()

        
        
        self.model=CrossFormer(hash_bit,label)
       
        self.aud_feature =resnet18(num_classes=64)
        self.cross_attention_s = CrossAttention(256)

        self.il=nn.Dense(256,label)
        self.ih=nn.Dense(256,hash_bit)
        self.al=nn.Dense(256,label)
        self.ah=nn.Dense(256,hash_bit)

    def construct(self, img, aud,flag=False):
       
        mvsa_feature = self.model(img)
        
        aud_feature= self.aud_feature(aud)[0]
        
        aud_feature = self.cross_attention_s(mvsa_feature, aud_feature)
        il=self.il(mvsa_feature)
        mvsa_feature = self.ih(mvsa_feature)
        al = self.al(aud_feature)
        aud_feature = self.ah(aud_feature)

        if flag==False:
           return mvsa_feature,aud_feature#self.cross_attention_s(mvsa_feature, aud_feature)
        
        return aud_feature,mvsa_feature,il,al



def factory(opt):
    model = BaseModel(opt['bit'], opt['labels'])
    return model

class TotalLoss(nn.Cell):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.alpha = 1
        self.beta = 0.1

    def construct(self,view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2):
        term1 = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean() + (
                (view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()

        view1_feature = ops.tanh(view1_feature)
        view2_feature = ops.tanh(view2_feature)

        cos = lambda x, y: x.mm(y.t()) / (
            (x ** 2).sum(1, keepdims=True).sqrt().mm((y ** 2).sum(1, keepdims=True).sqrt().t())).clamp(min=1e-6) / 2.
        theta11 = cos(view1_feature, view1_feature)
        theta12 = cos(view1_feature, view2_feature)
        theta22 = cos(view2_feature, view2_feature)
        Sim11 = calc_label_sim(labels_1, labels_1).float()
        Sim12 = calc_label_sim(labels_1, labels_2).float()
        Sim22 = calc_label_sim(labels_2, labels_2).float()
        term21 = ((1 + ops.exp(theta11)).log() - Sim11 * theta11).mean()
        term22 = ((1 + ops.exp(theta12)).log() - Sim12 * theta12).mean()
        term23 = ((1 + ops.exp(theta22)).log() - Sim22 * theta22).mean()
        term2 = term21 + term22 + term23

        term3 = ((view1_feature - view2_feature) ** 2).sum(1).sqrt().mean()

        im_loss = term1 + self.alpha * term2 + self.beta * term3
        return im_loss


def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha, beta):
    term1 = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean() + (
                (view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()

    view1_feature = ops.tanh(view1_feature)
    view2_feature = ops.tanh(view2_feature)

    cos = lambda x, y: x.mm(y.t()) / (
        (x ** 2).sum(1, keepdims=True).sqrt().mm((y ** 2).sum(1, keepdims=True).sqrt().t())).clamp(min=1e-6) / 2.
    theta11 = cos(view1_feature, view1_feature)
    theta12 = cos(view1_feature, view2_feature)
    theta22 = cos(view2_feature, view2_feature)
    Sim11 = calc_label_sim(labels_1, labels_1).float()
    Sim12 = calc_label_sim(labels_1, labels_2).float()
    Sim22 = calc_label_sim(labels_2, labels_2).float()
    term21 = ((1 + ops.exp(theta11)).log() - Sim11 * theta11).mean()
    term22 = ((1 + ops.exp(theta12)).log() - Sim12 * theta12).mean()
    term23 = ((1 + ops.exp(theta22)).log() - Sim22 * theta22).mean()
    term2 = term21 + term22 + term23

    term3 = ((view1_feature - view2_feature) ** 2).sum(1).sqrt().mean()

    im_loss = term1 + alpha * term2 + beta * term3
    return im_loss

def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim

class MyWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, img, aud, label):
        tf, imf, il, al = self._backbone(img, aud, True)
        return self._loss_fn(tf, imf, il, al, label, label)


if __name__ == '__main__':
    img = mindspore.ops.Ones()((32, 3, 224, 224), mindspore.float32)
    aud = mindspore.ops.Ones()((32, 1, 450, 64), mindspore.float32)
    label = mindspore.ops.Ones()((32, 13), mindspore.float32)
    model = BaseModel()

    tf, imf, il, al = model(img, aud, True)
    # loss=calcul_loss(score,image.shape[0],label)
    loss = calc_loss(imf, tf, il, al, label, label, 1, 0.1)
    print(loss)


