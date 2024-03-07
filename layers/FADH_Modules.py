from mindspore import nn
from mindspore import ops
import mindspore


    
def sim(img,text):
    w12=img*text
    w1=ops.norm(img,2)
    w2=ops.norm(text,2)
    return w12/(w1*w2)

# cross attention
class CrossAttention(nn.Cell):

    def __init__(self, hash_bit):
        super(CrossAttention, self).__init__()

        self.att_type = "sim_att"
        dim = hash_bit
        if self.att_type == "soft_att":
            self.cross_attention = nn.SequentialCell(
                nn.Dense(dim, dim),
                nn.Sigmoid()
            )
        elif self.att_type == "fusion_att":
            self.cross_attention_fc1 = nn.SequentialCell(
                nn.Dense(2*dim, dim),
                nn.Sigmoid()
            )
            self.cross_attention_fc2 = nn.SequentialCell(
                nn.Dense(2*dim, dim),
            )
            self.cross_attention = lambda x:self.cross_attention_fc1(x)*self.cross_attention_fc2(x)

        elif self.att_type == "similarity_att":
            self.fc_visual = nn.SequentialCell(
                nn.Dense(dim, dim),
            )
            self.fc_text = nn.SequentialCell(
                nn.Dense(dim, dim),
            )
        elif self.att_type == "sim_att":
            self.fc_visual = nn.SequentialCell(
                nn.Dense(dim, dim),
            )
            self.fc_text = nn.SequentialCell(
                nn.Dense(dim, dim),
            )
        elif self.att_type == "ls_att":
            self.fc_visual = nn.SequentialCell(
                nn.Dense(dim, dim),
            )
            self.fc_text = nn.SequentialCell(
                nn.Dense(2*dim, dim),
            )    
        
        else:
            raise Exception

    def construct(self, visual, text):

        if self.att_type == "soft_att":
            visual_gate = self.cross_attention(visual)
            return visual_gate*text

        elif self.att_type == "fusion_att":
            fusion_vec = ops.cat([visual,text], dim=-1)

            return self.cross_attention(fusion_vec)
        elif self.att_type == "similarity_att":
            visual = self.fc_visual(visual)
            text = self.fc_text(text)
            
            sims = visual*text
            return ops.sigmoid(sims) * text
        elif self.att_type == "sim_att":
            visual = self.fc_visual(visual)
            visual=sim(visual,text)
            return ops.sigmoid(visual)*text
        elif self.att_type=="ls_att":
            visual=(visual*text)*visual
            visual=self.fc_visual(visual)
            con=ops.cat((visual,text),-1)
            con=self.fc_text(con)
            cr=ops.sigmoid(con)
            gr=ops.sigmoid(con)
            text=(1-gr)*text+gr*cr
            return visual,text

if __name__ == '__main__':
    visual = ops.Ones()((60, 256), mindspore.float32)
    text = ops.Ones()((60,256), mindspore.float32)
    print(text.shape)
    print(visual.shape)
    module = CrossAttention(256)
    output = module(visual, text)
    print(output.shape)





