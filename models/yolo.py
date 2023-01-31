import argparse

from models.experimental import *

class Detect(nn.Module):
    def __init__(self, nc=15, anchors=()): #检测层
        super(Detect, self).__init__()
        self.stride = None # 建立模型时计算步幅
        self.nc = nc # 分类的个数
        self.no = nc + 5 # 每一个锚框输出元素的个数
        self.nl = len(anchors) # 检测层的数量
        self.na = len(anchors[0]) // 2 # 锚框的个数
        self.grid = [torch.zeros(1)] * self.nl # 初始化网格
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a) # shape(nl, na, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2)) # shape(nl, 1, na, 1, 1, 2)
        self.export = False

    def forward(self, x):
        z = [] # 推理输出
        self.training |= self.export
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape # x(bs, 255, 20, 20) to x(bs, 3, 20, 20, 85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                    y = x[i].sigmoid()
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                    z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)


    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, model_cfg='yolov5l.yaml', ch=3, nc=15):
        super(Model, self).__init__()
        if type(model_cfg) is dict:
            self.md = model_cfg # model dict
        else:
            with open(model_cfg) as f:
                self.md = yaml.load(f, Loader=yaml.FullLoader)
                