import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class NbeatsNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self,
                 device,
                 stack_types=[TREND_BLOCK, SEASONALITY_BLOCK],
                 nb_blocks_per_stack=3,
                 forecast_length=5,
                 backcast_length=10,
                 thetas_dims=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=256):
        super(NbeatsNet, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.stack_types = stack_types
        print(len(self.stack_types))
        self.stacks = []
        self.thetas_dim = thetas_dims
        self.parameters = []
        self.device = device
        print(f'| N-Beats')
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []

        for block_id in range(self.nb_blocks_per_stack):
            block_init = NbeatsNet.select_block(stack_type)
            if stack_type=='TREND_BLOCK':
                temp=0
            else:
                temp=1
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(self.hidden_layer_units, self.thetas_dim[temp],
                                   self.device, self.backcast_length, self.forecast_length)
                self.parameters.extend(block.parameters())
            print(f'     | -- {block}')
            blocks.append(block)
        return blocks

    @staticmethod
    def select_block(block_type):
        if block_type == NbeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NbeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def forward(self, backcast,real_t,imag_t):
        forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length,))  # maybe batch size here.
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast,real_t,imag_t)
                backcast = backcast.to(self.device) - b
                forecast = forecast.to(self.device) + f
        return backcast, forecast

'''
def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p < 10, 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()  # H/2-1
    s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))
'''
def seasonality_model(thetas , real, imag, t,device):

    p=thetas.size()[-1]
    p1,p2=(p//2,p//2) if p%2==0 else (p//2, p//2+1)
    s1=torch.zeros(size=(p1,len(t)))
    s2=torch.zeros(size=(p2,len(t)))
    real=real[0:p1+1]
    imag=imag[0:p2+1]
    t=torch.from_numpy(t).cuda()
    for i in range(p1):
        s1[i]=real[i]*t
    for i in range(p2):
        s2[i]=imag[i]*t
    #s1=[torch.tensor(real[s]*t) for s in range (p1)]
    #s2=[torch.tensor(imag[s]*t) for s in range (p2)]
    #s1=torch.tensor(s1)
    #print("s1:",s1)
    s=torch.cat([s1,s2])
    #print(thetas.mm(s.to(device)))
    return thetas.mm(s.to(device))

def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor([t ** i for i in range(p)]).float()
    return thetas.mm(T.to(device))


def linspace(backcast_length, forecast_length):
    lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class Block(nn.Module):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, share_thetas=False):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(backcast_length, units)
        self.conv1=nn.Conv1d(1,32,17,1,8)
        #self.fc2 = nn.Linear(units, units)
        #self.fc3 = nn.Linear(units, units)
        self.conv2=nn.Conv1d(32,64,17,1,8)
        self.conv3=nn.Conv1d(64,1,17,1,8)
        self.fc4 = nn.Linear(units, units)
        self.device = device
        self.backcast_linspace, self.forecast_linspace = linspace(backcast_length, forecast_length)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim)
            self.theta_f_fc = nn.Linear(units, thetas_dim)

    def forward(self, x,real_t,imag_t):
        x = F.relu(self.fc1(x.to(self.device)))
        x=x.view(x.shape[0],1,x.shape[1])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=x.view(x.shape[0],-1)
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5):
        super(SeasonalityBlock, self).__init__(units, thetas_dim, device, backcast_length,
                                               forecast_length, share_thetas=True)

    def forward(self, x,real_t,imag_t):
        x = super(SeasonalityBlock, self).forward(x,real_t,imag_t)
        backcast = seasonality_model(self.theta_b_fc(x),real_t,imag_t, self.backcast_linspace, self.device)
        forecast = seasonality_model(self.theta_f_fc(x),real_t,imag_t, self.forecast_linspace, self.device)
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5):
        super(TrendBlock, self).__init__(units, thetas_dim, device, backcast_length,
                                         forecast_length, share_thetas=True)

    def forward(self, x,real_t,imag_t):
        x = super(TrendBlock, self).forward(x,real_t,imag_t)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5):
        super(GenericBlock, self).__init__(units, thetas_dim, device, backcast_length, forecast_length)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x,real_t,imag_t):
        # no constraint for generic arch.
        x = super(GenericBlock, self).forward(x,real_t,imag_t)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast
