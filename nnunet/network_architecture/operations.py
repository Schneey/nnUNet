import torch
import torch.nn as nn

OPS={
    'none' : lambda C_in,stride, affine: Zero(stride),
    'skip_connect' : lambda C, stride, affine: Identity() if stride == (1,1,1) else FactorizedReduce(C, C,(1,1,1),(2,2,2),(0,0,0), affine=affine),
    'sep_conv_3x3x3': lambda C, stride, affine: ConvBNReLU(C, C, (3,3,3), stride, (1,1,1), affine=affine),
    'dil_conv_3x3x3_2': lambda C, stride, affine: DilConv(C, C, (3,3,3), stride, (2,2,2), 2, affine=affine),
    'dil_conv_3x3x3_3': lambda C, stride, affine: DilConv(C, C, (3,3,3), stride, (3,3,3), 3, affine=affine),
    'dil_conv_3x3x3_4': lambda C, stride, affine: DilConv(C, C, (3,3,3), stride, (4,4,4), 4, affine=affine),
    'avg_pool_3x3x3' : lambda C, stride, affine: nn.AvgPool3d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3x3' : lambda C, stride, affine: nn.MaxPool3d(3, stride=stride, padding=1),


}
class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == (1,1,1):
      return x.mul(0.)
    #print(x.size())
    else:
      self.stride=2
      y=x[:,:,::self.stride,::self.stride,::self.stride]
      return y.mul(0.)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv3d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv3d(C_in, C_out, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False),
      nn.InstanceNorm3d(C_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
      nn.LeakyReLU(negative_slope=0.01, inplace=False)
      )

  def forward(self, x):
    return self.op(x)




class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out,kernel_size, stride,padding,affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    
    self.conv_1 = nn.Conv3d(C_in, C_out // 2, kernel_size, stride=stride, padding=padding, bias=False)
    self.conv_2 = nn.Conv3d(C_in, C_out // 2, kernel_size, stride=stride, padding=padding, bias=False) 
    self.InstanceNorm=nn.InstanceNorm3d(C_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
    self.relu=nn.LeakyReLU(negative_slope=0.01, inplace=False)

  def forward(self, x):
    x = self.relu(x)
    # print(self.conv_1(x).size())
    # print(self.conv_2(x[:,:,:,1:,:]).size())
    y=x[:,:,:,1:,1:]
    out = torch.cat([self.conv_1(x), self.conv_1(y)], dim=1)
    out = self.InstanceNorm(out)
    out=self.relu(out)
    return out

class ConvBNReLU(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ConvBNReLU, self).__init__()
    self.op = nn.Sequential(
      nn.Conv3d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.InstanceNorm3d(C_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
      nn.LeakyReLU(negative_slope=0.01, inplace=False)
    )

  def forward(self, x):
    return self.op(x)


class FactorizedIncrease(nn.Module):

  def __init__(self, C_in, C_out,kernel_size, stride,padding,affine=True):
    super(FactorizedIncrease, self).__init__()
    assert C_out % 2 == 0
    
    self.conv_1 = nn.ConvTranspose3d(C_in, C_out // 2, kernel_size, stride=stride, padding=padding, bias=False)
    self.conv_2 = nn.ConvTranspose3d(C_in, C_out // 2, kernel_size, stride=stride, padding=padding, bias=False) 
    self.InstanceNorm=nn.InstanceNorm3d(C_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
    self.relu=nn.LeakyReLU(negative_slope=0.01, inplace=False)

  def forward(self, x):
    x = self.relu(x)
    # print(self.conv_1(x).size())
    # print(self.conv_2(x[:,:,:,:,:]).size())
    #cov2=self.conv_2(x[:,:,:,:,:])
    out = torch.cat([self.conv_1(x), self.conv_2(x)], dim=1)
    out = self.InstanceNorm(out)
    out=self.relu(out)
    return out