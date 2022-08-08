import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import math

class CoOccurrenceLayer(nn.Module):
    def __init__(self, co_matrix_shape: tuple, spatial_shape: list, stride: int = 1) -> None:
        
        super(CoOccurrenceLayer, self).__init__()
        self.co_matrix_shape = co_matrix_shape
        self.spatial_shape = spatial_shape
        self.stride = stride
        self.co_matrix = Parameter(torch.Tensor(*self.co_matrix_shape))
        self.spatial_filter = Parameter(torch.Tensor(*self.spatial_shape))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.co_matrix, a=math.sqrt(5))
        init.kaiming_uniform_(self.spatial_filter, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        num_quantization = self.co_matrix_shape[0]
        input_idx = self._quantization_input_as_bins(input, num_quantization)
        conv_out=self._cal_co_currence_conv(input,input_idx,num_quantization)
        return conv_out

    def _cal_co_currence_conv(self,imgs:torch.Tensor,selected_idx:torch.Tensor, num_q:int):
        b,c,h,w=imgs.shape
        selected_idx=selected_idx.flatten() # b c h w
        conv_out=torch.zeros_like(imgs)
        for i in range(num_q):
            
            ith_row_M=self.co_matrix[i]
            
            M_selected=ith_row_M[selected_idx.long()]
            
            M_selected=M_selected.reshape([b,c,h,w])

            M_dot_img=M_selected * imgs # b c h w
            # to keep the same shape
            M_dot_img_padded=self.fix_padding(M_dot_img,self.spatial_shape)
            
            un_sq_spatial_filter=self.spatial_filter.unsqueeze(0).unsqueeze(0) # 1 1 *filter_shape
            
            # if self.direction=='vertical':
            #     stride=1
            # elif self.direction=='diagonal':
            #     stride=[1,1]
            
            spatialed=torch.conv2d(M_dot_img_padded,un_sq_spatial_filter)
            
            mask=(selected_idx==i).float().reshape([b,c,h,w])          

            masked = spatialed * mask
            
            conv_out+=masked

        return conv_out
            


    def fix_padding(self, input: torch.Tensor, spatial_shape: list) -> torch.Tensor:
        pad_list = []
        for fs in spatial_shape[::-1]:
            pad_total = fs - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            pad_list.extend([pad_beg, pad_end])
        input_padding = F.pad(input, pad_list)
        return input_padding
        

    @staticmethod
    def _quantization_input_as_bins(img: torch.Tensor, num_quantization: int) -> torch.Tensor:
        # in case neg value exists in img input 
        img=torch.exp(img)
        # normalize the input to 0~1
        input_norm = (img - img.min()) / img.max()
        # ----- use round method -------- #
        # May cause the problem describe in issue #2
        # input to index
        # input_idx = input_norm * (num_quantization - 1)
        # # floor to int
        # input_idx = torch.round(input_idx).int()
        # ------ use floor method v2 -------- #
        eps = torch.FloatTensor([1e-5])
        input_idx = input_norm * num_quantization
        if input_idx.is_cuda:
            eps = eps.cuda()
        input_idx = torch.abs(input_idx - eps)
        input_idx = torch.floor(input_idx)
        return input_idx

class CofNet(nn.Module):
    def __init__(self,co_matrix_shape,spatial_filter_shape,img_channel,batch_size,linear_out_dim,h,w):
        super().__init__()
        self.batch_size=batch_size
        self.img_channel=img_channel
        self.h=h
        self.w=w
        self.cof=CoOccurrenceLayer(co_matrix_shape,spatial_filter_shape)
        self.linear=nn.Linear(img_channel*h*w,linear_out_dim)
    def forward(self,x):
        x=self.cof(x)
        x=torch.relu(x)
        x=x.flatten(1,-1)
        # x=x.reshape(self.batch_size,-1)
        # x=x.view(self.batch_size,-1)
        x=self.linear(x)
        return x

if __name__=='__main__':
    img_channel=1
    batch_size=128
    h,w=40,40
    # generate gray imgs as network input
    imgs=torch.rand(batch_size,img_channel,h,w)# b c h w
    co_shape=(20,20)
    spatial_shape=(10,10)
    linear_out=4 # could be any dim.
    net=CofNet(co_matrix_shape=co_shape,spatial_filter_shape=spatial_shape,\
                img_channel=img_channel,batch_size=batch_size,linear_out_dim=linear_out,h=h,w=w)
    out=net(imgs)
