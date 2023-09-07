import torch
from functools import partial



# random tensors to test with
def gen_x(B, C, H, W):
    return torch.randn(B, C, H, W)


def gen_y(B, C, I, J):
    return torch.randn(B, C, I, J)


def gen_u(B, H, W, I, J):
    return torch.randn(B, H, W, I, J)


def gen_v(B, C, I, J):
    return torch.randn(B, C, I, J)

def gen_xx(B, C, H, W):
    return torch.randn(B, C, H, W)
    
def gen_yy(W, I):
    return torch.randn(W, I)



def einsum_bchw_bcij____bhwij(s, t):
    return torch.einsum('bchw,bcij->bhwij', s, t)
    # return torch.sum(s[:,:,:,:, None, None] * t[:, :, None, None,:,:], dim=1)


def einsum_bhwij_bcij____bchw(s, t):
    torch.einsum('bhwij,bcij->bchw', s, t)
    # return torch.sum(s[:, None] * t[:, :, None, None], dim=(-1, -2))


def einsum_abcd_df____abcf(s,t):
    return torch.einsum('abcd,df->abcf', s, t)
    # return torch.sum(s[:,:,:,:,None] * t[None,None,None,:,:], dim=-2) 



# Hard coded dictionary to map equations to hard coded einsum implementations
implement_each_einsum = {'bchw,bcij->bhwij': einsum_bchw_bcij____bhwij, 
                         'bhwij,bcij->bchw': einsum_bhwij_bcij____bchw, 
                         'abcd,df->abcf': einsum_abcd_df____abcf}


def compare(library_fun, manual_fun, s, t):
    print(f"\nThe max error is: {torch.max(torch.abs(library_fun(s, t)-manual_fun(s, t)))}")


# if __name__ == '__main__':
#     # some dimensions to use for the test
#     B, C, H, W, I, J = 4, 5, 32, 32, 7, 6
#     print("Testing the manual implementations in this file...\n\n")
#     for attempt in range(10):
#         compare(partial(torch.einsum, 'bchw,bcij->bhwij'), einsum_bchw_bcij____bhwij, s=gen_x(B, C, H, W), t=gen_y(B, C, I, J))
#         compare(partial(torch.einsum, 'bhwij,bcij->bchw'), einsum_bhwij_bcij____bchw, s=gen_u(B, H, W, I, J), t=gen_v(B, C, I, J))
#         compare(partial(torch.einsum, 'abcd,df->abcf'), einsum_abcd_df____abcf, s=gen_xx(B, C, H, W), t=gen_yy(W, I))


