from torch import Tensor

# class SurfaceLoss():
#     def __init__(self, **kwargs):
#         print(f"Initialized {self.__class__.__name__}")

#     def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
#         pc = probs[:, self.idc, ...].type(torch.float32)
#         dc = dist_maps[:, self.idc, ...].type(torch.float32)

#         multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

#         loss = multipled.mean()

#         return loss

def surface_loss(y_hat: Tensor, y_dist: Tensor):
    multipled = y_hat * y_dist

    loss = multipled.mean()

    return loss