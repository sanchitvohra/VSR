import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------- utility classes --------------------- #
class BicubicUpsample(nn.Module):
    """ A bicubic upsampling class with similar behavior to that in TecoGAN-Tensorflow

        Note that it's different from torch.nn.functional.interpolate and
        matlab's imresize in terms of bicubic kernel and sampling scheme

        Theoretically it can support any scale_factor >= 1, but currently only
        scale_factor = 4 is tested

        References:
            The original paper: http://verona.fi-p.unam.mx/boris/practicas/CubConvInterp.pdf
            https://stackoverflow.com/questions/26823140/imresize-trying-to-understand-the-bicubic-interpolation
    """

    def __init__(self, scale_factor, a=-0.75):
        super(BicubicUpsample, self).__init__()

        # calculate weights
        cubic = torch.FloatTensor([
            [0, a, -2 * a, a],
            [1, 0, -(a + 3), a + 2],
            [0, -a, (2 * a + 3), -(a + 2)],
            [0, 0, a, -a]
        ])  # accord to Eq.(6) in the reference paper

        kernels = [
            torch.matmul(cubic, torch.FloatTensor([1, s, s ** 2, s ** 3]))
            for s in [1.0*d/scale_factor for d in range(scale_factor)]
        ]  # s = x - floor(x)

        # register parameters
        self.scale_factor = scale_factor
        self.register_buffer('kernels', torch.stack(kernels))

    def forward(self, input):
        n, c, h, w = input.size()
        s = self.scale_factor

        # pad input (left, right, top, bottom)
        input = F.pad(input, (1, 2, 1, 2), mode='replicate')

        # calculate output (height)
        kernel_h = self.kernels.repeat(c, 1).view(-1, 1, s, 1)
        output = F.conv2d(input, kernel_h, stride=1, padding=0, groups=c)
        output = output.reshape(
            n, c, s, -1, w + 3).permute(0, 1, 3, 2, 4).reshape(n, c, -1, w + 3)

        # calculate output (width)
        kernel_w = self.kernels.repeat(c, 1).view(-1, 1, 1, s)
        output = F.conv2d(output, kernel_w, stride=1, padding=0, groups=c)
        output = output.reshape(
            n, c, s, h * s, -1).permute(0, 1, 3, 4, 2).reshape(n, c, h * s, -1)

        return output