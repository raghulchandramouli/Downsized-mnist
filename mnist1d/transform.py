import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


def pad(x, padding):
    low, high = padding
    p = low + int(np.random.rand() * (high - low + 1))
    if len(x.shape) == 1:
        return np.concatenate([x, np.zeros((p,))])
    return np.concatenate([x, np.zeros((x.shape[0], p))], axis=-1)


def shear(x, scale=10):
    coeff = scale * (np.random.rand() - 0.5)
    return x - coeff * np.linspace(-0.5, 0.5, len(x))


def translate(x, max_translation):
    k = np.random.choice(max_translation)
    return np.concatenate([x[-k:], x[:-k]])


def corr_noise_like(x, scale):
    return gaussian_filter(scale * np.random.randn(*x.shape), 2)


def iid_noise_like(x, scale):
    return scale * np.random.randn(*x.shape)


def interpolate(x, n):
    scale = np.linspace(0, 1, len(x))
    new_scale = np.linspace(0, 1, n)
    return interp1d(scale, x, axis=0, kind="linear")(new_scale)


def transform(x, y, args, eps=1e-8):
    new_x = pad(x + eps, args.padding)
    new_x = interpolate(new_x, args.template_len + args.padding[-1])
    new_y = interpolate(y, args.template_len + args.padding[-1])
    new_x *= 1 + args.scale_coeff * (np.random.rand() - 0.5)
    new_x = translate(new_x, args.max_translation)

    mask = new_x != 0
    new_x = mask * new_x + (1 - mask) * corr_noise_like(new_x, args.corr_noise_scale)
    new_x = new_x + iid_noise_like(new_x, args.iid_noise_scale)

    new_x = shear(new_x, args.shear_scale)
    new_x = interpolate(new_x, args.final_seq_length)
    new_y = interpolate(new_y, args.final_seq_length)
    return new_x, new_y
