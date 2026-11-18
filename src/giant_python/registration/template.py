"""Registration templates: the initial consensus template and its growth.

The initial template follows GIAnT-MATLAB ``MultiRoiRegistration``: frames
spread over the trial are matched pairwise; the frame with the highest median
correlation is the reference, and the frames correlating with it at least as
well as that median are shifted onto it and averaged. The running template is
the square root of the mean of all registered frames added so far (pixels
with fewer than ``min_samples`` contributions are NaN); frames are matched to
the average of the initial and the running template.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .interpolation import interp_frame
from .matching import match


@dataclass
class ConsensusResult:
    """Outcome of the pairwise matching of the initial frames.

    Attributes
    ----------
    correlation : ndarray
        ``(n, n)`` peak correlations between frames (1 on the diagonal).
    motion : ndarray
        ``(2, n, n)`` [row, col] motion; ``motion[:, i, j]`` matches frame
        ``j`` to frame ``i`` as template.
    reference : int
        Zero-based index of the reference frame.
    members : ndarray of int
        Zero-based indices of the frames averaged into the template.
    """

    correlation: np.ndarray
    motion: np.ndarray
    reference: int
    members: np.ndarray


def window_image(values: torch.Tensor, index: torch.Tensor, shape, fill):
    """Place per-pixel values into ``(..., H, W)`` images."""
    lead = values.shape[:-1]
    out = torch.full(
        lead + (shape[0] * shape[1],),
        fill,
        dtype=values.dtype,
        device=values.device,
    )
    out[..., index] = values
    return out.reshape(lead + tuple(shape))


def consensus(
    frames: torch.Tensor,
    freshness: torch.Tensor,
    images: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    search: int,
    dtype=torch.float32,
) -> ConsensusResult:
    """Match every pair of initial frames and choose the consensus set.

    Parameters
    ----------
    frames, freshness : torch.Tensor
        ``(n, pixels)`` frame values (NaN where invalid) and freshness.
    images : torch.Tensor
        ``(n, H, W)`` the same frames as window images.
    rows, cols : torch.Tensor of int
        Zero-based pixel positions in the window.
    search : int
        Largest pairwise shift.
    dtype : torch.dtype
        Precision of the correlation sums.

    Returns
    -------
    ConsensusResult
        Pairwise correlations and motion, reference and members.
    """
    n = frames.shape[0]
    corr = np.ones((n, n))
    mot = np.zeros((2, n, n))
    weights = torch.where(torch.isnan(frames), 0.0, freshness)
    for f1 in range(n - 1):
        m, r = match(
            frames[f1 + 1 :],
            weights[f1 + 1 :],
            images[f1].to(torch.float64),
            rows,
            cols,
            search,
            dtype,
        )
        m = m.cpu().numpy()
        r = r.cpu().numpy()
        mot[:, f1, f1 + 1 :] = m.T
        mot[:, f1 + 1 :, f1] = -m.T
        corr[f1, f1 + 1 :] = r
        corr[f1 + 1 :, f1] = r
    med = np.median(corr, axis=0)
    ref = int(np.argmax(med))
    members = np.flatnonzero(corr[:, ref] >= med[ref])
    return ConsensusResult(corr, mot, ref, members)


class TemplateState:
    """Running sums of registered frames and the template they give.

    Parameters
    ----------
    total : torch.Tensor
        ``(H + 2m, W + 2m)`` float64 sum of registered frames (NaN skipped).
    count : torch.Tensor
        Same shape: number of frames contributing to each pixel.
    initial : torch.Tensor
        The initial template ``T0``.
    min_samples : int
        Pixels with fewer contributions are NaN in the template.
    """

    def __init__(self, total, count, initial, min_samples: int):
        """Store the sums and compute the template."""
        self.total = total
        self.count = count
        self.initial = initial
        self.min_samples = int(min_samples)
        self.template = self._template()

    def _template(self) -> torch.Tensor:
        """``sqrt(total / count)``, NaN below ``min_samples``."""
        t = torch.sqrt(self.total / self.count)
        return t.masked_fill(self.count < self.min_samples, float("nan"))

    def add(self, image: torch.Tensor) -> None:
        """Add one registered frame (NaN pixels are skipped)."""
        sel = ~torch.isnan(image)
        self.total = self.total + torch.where(sel, image, 0.0).to(
            self.total.dtype
        )
        self.count = self.count + sel
        self.template = self._template()

    def update(self, images: torch.Tensor, sign: int = 1) -> None:
        """Add (``sign=1``) or remove (``-1``) a stack of registered frames.

        The template is recomputed once, after all frames.
        """
        if images.shape[0] == 0:
            return
        sel = ~torch.isnan(images)
        z = torch.where(sel, images, 0.0).to(self.total.dtype)
        self.total = self.total + sign * z.sum(0)
        self.count = self.count + sign * sel.sum(0)
        self.template = self._template()

    def matching_template(self) -> torch.Tensor:
        """Mean of the initial and running templates, ignoring NaNs."""
        a, b = self.initial, self.template
        both = (a + b) / 2
        out = torch.where(torch.isnan(a), b, both)
        return torch.where(torch.isnan(b), a, out)


def initial_template(
    images: torch.Tensor,
    fresh_images: torch.Tensor,
    result: ConsensusResult,
    margin: int,
    min_samples: int,
) -> TemplateState:
    """Average the consensus frames, shifted onto the reference frame.

    Parameters
    ----------
    images, fresh_images : torch.Tensor
        ``(n, H, W)`` initial frames (NaN where unsampled) and freshness
        (0 where unsampled).
    result : ConsensusResult
        Output of :func:`consensus`.
    margin : int
        Padding of the output grid (``maxshift``).
    min_samples : int
        Minimum contributions for a template pixel.

    Returns
    -------
    TemplateState
        Sums of the shifted consensus frames, with ``T0`` as both initial
        and running template.
    """
    ref = result.reference
    stack = []
    for i in result.members:
        img, _ = interp_frame(
            images[i],
            fresh_images[i],
            -result.motion[0, i, ref],
            -result.motion[1, i, ref],
            margin,
        )
        stack.append(img.to(torch.float64))
    stack = torch.stack(stack)
    valid = ~torch.isnan(stack)
    total = torch.where(valid, stack, 0.0).sum(0)
    count = valid.sum(0).to(torch.float64)
    state = TemplateState(total, count, None, min_samples)
    state.initial = state.template.clone()
    return state
