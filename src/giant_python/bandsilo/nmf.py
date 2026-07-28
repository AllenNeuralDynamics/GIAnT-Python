"""Superpixel-space NMF source localization.

Ported from the source-localization block of
``extractSLAP2IntegrationSources.py`` (ref 2519-2908). Given the peak seeds
from :mod:`giant_python.bandsilo.peaks` and the noise-normalized residual /
per-motion ``H`` matrices from :mod:`giant_python.bandsilo.background`, this
module fits Gaussian spatial profiles (``A``) and per-motion temporal weights
(``phi``) that reconstruct the residual in superpixel space.

Each outer iteration:

1. runs motion-binned multiplicative NMF (in random motion order) to get a
   per-motion superpixel target ``X_mots`` (:func:`multiplicative_nmf`),
2. fits the parametric Gaussian profiles to those targets with Adam
   (:func:`fit_gaussian_profiles`),
3. rebuilds ``A`` and least-squares-solves ``phi``
   (:func:`fit_phi_all_motions`),
4. sorts sources by explained variance, and every fourth iteration prunes
   sources whose SNR falls below ``1/3``.

The final per-source low-resolution SNR is returned alongside ``A``, ``phi``,
the fitted ``source_params`` ``[z, y, x, sigma_y, sigma_x, tilt]``, and the
(sorted/pruned) ``source_seeds``.

This block is intentionally stochastic (random motion order via
``torch.randperm`` in NMF and Adam); the pipeline does not seed the RNG. The
``randperm`` calls are preserved in the same order as the reference so a seeded
run reproduces it exactly. The reference's loss-only ``phi`` recomputation and
``print``/warning diagnostics are dropped (they do not affect the outputs and
consume no RNG).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch


def sel_pix_gaussian_profile(
    gaussian_params: torch.Tensor, pixel_coords_tensor: torch.Tensor
) -> torch.Tensor:
    """Render normalized bivariate-Gaussian profiles over selected pixels.

    Parameters
    ----------
    gaussian_params : torch.Tensor of shape (N, 6)
        Per-source ``[z, mu_y, mu_x, sigma_y, sigma_x, tilt]`` (the tilt is
        mapped to a correlation coefficient via ``tanh``).
    pixel_coords_tensor : torch.Tensor of shape (n_sel, 3)
        Selected-pixel ``[z, row, col]`` coordinates.

    Returns
    -------
    torch.Tensor of shape (n_sel, N)
        Column-normalized profiles; each source is confined to its own z-plane
        and to within 3 sigma of its center.
    """
    z_planes = gaussian_params[:, 0].unsqueeze(0)
    y_means = gaussian_params[:, 1].unsqueeze(0)
    x_means = gaussian_params[:, 2].unsqueeze(0)
    y_sigmas = gaussian_params[:, 3].unsqueeze(0)
    x_sigmas = gaussian_params[:, 4].unsqueeze(0)
    corr_coef = torch.tanh(gaussian_params[:, 5].unsqueeze(0))

    y_centered = pixel_coords_tensor[:, 1].unsqueeze(1) - y_means
    x_centered = pixel_coords_tensor[:, 2].unsqueeze(1) - x_means

    z_score_y = y_centered / y_sigmas
    z_score_x = x_centered / x_sigmas

    exponent = (-1 / (2 * (1 - corr_coef**2))) * (
        z_score_y**2 - 2 * corr_coef * z_score_x * z_score_y + z_score_x**2
    )
    profile = torch.exp(exponent)
    profile = (
        profile
        * (
            (torch.sqrt(z_score_y**2 + z_score_x**2) <= 3)
            & (pixel_coords_tensor[:, 0].unsqueeze(1) == z_planes)
        ).float()
    )

    return profile / torch.sum(profile, dim=0, keepdim=True)


def sel_pix_patch_profile(
    patch_params: torch.Tensor, pixel_coords_tensor: torch.Tensor
) -> torch.Tensor:
    """Render rectangular in-plane support patches over selected pixels.

    Parameters
    ----------
    patch_params : torch.Tensor of shape (N, 5)
        Per-source ``[z, mu_y, mu_x, y_radius, x_radius]``.
    pixel_coords_tensor : torch.Tensor of shape (n_sel, 3)
        Selected-pixel ``[z, row, col]`` coordinates.

    Returns
    -------
    torch.Tensor of bool, shape (n_sel, N)
        True where a pixel is inside a source's patch (same z-plane, within the
        radii).
    """
    z_planes = patch_params[:, 0].unsqueeze(0)
    y_means = patch_params[:, 1].unsqueeze(0)
    x_means = patch_params[:, 2].unsqueeze(0)
    y_radii = patch_params[:, 3].unsqueeze(0)
    x_radii = patch_params[:, 4].unsqueeze(0)

    y_centered = pixel_coords_tensor[:, 1].unsqueeze(1) - y_means
    x_centered = pixel_coords_tensor[:, 2].unsqueeze(1) - x_means

    return (
        (torch.abs(y_centered) < y_radii)
        & (torch.abs(x_centered) < x_radii)
        & (pixel_coords_tensor[:, 0].unsqueeze(1) == z_planes)
    )


def init_source_params(source_seeds: np.ndarray) -> torch.Tensor:
    """Initialize ``[z, y, x, sigma_y=1, sigma_x=1, tilt=0]`` per source.

    Parameters
    ----------
    source_seeds : ndarray of shape (N, 3)
        Peak seeds ``[z, mu_y, mu_x]``.

    Returns
    -------
    torch.Tensor of shape (N, 6)
        The initial source parameters.
    """
    n = source_seeds.shape[0]
    return torch.cat(
        [
            torch.tensor(source_seeds, dtype=torch.float32),
            torch.ones(n, 2, dtype=torch.float32),
            torch.zeros(n, 1, dtype=torch.float32),
        ],
        dim=1,
    )


def build_a_patches(
    source_seeds: np.ndarray,
    pixel_coords_tensor: torch.Tensor,
    sel_pix_idxs: np.ndarray,
    n_pixels: int,
    d_xy: float,
) -> torch.Tensor:
    """Build the per-source rectangular support mask over the full pixel grid.

    Parameters
    ----------
    source_seeds : ndarray of shape (N, 3)
        Peak seeds ``[z, mu_y, mu_x]``.
    pixel_coords_tensor : torch.Tensor of shape (n_sel, 3)
        Selected-pixel coordinates.
    sel_pix_idxs : ndarray of int
        Flat selected-pixel indices into the ``n_pixels`` grid.
    n_pixels : int
        Total number of image pixels (``z * rows * cols``).
    d_xy : float
        Half-width of the square support patch, in pixels.

    Returns
    -------
    torch.Tensor of bool, shape (n_pixels, N)
        The support mask.
    """
    n_sources = source_seeds.shape[0]
    a_patches = torch.zeros((n_pixels, n_sources), dtype=torch.bool)
    patch_params = torch.cat(
        [
            torch.tensor(source_seeds, dtype=torch.float32),
            d_xy * torch.ones(n_sources, 2, dtype=torch.float32),
        ],
        dim=1,
    )
    a_patches[sel_pix_idxs, :] = sel_pix_patch_profile(
        patch_params, pixel_coords_tensor
    )
    return a_patches


def build_x_support_mots(
    h_mots: List[torch.Tensor],
    a_patches: torch.Tensor,
    sel_pix_idxs: np.ndarray,
    n_motions: int,
) -> List[torch.Tensor]:
    """Project each source's support patch into per-motion superpixel space.

    Parameters
    ----------
    h_mots : list of torch.Tensor
        Per-motion sparse superpixel<-pixel ``H`` matrices.
    a_patches : torch.Tensor of bool, shape (n_pixels, N)
        Support mask from :func:`build_a_patches`.
    sel_pix_idxs : ndarray of int
        Flat selected-pixel indices.
    n_motions : int
        Number of motion bins.

    Returns
    -------
    list of torch.Tensor of bool
        One ``(n_superpixels, N)`` support mask per motion bin.
    """
    return [
        torch.sparse.mm(h_mots[i], a_patches[sel_pix_idxs, :].float()) > 0
        for i in range(n_motions)
    ]


def project_spatial_profiles(
    source_params: torch.Tensor,
    pixel_coords_tensor: torch.Tensor,
    sel_pix_idxs: np.ndarray,
    n_pixels: int,
    a_patches: torch.Tensor,
) -> torch.Tensor:
    """Build the mass-normalized Gaussian spatial profiles ``A``.

    The Gaussian profile is masked by the support patch and normalized so each
    source's column sums to 1.

    Parameters
    ----------
    source_params : torch.Tensor of shape (N, 6)
        Source parameters.
    pixel_coords_tensor : torch.Tensor of shape (n_sel, 3)
        Selected-pixel coordinates.
    sel_pix_idxs : ndarray of int
        Flat selected-pixel indices.
    n_pixels : int
        Total number of image pixels.
    a_patches : torch.Tensor of bool, shape (n_pixels, N)
        Support mask.

    Returns
    -------
    torch.Tensor of shape (n_pixels, N)
        The normalized spatial profiles.
    """
    n_sources = source_params.shape[0]
    a = torch.zeros((n_pixels, n_sources), dtype=torch.float32)
    a[sel_pix_idxs, :] = sel_pix_gaussian_profile(
        source_params, pixel_coords_tensor
    )
    a[~a_patches] = 0
    mass = torch.sum(a, dim=0, keepdim=True)
    mass[mass <= 0] = 1
    a /= mass
    return a


def _motion_frames(mot_inds_yx: np.ndarray, i: int) -> np.ndarray:
    """Return the frame indices belonging to motion bin ``i``."""
    return np.flatnonzero(mot_inds_yx == i)


def _solve_phi_motion(
    x: torch.Tensor, data_motion: torch.Tensor
) -> torch.Tensor:
    """Least-squares temporal weights for one motion's superpixel profiles."""
    xtx = x.T @ x
    xtd = x.T @ data_motion
    regularized = xtx + 1e-10 * torch.eye(xtx.shape[0])
    return torch.linalg.solve(regularized, xtd).T


def fit_phi_all_motions(
    a: torch.Tensor,
    h_mots: List[torch.Tensor],
    sel_pix_idxs: np.ndarray,
    data_for_nmf: torch.Tensor,
    mot_inds_yx: np.ndarray,
    n_motions: int,
    phi_low_res: torch.Tensor,
) -> torch.Tensor:
    """Least-squares-solve the temporal weights ``phi`` for every motion bin.

    Parameters
    ----------
    a : torch.Tensor of shape (n_pixels, N)
        Spatial profiles.
    h_mots : list of torch.Tensor
        Per-motion sparse ``H`` matrices.
    sel_pix_idxs : ndarray of int
        Flat selected-pixel indices.
    data_for_nmf : torch.Tensor of shape (n_superpixels, n_frames)
        The residual data being reconstructed.
    mot_inds_yx : ndarray of shape (n_frames,)
        Per-frame motion-bin index (``-1`` for dropped frames).
    n_motions : int
        Number of motion bins.
    phi_low_res : torch.Tensor of shape (n_frames, N)
        Temporal weights, updated in place per motion bin and returned.

    Returns
    -------
    torch.Tensor
        ``phi_low_res`` with each motion's frames filled.
    """
    for i in range(n_motions):
        frames = _motion_frames(mot_inds_yx, i)
        x = torch.sparse.mm(h_mots[i], a[sel_pix_idxs, :])
        phi_low_res[frames, :] = _solve_phi_motion(x, data_for_nmf[:, frames])
    return phi_low_res


def multiplicative_nmf(
    a: torch.Tensor,
    h_mots: List[torch.Tensor],
    x_support_mots: List[torch.Tensor],
    sel_pix_idxs: np.ndarray,
    data_for_nmf: torch.Tensor,
    mot_inds_yx: np.ndarray,
    n_motions: int,
    sparse_fac: float,
    max_iters: int,
) -> List[torch.Tensor]:
    """Motion-binned multiplicative NMF, producing per-motion spatial targets.

    Visits motion bins in a random order (``torch.randperm``). For each bin it
    projects ``A`` into superpixel space, initializes ``phi`` by projected
    least squares, then alternates multiplicative updates of ``phi`` and the
    spatial
    target ``X`` under the support constraint, periodically sparsifying ``X``.

    Parameters
    ----------
    a : torch.Tensor of shape (n_pixels, N)
        Current spatial profiles.
    h_mots : list of torch.Tensor
        Per-motion sparse ``H`` matrices.
    x_support_mots : list of torch.Tensor of bool
        Per-motion superpixel support masks.
    sel_pix_idxs : ndarray of int
        Flat selected-pixel indices.
    data_for_nmf : torch.Tensor of shape (n_superpixels, n_frames)
        Residual data.
    mot_inds_yx : ndarray of shape (n_frames,)
        Per-frame motion-bin index.
    n_motions : int
        Number of motion bins.
    sparse_fac : float
        Sparsification floor subtracted from the max-normalized ``X``.
    max_iters : int
        Multiplicative-update iterations per motion bin.

    Returns
    -------
    list of torch.Tensor
        Per-motion fitted superpixel spatial targets ``X_mots``.
    """
    x_mots: List[torch.Tensor] = [None] * n_motions
    shuffled_indices = torch.randperm(n_motions)
    for idx in shuffled_indices:
        i = idx.item()
        frames = _motion_frames(mot_inds_yx, i)

        x = torch.sparse.mm(h_mots[i], a[sel_pix_idxs, :])
        x[~x_support_mots[i]] = 0
        x = x / torch.norm(x, dim=0, keepdim=True)

        phi = torch.clamp(_solve_phi_motion(x, data_for_nmf[:, frames]), min=0)

        for iter_idx in range(max_iters):
            numerator = x.T @ data_for_nmf[:, frames]
            denominator = (x.T @ x) @ phi.T + 1e-10
            phi = torch.clamp(phi * (numerator / denominator).T, min=0)

            numerator = data_for_nmf[:, frames] @ phi
            denominator = x @ (phi.T @ phi) + 1e-10
            x = torch.clamp(x * (numerator / denominator), min=0)

            x = x * x_support_mots[i].float()

            norms = torch.norm(x, dim=0, keepdim=True)
            x = torch.where(norms > 0, x / norms, x)
            phi = torch.where(norms > 0, phi * norms, phi)

            if iter_idx % 3 == 0:
                x_max = torch.max(x, dim=0, keepdim=True)[0]
                x = torch.where(x_max > 0, x / x_max, x)
                x = torch.clamp(x, min=sparse_fac) - sparse_fac
                norms = torch.norm(x, dim=0, keepdim=True)
                x = torch.where(norms > 0, x / norms, x)

        x_mots[i] = x
    return x_mots


def fit_gaussian_profiles(
    source_params: torch.Tensor,
    source_seeds: np.ndarray,
    x_mots: List[torch.Tensor],
    h_mots: List[torch.Tensor],
    pixel_coords_tensor: torch.Tensor,
    a_patches: torch.Tensor,
    sel_pix_idxs: np.ndarray,
    d_xy: float,
    num_epochs: int,
    n_motions: int,
    learning_rate: float,
    gd_tol: float,
) -> torch.Tensor:
    """Adam-fit the Gaussian source parameters to the per-motion NMF targets.

    Optimizes location, scale, and tilt to make the projected Gaussian profiles
    match ``X_mots`` (one motion per epoch, in random rounds). Locations are
    clamped to +/- ``d_xy`` of the seed and sigmas to ``[0.3, 5]``. Stops early
    when consecutive full-round losses differ by less than ``gd_tol``.

    Parameters
    ----------
    source_params : torch.Tensor of shape (N, 6)
        Current source parameters (z fixed; loc/scale/tilt optimized).
    source_seeds : ndarray of shape (N, 3)
        Peak seeds (for the location clamp bounds).
    x_mots : list of torch.Tensor
        Per-motion NMF spatial targets.
    h_mots : list of torch.Tensor
        Per-motion sparse ``H`` matrices.
    pixel_coords_tensor : torch.Tensor of shape (n_sel, 3)
        Selected-pixel coordinates.
    a_patches : torch.Tensor of bool
        Support mask.
    sel_pix_idxs : ndarray of int
        Flat selected-pixel indices.
    d_xy : float
        Location clamp half-width.
    num_epochs : int
        Maximum optimization epochs (one motion each).
    n_motions : int
        Number of motion bins (epochs per round).
    learning_rate : float
        Base Adam learning rate (scaled per parameter group).
    gd_tol : float
        Convergence tolerance on consecutive round losses.

    Returns
    -------
    torch.Tensor of shape (N, 6)
        The updated source parameters.
    """
    optim_loc = source_params[:, 1:3].clone().requires_grad_(True)
    optim_scale = source_params[:, 3:5].clone().requires_grad_(True)
    optim_tilt = source_params[:, 5].unsqueeze(1).clone().requires_grad_(True)
    optimizer = torch.optim.Adam(
        [
            {"params": optim_loc, "lr": 10 * learning_rate},
            {"params": optim_scale, "lr": 0.1 * learning_rate},
            {"params": optim_tilt, "lr": 0.1 * learning_rate},
        ]
    )

    z_col = source_params[:, 0].unsqueeze(1)
    loc_min = torch.from_numpy(source_seeds[:, 1:3] - d_xy).float()
    loc_max = torch.from_numpy(source_seeds[:, 1:3] + d_xy).float()

    losses = []
    shuffled_indices = None
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        a_step = sel_pix_gaussian_profile(
            torch.cat([z_col, optim_loc, optim_scale, optim_tilt], dim=1),
            pixel_coords_tensor,
        )
        a_step = a_step * a_patches[sel_pix_idxs, :].float()
        mass = torch.sum(a_step, dim=0, keepdim=True)
        mass[mass <= 0] = 1
        a_step = a_step / mass

        if epoch % n_motions == 0:
            shuffled_indices = torch.randperm(n_motions)
        i = shuffled_indices[epoch % n_motions].item()

        x_step = torch.sparse.mm(h_mots[i], a_step)
        norms = torch.norm(x_step, dim=0, keepdim=True)
        x_step = torch.where(norms > 0, x_step / norms, x_step)
        loss = torch.sum((x_step - x_mots[i]) ** 2)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            optim_loc.clamp_(min=loc_min, max=loc_max)
            optim_scale[:, 0].clamp_(min=0.3, max=5)
            optim_scale[:, 1].clamp_(min=0.3, max=5)

        if epoch % n_motions == n_motions - 1:
            total_loss = 0
            for i in range(n_motions):
                x_step = torch.sparse.mm(h_mots[i], a_step)
                norms = torch.norm(x_step, dim=0, keepdim=True)
                x_step = torch.where(norms > 0, x_step / norms, x_step)
                total_loss += torch.sum((x_step - x_mots[i]) ** 2)
            losses.append(total_loss.item())
            if (
                epoch // n_motions > 0
                and abs(losses[-1] - losses[-2]) < gd_tol
            ):
                break

    return torch.cat(
        [z_col, optim_loc.detach(), optim_scale.detach(), optim_tilt.detach()],
        dim=1,
    )


def variance_sortorder(
    phi_low_res: torch.Tensor, n_sources: int
) -> np.ndarray:
    """Return the descending-explained-variance source ordering."""
    phi = phi_low_res[:, :n_sources].numpy()
    variance = np.nansum((phi - np.nanmean(phi, axis=0)) ** 2, axis=0)
    return np.argsort(-variance)


def reorder_sources(
    order,
    source_params: torch.Tensor,
    source_seeds: np.ndarray,
    a: torch.Tensor,
    a_patches: torch.Tensor,
    x_support_mots: List[torch.Tensor],
    phi_low_res: torch.Tensor,
):
    """Apply a source ordering / selection to every per-source array.

    Parameters
    ----------
    order : array-like of int
        Source indices in the new order (a permutation for sorting, or a subset
        for pruning).
    source_params, source_seeds, a, a_patches, phi_low_res : array/tensor
        Per-source state to reindex along the source axis.
    x_support_mots : list of torch.Tensor
        Per-motion support masks (reindexed along the source axis).

    Returns
    -------
    tuple
        The reindexed ``(source_params, source_seeds, a, a_patches,
        x_support_mots, phi_low_res)``.
    """
    source_params = source_params[order, :]
    source_seeds = source_seeds[order, :]
    a = a[:, order]
    a_patches = a_patches[:, order]
    x_support_mots = [support[:, order] for support in x_support_mots]
    phi_low_res = phi_low_res[:, order]
    return (
        source_params,
        source_seeds,
        a,
        a_patches,
        x_support_mots,
        phi_low_res,
    )


def compute_source_snr(
    a: torch.Tensor,
    h_mots: List[torch.Tensor],
    sel_pix_idxs: np.ndarray,
    data_for_nmf: torch.Tensor,
    phi_low_res: torch.Tensor,
    mot_inds_yx: np.ndarray,
    n_motions: int,
    n_sources: int,
) -> torch.Tensor:
    """Per-source low-resolution SNR (explained variance / residual variance).

    Reconstructs the residual from ``A`` and ``phi``, then for each source sums
    the squared explained signal and squared residual over the superpixels the
    source contributes to.

    Parameters
    ----------
    a : torch.Tensor of shape (n_pixels, N)
        Spatial profiles.
    h_mots : list of torch.Tensor
        Per-motion sparse ``H`` matrices.
    sel_pix_idxs : ndarray of int
        Flat selected-pixel indices.
    data_for_nmf : torch.Tensor of shape (n_superpixels, n_frames)
        Residual data.
    phi_low_res : torch.Tensor of shape (n_frames, N)
        Temporal weights.
    mot_inds_yx : ndarray of shape (n_frames,)
        Per-frame motion-bin index.
    n_motions, n_sources : int
        Number of motion bins / sources.

    Returns
    -------
    torch.Tensor of shape (N,)
        Per-source SNR (``varExp / varResidual``).
    """
    residual_recon = torch.full_like(data_for_nmf, float("nan"))
    for i in range(n_motions):
        frames = _motion_frames(mot_inds_yx, i)
        x = torch.sparse.mm(h_mots[i], a[sel_pix_idxs, :])
        residual_recon[:, frames] = (
            data_for_nmf[:, frames] - x @ phi_low_res[frames, :].T
        )

    var_exp = torch.zeros(n_sources, dtype=torch.float32)
    var_residual = torch.zeros(n_sources, dtype=torch.float32)
    for j in range(n_sources):
        for i in range(n_motions):
            frames = _motion_frames(mot_inds_yx, i)
            x = torch.sparse.mm(h_mots[i], a[sel_pix_idxs, j].unsqueeze(1))
            contributing = (x > 0).nonzero()[:, 0]
            var_exp[j] += torch.sum(
                torch.sum(
                    x[contributing] * phi_low_res[frames, j].unsqueeze(-1).T,
                    dim=0,
                )
                ** 2
            )
            var_residual[j] += torch.sum(
                torch.sum(residual_recon[contributing][:, frames], dim=0) ** 2
            )
    # TODO: maybe consider making this a peak SNR ratio
    return var_exp / var_residual


def fit_sources(
    source_seeds: np.ndarray,
    residual: np.ndarray,
    h_mots: List[torch.Tensor],
    unique_motion_to_keep_yx: np.ndarray,
    mot_inds_yx: np.ndarray,
    sel_pix_idxs: np.ndarray,
    pixel_coords: np.ndarray,
    n_pixels: int,
    d_xy: float,
    sparse_fac: float,
    outer_loop_iters: int = 10,
    mult_nmf_max_iters: int = 10,
    learning_rate: float = 0.01,
    gd_tol: float = 1e-4,
) -> dict:
    """Localize sources by superpixel-space NMF + Gaussian-profile fitting.

    Parameters
    ----------
    source_seeds : ndarray of shape (N, 3)
        Peak seeds ``[z, mu_y, mu_x]`` from :mod:`giant_python.bandsilo.peaks`.
    residual : ndarray of shape (n_superpixels, n_frames)
        Noise-normalized residual to reconstruct.
    h_mots : list of torch.Tensor
        Per-motion sparse ``H`` matrices.
    unique_motion_to_keep_yx : ndarray of shape (n_motions, 2)
        Kept 2-D motion vectors (only its length is used).
    mot_inds_yx : ndarray of shape (n_frames,)
        Per-frame motion-bin index (``-1`` for dropped frames).
    sel_pix_idxs : ndarray of int
        Flat selected-pixel indices into the ``n_pixels`` grid.
    pixel_coords : ndarray of shape (n_sel, 3)
        Selected-pixel ``[z, row, col]`` coordinates.
    n_pixels : int
        Total number of image pixels.
    d_xy : float
        Support-patch half-width / location clamp radius.
    sparse_fac : float
        NMF sparsification floor.
    outer_loop_iters : int
        Number of outer NMF/fit iterations.
    mult_nmf_max_iters : int
        Multiplicative-update iterations per motion bin.
    learning_rate : float
        Base Adam learning rate.
    gd_tol : float
        Adam convergence tolerance.

    Returns
    -------
    dict
        ``A`` (n_pixels, N), ``phi_low_res`` (n_frames, N), ``source_params``
        (N, 6), ``source_seeds`` (N, 3), ``source_snr`` (N,), and
        ``n_sources``.
    """
    n_sources = source_seeds.shape[0]
    n_motions = unique_motion_to_keep_yx.shape[0]
    n_frames = residual.shape[1]
    pixel_coords_tensor = torch.tensor(pixel_coords, dtype=torch.float32)
    data_for_nmf = torch.from_numpy(residual.astype(np.float32, copy=False))
    num_epochs = n_motions * 5

    source_params = init_source_params(source_seeds)
    a_patches = build_a_patches(
        source_seeds, pixel_coords_tensor, sel_pix_idxs, n_pixels, d_xy
    )
    x_support_mots = build_x_support_mots(
        h_mots, a_patches, sel_pix_idxs, n_motions
    )
    a = project_spatial_profiles(
        source_params, pixel_coords_tensor, sel_pix_idxs, n_pixels, a_patches
    )
    phi_low_res = torch.full(
        (n_frames, n_sources), float("nan"), dtype=torch.float32
    )

    for outer_loop_iter in range(outer_loop_iters):
        x_mots = multiplicative_nmf(
            a,
            h_mots,
            x_support_mots,
            sel_pix_idxs,
            data_for_nmf,
            mot_inds_yx,
            n_motions,
            sparse_fac,
            mult_nmf_max_iters,
        )
        source_params = fit_gaussian_profiles(
            source_params,
            source_seeds,
            x_mots,
            h_mots,
            pixel_coords_tensor,
            a_patches,
            sel_pix_idxs,
            d_xy,
            num_epochs,
            n_motions,
            learning_rate,
            gd_tol,
        )
        a = project_spatial_profiles(
            source_params,
            pixel_coords_tensor,
            sel_pix_idxs,
            n_pixels,
            a_patches,
        )
        phi_low_res = fit_phi_all_motions(
            a,
            h_mots,
            sel_pix_idxs,
            data_for_nmf,
            mot_inds_yx,
            n_motions,
            phi_low_res,
        )

        order = variance_sortorder(phi_low_res, n_sources)
        (
            source_params,
            source_seeds,
            a,
            a_patches,
            x_support_mots,
            phi_low_res,
        ) = reorder_sources(
            order,
            source_params,
            source_seeds,
            a,
            a_patches,
            x_support_mots,
            phi_low_res,
        )

        if (outer_loop_iter + 1) % 4 == 3:
            snr = compute_source_snr(
                a,
                h_mots,
                sel_pix_idxs,
                data_for_nmf,
                phi_low_res,
                mot_inds_yx,
                n_motions,
                n_sources,
            )
            keep = (snr > 1 / 3).nonzero()[:, 0]
            n_sources = keep.shape[0]
            (
                source_params,
                source_seeds,
                a,
                a_patches,
                x_support_mots,
                phi_low_res,
            ) = reorder_sources(
                keep,
                source_params,
                source_seeds,
                a,
                a_patches,
                x_support_mots,
                phi_low_res,
            )

    source_snr = compute_source_snr(
        a,
        h_mots,
        sel_pix_idxs,
        data_for_nmf,
        phi_low_res,
        mot_inds_yx,
        n_motions,
        n_sources,
    ).numpy()

    return {
        "A": a,
        "phi_low_res": phi_low_res,
        "source_params": source_params,
        "source_seeds": source_seeds,
        "source_snr": source_snr,
        "n_sources": n_sources,
    }
