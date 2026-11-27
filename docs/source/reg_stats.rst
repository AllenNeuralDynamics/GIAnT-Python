Registration summary statistics
===============================

``giant register`` writes, next to each registered movie, a file of per-trial
summary statistics, ``<stem>_REGSTATS-80Hz.h5`` (``stats_version`` 1). They
are maps of the registered 80 Hz movie alone, made for the next version of the
F0 network (``slap2-F0-network``): how much each pixel varies beyond photon
noise slowly (drift, bleaching, slow motion) and fast (events), whether its
fast changes are positive transients, and how it follows the trial's shared
modes. Every moment comes with its expectation under photon noise alone at
that pixel, so that the normalised maps read 0 for photon noise at any
brightness, and both are stored, so that later versions can renormalise
without the movie.

.. code-block:: bash

   giant register path/to/trial_table.h5            # writes the statistics too
   giant register path/to/trial_table.h5 --no-stats # does not
   giant reg-stats path/to/trial_table.h5           # for trials registered earlier

``giant reg-stats`` computes the same maps for movies registered earlier,
GIAnT-Python's (HDF5 or TIFF) and GIAnT-MATLAB's: it reads each trial's raw
recording again (for the photon-noise model), then the registered movie and
``varFacDS`` frame by frame, and passes them through the same code as
registration, so the maps are equal (bit for bit on the same device; across
CPU and GPU to floating-point rounding). Frame times, motion, registration
window and online compensation come from the alignment file and are checked
against the raw recording. Options: ``--out DIR`` (write the files there
instead of next to the movies), ``--datadr`` and ``--moco`` (folders that
moved since the trial table was written), ``--device``, ``--max-workers``,
``--overwrite`` (statistics that record the same version and registration
are otherwise kept).

From Python:

.. code-block:: python

   from giant_python.registration.stats import read_stats, combine_trials

   st = read_stats("E1T1DMD1_REGSTATS-80Hz.h5")
   st["maps"]["fast_skew_rel"]          # (channels, rows, cols)

Registration's other outputs do not change: the movie and the alignment file
are bit-identical with and without statistics, and nothing that reads them
(GIAnT-Python's loaders, ``slap2_f0``'s reader, MATLAB) sees the new file. A
registration that skips a trial because its outputs exist leaves the trial's
statistics as they are; one with ``--no-stats`` removes an existing file of
the trial, which would describe the earlier registration. If the statistics
fail, registration logs the error and publishes the movie without them.

Definitions
-----------

All statistics are per trial, per DMD and per channel, on the registered grid
(``meanIM``'s rows and columns), over every registered frame (SILo's motion
discards are not applied; they are made later). A pixel's value in a frame is
*valid* where the movie is not NaN. Frames are numbered from 0.

Single frame
   ``n_valid``: valid frames. ``mean``: the mean of the valid values, formed
   in float32 exactly as registration forms ``meanIM`` (identical to it).
   ``var``: ``mean(x^2) - mean(x)^2`` over the valid frames (population
   variance, in float64). ``zero_share``: the share of valid frames that are
   exactly 0.

Slow part
   Blocks of 64 frames from frame 0; a trailing partial block is dropped. A
   block is valid at a pixel with at least 48 valid frames; its value is the
   mean of those. ``slow_var``: the variance of the valid blocks' means with
   ``J - 1`` in the denominator (``J``: valid blocks, ``n_blocks``).

Fast part
   ``z_t = x_t - (mean of the valid frames t - 32 .. t + 32)`` for centres
   ``32 <= t <= N - 33`` (the 65-frame window inside the trial) whose own
   frame is valid and whose window has at least 49 valid frames.
   ``fast_var`` and ``fast_m3``: the second and third central moments of
   ``z`` over those centres (``n_fast``).

Shared modes (``svd/``)
   8-frame time bins from frame 0 (a trailing partial bin dropped); a
   pixel's bin is the mean of its valid frames, missing with fewer than 6.
   2 x 2 pixel bins from pixel (0, 0) (a trailing odd row or column
   dropped); a bin is the mean of its four pixels' bins, missing unless all
   four are present. Spatial bins without a missing time bin enter the SVD,
   each centred on its mean over time and divided by its photon-noise SD
   (below). The top three components of ``Y = U S V^T`` are kept, each
   signed so that its spatial vector sums to a non-negative value. Stored:
   ``time_courses`` (unit norm), ``singular_values``, ``var_share``
   (``s^2 / sum(Y^2)``), ``skew`` and ``lag1`` (skewness and lag-1
   autocorrelation of each time course), ``total_var``, ``n_space``,
   ``n_bins``, and every pixel's ``weight``: the coefficient of each time
   course in a least-squares fit of the pixel's own 8-frame bin means (its
   valid bins, at least 64, with an intercept), in movie units per unit of
   the unit-norm time course; ``weight_ph`` divides it by the pixel's
   ``sigma8_ph``.

Photon-noise model
------------------

A raw SLAP2 sample is the summed height of the photon pulses that arrived
during it, and pulse heights vary (about 1 to 250 raw units per photon). A
sample of mean ``mu`` therefore has, under photon noise alone, variance
``g2 mu`` and third cumulant ``g3 mu`` (compound Poisson; ``g2 = E[A^2] /
E[A]``, ``g3 = E[A^3] / E[A]`` for pulse height ``A``). Both depend on
brightness on SLAP2 (``g2`` from about 110-120 in the dimmest slots to
130-140), so each trial measures them per channel as a curve over brightness
(``gain/``): for every slot and every three successive cycles with valid
samples, the second difference ``d = 2 x1 - x0 - x2`` has ``E[d^2] = 6
kappa2`` and ``E[d^3] = 6 kappa3`` and does not see linear drift; slots are
sorted by their mean and split into 16 groups of equal size, and per group
``g2 = sum(d^2) / (6 sum(x1))``, ``g3 = sum(d^3) / (6 sum(x1))`` (exact
integer sums over all cycles the trial's frames read). A pixel's gains
(``gain_g2``, ``gain_g3``) are interpolated linearly in ``log(mean)`` and
held constant beyond the first and last group. The rate of non-zero
samples, ``nu = -log(share of zeros)``, is stored per unit mean.

A registered frame is linear in the raw samples: frame formation weights the
samples of raw pixel ``k`` with ``w_s(t) = exp(-|t - l_s| / dt)`` within
``max(3 dt, P)`` lines, and registration resamples by a rigid shift with
bilinear coefficients weighted by the raw pixels' summed weights, so that
``x_t(p) = sum_s a_ts x_s`` with ``a_ts = c_t(ab) w_s(t) / D_t(p)`` and
``D_t(p) = 1 / varFacDS``. With the samples independent and of the pixel's
mean ``mu``,

.. code-block:: text

   Cov(x_t, x_u)      = g2 mu A2(t, u),     A2(t, u)    = sum_s a_ts a_us
   kappa3(x_t, x_t, x_u) = g3 mu A3(t, t, u), A3(t, t, u) = sum_s a_ts^2 a_us

Frames ``t`` and ``t + l`` share samples for ``l <= n_lags`` (6 at 80 Hz
for rasters of up to 402 lines per cycle). These sums are computed exactly,
frame by frame, from the raster's layout, the frame times and each frame's
motion, assuming every sample in reach is valid; they are rescaled by the
actual ``D_t(p)`` from ``varFacDS`` (where samples were invalid the
remaining ones are taken as proportional). Each moment's expectation then
follows from its estimator:

``var_q``
   ``S0 / n - (S0 + 2 S1) / n^2`` with ``S0 = sum_t A2(t, t)`` and ``S1``
   the sum of ``A2`` over pairs of valid frames ``t < u`` (the second term
   is the variance of the mean).
``slow_q``
   ``(sum_j V_j - (sum_j V_j + 2 sum_j C_j) / J) / (J - 1)``, with ``V_j``
   the photon variance of block mean ``j`` and ``C_j`` the covariance of
   adjacent blocks' means (from frames that share samples across the block
   boundary).
``fast_q``
   the mean over centres of ``A2(t, t) - 2 L_t / n_t + P_t / n_t^2``,
   where ``L_t`` sums ``A2(t, u)`` over the window and ``P_t`` over pairs
   of window frames (exact).
``fast_q3``
   the mean over centres of ``R0 - 3 K_t / n_t + 3 L_t^2 / n_t^2 -
   sum_u L_u^2 / n_t^3`` with ``R0 = A3(t, t, t)`` and ``K_t = sum_u
   A3(t, t, u)``; the last two terms (products of three coefficients of
   which two belong to other frames, about 0.5% of the total) are
   approximated, within 0.25% of the exact third cumulant on synthetic
   layouts.
``sigma8_ph``
   ``sqrt(g2 mu v8)`` with ``v8`` the mean over valid 8-frame bins of the
   photon variance of the bin's mean; a 2 x 2 bin's photon variance is
   ``spatial_factor * sum sigma8^2 / 16``, where ``spatial_factor`` (an
   attribute) is the trial's mean of ``(1 + rho_r)(1 + rho_c)``, ``rho =
   f (1 - f) / ((1 - f)^2 + f^2)`` for the frame's fractional shifts ``f``,
   by which bilinear interpolation's shared raw pixels raise the variance of
   a 2 x 2 sum (equal raw pixels assumed).
``zero_ph``
   ``sum_N h(N) exp(-N nu) / n``: a frame is exactly 0 when all of its
   ``N`` contributing samples are, and ``h`` counts the valid frames by
   ``N``.

For each moment ``m`` the file holds ``m``, ``m_q`` (the expectation per
unit of ``g mu``: a property of frame formation and registration only),
``m_ph = g mu m_q`` and the normalised map:

* ``var_rel = var / var_ph - 1``, ``slow_rel``, ``fast_rel`` likewise
  (``sqrt(1 + rel)`` is the SD relative to photon noise);
* ``fast_skew_rel = (fast_m3 - fast_m3_ph) / fast_var_ph^1.5``: the third
  moment beyond photon noise, in units of the photon-noise SD;
* ``zero_excess = zero_share - zero_ph``.

Under photon noise all read 0. A ratio estimated from few blocks is
unbiased, but its median is below 0 (as for a chi-square: about -0.02 for
34 blocks).

**Approximations.** The expectation is that of a pixel whose brightness is
its mean and the same for the 2 x 2 raw pixels it interpolates: at sharp
edges, under sub-pixel motion, it is off by the difference between the
coefficients' and their squares' weighting (and real motion at edges adds
true slow and fast variance). With online compensation, a sample that two
frames share is placed by the earlier frame's displacement (they differ only
around an x/y switch). Rasters whose frames share samples over more than 7
frames (cycles longer than about 530 lines at 80 Hz) are not supported.

Session maps
------------

Registered trials do not share a frame (each is registered to a template of
its own frames), so the files hold per-trial maps.
:func:`~giant_python.registration.stats.session.combine_trials` combines
them given SILo's across-trial alignment (``ExtractionPlan.warp``, or rigid
shifts): each trial's maps are resampled into the consensus frame and
averaged, each weighted by the frames behind it (``n_valid``, ``n_blocks``,
``n_fast``); the normalised maps are formed from the averaged moments and
expectations. The shared modes are not combined.

File layout
-----------

Arrays are in GIAnT's MATLAB layout, as the movie: a map that is
``(channels, rows, cols)`` in Python is the HDF5 dataset ``(channels, cols,
rows)``, which MATLAB's ``h5read`` returns as ``rows x cols x channels``;
``read_stats`` returns Python's orientation. Maps are float32, NaN where a
pixel never had data or has too few frames (64 valid frames or fast
centres, 8 valid blocks)::

   <stem>_REGSTATS-80Hz.h5
   ├── (attributes) stats_version, layout, created, source (registration |
   │   reg-stats), movie, movie_size, alignment, alignment_digest (SHA-256
   │   of DSframes, motionDSr, motionDSc), align_hz, n_frames, n_lags,
   │   n_index, spatial_factor, device
   ├── maps/
   │   ├── n_valid, mean, var, var_q, var_ph, var_rel
   │   ├── zero_share, zero_ph, zero_excess
   │   ├── n_blocks, slow_var, slow_q, slow_var_ph, slow_rel
   │   ├── n_fast, fast_var, fast_q, fast_var_ph, fast_rel,
   │   │   fast_m3, fast_q3, fast_m3_ph, fast_skew_rel
   │   └── gain_g2, gain_g3, sigma8_ph
   ├── svd/
   │   ├── time_courses      (channels, 3, time bins)
   │   ├── singular_values, var_share, skew, lag1   (channels, 3)
   │   ├── total_var, n_space (channels,), n_bins
   │   └── weight, weight_ph (channels, 3, cols, rows)
   └── gain/
       └── mu, g2, g3, nu_per_mu, n_slots   (channels, groups)

Cost
----

The statistics are formed while registration writes the movie, on the
registration's device. See the pull request that added them for timings and
memory on a 780125 trial.
