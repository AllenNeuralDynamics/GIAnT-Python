"""Batched registration that grows the template outward from seed frames.

An alternative to MATLAB's strictly sequential frame loop that keeps its
flavour (each frame starts from an aligned neighbour's motion and is matched
to a template of the frames aligned so far) while matching many frames at
once:

1. *Seeds.* The DS frames nearest the consensus frames of the initial
   template are matched first, starting from their consensus motion. Each
   becomes an *island* of aligned frames.
2. *Wavefront.* Each sub-round takes, from every island edge, the next
   unaligned frame outward (``frames_per_front`` of them), starting from the
   edge frame's rounded motion with a search radius of ``clip_shift``. All
   are matched to the current template in one batch; their registered
   images are then added to the template.
3. *Boundary retries.* A frame whose correlation peak lies on the edge of
   its search range is matched again in the next sub-round, from that peak
   with a doubled radius (up to ``maxshift``).
4. *New seeds.* When fewer than ``min_fronts`` fronts remain, or a gap
   between islands is more than twice the median gap, frames at the middle
   of the longest gaps are matched over the whole ``maxshift`` range (FFT).
   They become islands only if their peak correlation and peak margin are
   at least the ``seed_accept_quantile`` of the aligned frames', the peak is
   not on a search edge, and their motion is plausible given the nearest
   aligned frames; otherwise growth reaches them later.
5. *Junctions.* Where two islands meet, a jump in motion larger than
   ``junction_jump_px`` (default: ``max(1, 99.9th percentile of per-frame
   steps)``) means one island may have locked onto a wrong peak: the island
   with the lower median correlation is realigned from the other, and the
   solution with the higher mean correlation is kept.
6. *Polish* (optional): every frame is matched again, within
   ``polish_radius`` of its motion, to the final template without its own
   contribution.

Outputs are then rendered in frame order, exactly as in sequential mode.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from .interpolation import interp_frame
from .matching import match_maps, peak_index, peak_margin, wide_correlation
from .outputs import matlab_round
from .template import TemplateState

log = logging.getLogger(__name__)

GROWN, CONSENSUS, SEED, REGROWN, AT_LIMIT = 0, 1, 2, 3, 4
"""Values of :attr:`GrowthQC.kind`."""


class FrameCache:
    """Every DS frame of a trial at the window's pixels, formed once.

    Parameters
    ----------
    source : PlainFrames or CompensatedFrames
        Forms the frames.
    ds : ndarray of int
        Frame times.
    chunk : int
        Frames formed at a time.
    max_device_bytes : float
        Larger caches are kept in CPU memory and moved per batch.
    """

    def __init__(
        self, source, ds, chunk: int = 64, max_device_bytes: float = 3e9
    ):
        """Form and store all frames."""
        self.window = source.window
        self.device = source.former.device
        n = ds.size
        n_pix = self.window.rows.numel()
        n_ch = source.former.layout.num_channels
        size = (n_ch + 1) * n * n_pix * 4
        store = self.device if size <= max_device_bytes else "cpu"
        self.values = torch.empty(n_ch, n, n_pix, device=store)
        self.freshness = torch.empty(n, n_pix, device=store)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            v, f = source.frames(ds[a:b])
            self.values[:, a:b] = v.to(store)
            self.freshness[a:b] = f.to(store)

    def __len__(self) -> int:
        """Number of frames."""
        return self.freshness.shape[0]

    def batch(self, idx):
        """``(channels, B, pixels)`` values and ``(B, pixels)`` freshness."""
        sel = torch.as_tensor(np.asarray(idx), device=self.values.device)
        return (
            self.values[:, sel].to(self.device),
            self.freshness[sel].to(self.device),
        )

    def moving(self, idx):
        """Matching image ``sqrt(sum of channels)`` and freshness."""
        vals, fresh = self.batch(idx)
        return torch.sqrt(vals.sum(0)), fresh

    def registered(self, k: int, motion, margin: int):
        """Registered channels and variance factor of frame ``k``."""
        vals, fresh = self.batch([k])
        fimg = self.window.image(fresh[0], 0.0)
        out = []
        for ch in range(vals.shape[0]):
            img = self.window.image(vals[ch, 0], float("nan"))
            out.append(interp_frame(img, fimg, motion[0], motion[1], margin))
        return out

    def registered_sum(self, ks, motions, margin: int) -> torch.Tensor:
        """``(B, H + 2m, W + 2m)`` registered channel sums of frames."""
        imgs = []
        for k, mot in zip(ks, motions):
            regs = self.registered(int(k), mot, margin)
            total = regs[0][0]
            for reg, _ in regs[1:]:
                total = total + reg
            imgs.append(total)
        h, w = self.window.shape
        if not imgs:
            return torch.empty(
                0, h + 2 * margin, w + 2 * margin, device=self.device
            )
        return torch.stack(imgs)


@dataclass
class GrowthQC:
    """Per-frame record of how each frame was registered.

    Attributes
    ----------
    peak, margin : ndarray
        Correlation at the peak, and its excess over the best correlation
        more than 2 shifts away.
    round : ndarray of int
        Sub-round in which the final motion was found (0: seeds).
    anchor : ndarray of int
        Frame whose motion started the search (-1 for seeds).
    retries : ndarray of int
        Boundary retries.
    kind : ndarray of int
        ``GROWN``, ``CONSENSUS``, ``SEED``, ``REGROWN`` or ``AT_LIMIT``.
    junctions : list
        ``(frame, neighbour, jump, outcome)`` for each conflicting junction.
    seeds_tried, seeds_accepted : int
        Wide-search seed candidates and how many became islands.
    """

    peak: np.ndarray
    margin: np.ndarray
    round: np.ndarray
    anchor: np.ndarray
    retries: np.ndarray
    kind: np.ndarray
    junctions: List[tuple] = field(default_factory=list)
    seeds_tried: int = 0
    seeds_accepted: int = 0

    def fields(self) -> dict:
        """The record as alignment-file fields (MATLAB-style names)."""
        return {
            "peakCorr": self.peak,
            "peakMargin": self.margin,
            "round": self.round.astype(np.float64),
            "anchor": self.anchor.astype(np.float64) + 1,
            "retries": self.retries.astype(np.float64),
            "kind": self.kind.astype(np.float64),
            "seedsTried": float(self.seeds_tried),
            "seedsAccepted": float(self.seeds_accepted),
            "junctionConflicts": float(len(self.junctions)),
        }


class Grower:
    """The growth scheduler for one trial.

    Parameters
    ----------
    cache : FrameCache
        The trial's frames.
    state : TemplateState
        Initial template (its sums include the consensus frames).
    params : MultiRoiRegistrationParams
        Settings (``clip_shift``, ``maxshift``, growth options).
    seeds : list of (int, int, int)
        Consensus seeds: frame index and starting [row, col] motion.
    candidates : sequence of int
        Further seed candidates (frames nearest the other initial frames).
    dtype : torch.dtype
        Precision of the correlation sums.
    independent : ndarray of bool, optional
        Frames that share no samples with the initial template's frames.
        Only these calibrate seed acceptance: frames overlapping the
        template's own samples correlate with it more than others do.
    """

    def __init__(
        self,
        cache: FrameCache,
        state: TemplateState,
        params,
        seeds,
        candidates,
        dtype=torch.float32,
        independent=None,
    ):
        """Set up per-frame state."""
        self.cache = cache
        self.state = state
        self.p = params
        self.dtype = dtype
        self.m = int(params.maxshift)
        n = len(cache)
        self.n = n
        self.motion = np.full((n, 2), np.nan)
        self.aligned = np.zeros(n, dtype=bool)
        self.root = np.full(n, -1)
        self.step = np.full(n, np.nan)
        self.qc = GrowthQC(
            peak=np.full(n, np.nan),
            margin=np.full(n, np.nan),
            round=np.zeros(n, dtype=np.int64),
            anchor=np.full(n, -1),
            retries=np.zeros(n, dtype=np.int64),
            kind=np.zeros(n, dtype=np.int64),
        )
        self.seeds = [(int(k), int(r), int(c)) for k, r, c in seeds]
        self.candidates = [int(k) for k in candidates]
        self.independent = (
            np.ones(n, dtype=bool) if independent is None else independent
        )
        self.tried = set()
        self.retry = {}  # frame -> (init_r, init_c, search, anchor)
        self.sub_round = 0

    # ----------------------------------------------------------- matching
    def _match(self, items):
        """Match ``(frame, init_r, init_c, search)`` items in init groups.

        Returns
        -------
        motion : ndarray
            ``(len(items), 2)`` motion.
        peak, margin : ndarray
            Peak correlation and margin.
        edge : ndarray of bool
            Whether the integer peak lies on the search range's edge.
        """
        n = len(items)
        motion = np.full((n, 2), np.nan)
        peak = np.full(n, np.nan)
        margin = np.full(n, np.nan)
        edge = np.zeros(n, dtype=bool)
        if n == 0:
            return motion, peak, margin, edge
        h, w = self.cache.window.shape
        m = self.m
        tmpl = self.state.matching_template()
        groups = {}
        for i, (_, r, c, s) in enumerate(items):
            groups.setdefault((r, c, s), []).append(i)
        win = self.cache.window
        for (r, c, s), idx in groups.items():
            ks = [items[i][0] for i in idx]
            mov, fresh = self.cache.moving(ks)
            crop = tmpl[m - r : m - r + h, m - c : m - c + w]
            mot, pk, corr = match_maps(
                mov, fresh, crop, win.rows, win.cols, s, self.dtype
            )
            rr, cc, _ = peak_index(corr)
            on_edge = (rr == 0) | (rr == 2 * s) | (cc == 0) | (cc == 2 * s)
            motion[idx] = mot.cpu().numpy() + np.array([r, c])
            peak[idx] = pk.cpu().numpy()
            margin[idx] = peak_margin(corr).cpu().numpy()
            edge[idx] = on_edge.cpu().numpy()
        return motion, peak, margin, edge

    def _init_of(self, motion):
        """Rounded, clamped starting shift from a motion estimate."""
        r = int(np.clip(matlab_round(motion[0]), -self.m, self.m))
        c = int(np.clip(matlab_round(motion[1]), -self.m, self.m))
        return r, c

    # ---------------------------------------------------------- accepting
    def _accept(self, ks, motions, peaks, margins, anchors, kinds, roots):
        """Record aligned frames and add them to the template."""
        ks = np.asarray(ks, dtype=np.int64)
        if ks.size == 0:
            return
        self.motion[ks] = motions
        self.aligned[ks] = True
        self.qc.peak[ks] = peaks
        self.qc.margin[ks] = margins
        self.qc.anchor[ks] = anchors
        self.qc.kind[ks] = kinds
        self.qc.round[ks] = self.sub_round
        self.root[ks] = roots
        for i, (k, a) in enumerate(zip(ks, anchors)):
            if a >= 0:
                self.step[k] = np.hypot(*(motions[i] - self.motion[a])) / abs(
                    k - a
                )
        imgs = self.cache.registered_sum(ks, motions, self.m)
        self.state.update(imgs)

    def _withdraw(self, ks):
        """Remove aligned frames from the template and mark them unaligned."""
        ks = np.asarray(ks, dtype=np.int64)
        imgs = self.cache.registered_sum(ks, self.motion[ks], self.m)
        self.state.update(imgs, sign=-1)
        self.aligned[ks] = False

    # -------------------------------------------------------------- seeds
    def _seed_round(self):
        """Match the consensus seeds (sub-round 0)."""
        items = [(k, r, c, self.p.clip_shift) for k, r, c in self.seeds]
        self._run_items(
            items,
            anchors=[-1] * len(items),
            kind=CONSENSUS,
            roots=[k for k, _, _ in self.seeds],
        )

    def _step_limit(self) -> float:
        """Plausible motion change per frame (99.9th percentile of steps)."""
        steps = self.step[np.isfinite(self.step)]
        if steps.size < 50:
            return 1.0
        return max(0.25, float(np.quantile(steps, 0.999)))

    def _plausible(self, k: int, motion) -> bool:
        """Whether a seed's motion fits the nearest aligned frames."""
        v = self._step_limit()
        al = np.flatnonzero(self.aligned)
        for side in (al[al < k][-1:], al[al > k][:1]):
            for a in side:
                if np.hypot(*(motion - self.motion[a])) > v * abs(k - a) + 1:
                    return False
        return True

    def try_seeds(self, ks):
        """Wide-search seed candidates; accept those passing the checks."""
        ks = [
            int(k) for k in ks if not self.aligned[k] and k not in self.tried
        ]
        if not ks:
            return
        self.tried.update(ks)
        self.qc.seeds_tried += len(ks)
        vals, fresh = self.cache.moving(ks)
        win = self.cache.window
        corr = wide_correlation(
            win.image(vals, float("nan")),
            win.image(fresh, 0.0),
            self.state.matching_template(),
        )
        rr, cc, _ = peak_index(corr)
        items = [
            (k, self.m - int(r), self.m - int(c), self.p.clip_shift)
            for k, r, c in zip(ks, rr.cpu().numpy(), cc.cpu().numpy())
        ]
        motion, peak, margin, edge = self._match(items)
        good = self._calibration_set()
        q = self.p.seed_accept_quantile
        r_min = np.quantile(self.qc.peak[good], q) if good.any() else -1
        mg = self.qc.margin[good]
        mg = mg[np.isfinite(mg)]
        m_min = np.quantile(mg, q) if mg.size else -np.inf
        ok = [
            i
            for i, (k, *_rest) in enumerate(items)
            if not edge[i]
            and np.isfinite(peak[i])
            and np.isfinite(motion[i]).all()
            and peak[i] >= r_min
            and (not np.isfinite(margin[i]) or margin[i] >= m_min)
            and self._plausible(k, motion[i])
        ]
        if ok:
            acc = [items[i][0] for i in ok]
            self._accept(
                acc,
                motion[ok],
                peak[ok],
                margin[ok],
                [-1] * len(ok),
                [SEED] * len(ok),
                acc,
            )
            self.qc.seeds_accepted += len(ok)

    def _calibration_set(self) -> np.ndarray:
        """Grown frames independent of the initial template's samples."""
        return (
            self.aligned
            & self.independent
            & np.isfinite(self.qc.peak)
            & np.isin(self.qc.kind, (GROWN, REGROWN))
        )

    def _calibrated(self) -> bool:
        """Whether enough frames were grown to judge seed candidates."""
        return int(self._calibration_set().sum()) >= 100

    def _gap_candidates(self, fronts: int):
        """Middle frames of the gaps that most need a seed."""
        gaps = _runs(~self.aligned)
        if not gaps:
            return []
        lengths = np.array([b - a for a, b in gaps])
        need = fronts < self.p.min_fronts
        median = np.median(lengths)
        out = []
        for i in np.argsort(-lengths, kind="stable"):
            a, b = gaps[i]
            if lengths[i] < 16:
                break
            if not need and lengths[i] <= 2 * median:
                break
            mid = (a + b - 1) // 2
            if mid not in self.tried:
                out.append(mid)
            if len(out) >= 8:
                break
        return out

    # ------------------------------------------------------------- growth
    def _fronts(self):
        """``(frame, anchor)`` for the next frames outward of every island."""
        k_front = int(self.p.frames_per_front)
        targets = {}
        for a, b in _runs(self.aligned):
            for edge, step in ((a, -1), (b - 1, 1)):
                for d in range(1, k_front + 1):
                    f = edge + step * d
                    if not 0 <= f < self.n or self.aligned[f]:
                        break
                    if f in self.retry:
                        break
                    prev = targets.get(f)
                    if prev is None or self.qc.peak[edge] > self.qc.peak[prev]:
                        targets[f] = edge
        return sorted(targets.items())

    def _run_items(self, items, anchors, kind, roots):
        """Match items; accept, or queue boundary retries."""
        motion, peak, margin, edge = self._match(items)
        acc = []
        for i, (k, r, c, s) in enumerate(items):
            if edge[i] and s < self.m and np.isfinite(peak[i]):
                nr, nc = self._init_of(motion[i])
                self.retry[k] = (
                    nr,
                    nc,
                    min(self.m, 2 * s),
                    anchors[i],
                    roots[i],
                    kind,
                )
                self.qc.retries[k] += 1
                continue
            if not np.isfinite(motion[i]).all():
                motion[i] = self.motion[anchors[i]] if anchors[i] >= 0 else 0
            acc.append(i)
        kinds = [AT_LIMIT if edge[i] else kind for i in acc]
        self._accept(
            [items[i][0] for i in acc],
            motion[acc],
            peak[acc],
            margin[acc],
            [anchors[i] for i in acc],
            kinds,
            [roots[i] for i in acc],
        )
        return [items[i][0] for i in acc]

    def _sub_round(self):
        """One wavefront step: retries plus one frame per front."""
        self.sub_round += 1
        fronts = self._fronts()
        cand = []
        if self._calibrated():
            cand = [k for k in self.candidates if k not in self.tried]
            cand += self._gap_candidates(len(fronts))
        if cand:
            self.try_seeds(cand)
            fronts = self._fronts()
        items, anchors, roots, kinds = [], [], [], []
        for k, (r, c, s, a, root, kind) in sorted(self.retry.items()):
            if self.aligned[k]:
                continue
            items.append((k, r, c, s))
            anchors.append(a)
            roots.append(root)
            kinds.append(kind)
        self.retry = {}
        for k, a in fronts:
            r, c = self._init_of(self.motion[a])
            items.append((k, r, c, self.p.clip_shift))
            anchors.append(a)
            roots.append(self.root[a])
            kinds.append(GROWN)
        if not items:
            rest = np.flatnonzero(~self.aligned)
            if rest.size == 0:  # the seeds above aligned the last frames
                return
            self.try_seeds(rest[:1])
            if not self.aligned[rest[0]]:
                items = [(int(rest[0]), 0, 0, self.p.clip_shift)]
                anchors, roots, kinds = [-1], [int(rest[0])], [SEED]
        done = []
        for kind in sorted(set(kinds)):
            sel = [i for i, kd in enumerate(kinds) if kd == kind]
            done += self._run_items(
                [items[i] for i in sel],
                [anchors[i] for i in sel],
                kind,
                [roots[i] for i in sel],
            )
        self._check_junctions(done)

    # ---------------------------------------------------------- junctions
    def _jump_limit(self) -> float:
        """Largest plausible motion jump between adjacent frames."""
        if self.p.junction_jump_px is not None:
            return float(self.p.junction_jump_px)
        return max(1.0, self._step_limit())

    def _check_junctions(self, ks):
        """Resolve junctions formed by newly aligned frames."""
        limit = self._jump_limit()
        for k in ks:
            a = self.qc.anchor[k]
            for g in (k - 1, k + 1):
                if g == a or not 0 <= g < self.n or not self.aligned[g]:
                    continue
                if self.qc.anchor[g] == k or self.root[g] == self.root[k]:
                    continue
                jump = float(np.hypot(*(self.motion[k] - self.motion[g])))
                if jump > limit:
                    self._resolve(k, g, jump)

    def _chain(self, order, start):
        """Realign frames one by one, each from the previous one's motion."""
        motion = np.zeros((len(order), 2))
        peak = np.zeros(len(order))
        margin = np.zeros(len(order))
        prev = start
        for i, f in enumerate(order):
            r, c = self._init_of(prev)
            s = self.p.clip_shift
            while True:
                mot, pk, mg, edge = self._match([(int(f), r, c, s)])
                if not edge[0] or s >= self.m or not np.isfinite(mot[0]).all():
                    break
                r, c = self._init_of(mot[0])
                s = min(self.m, 2 * s)
            if not np.isfinite(mot[0]).all():
                mot[0] = prev
            motion[i], peak[i], margin[i] = mot[0], pk[0], mg[0]
            prev = mot[0]
        return motion, peak, margin

    def _resolve(self, k: int, g: int, jump: float):
        """Realign the weaker of the islands meeting at ``(k, g)``.

        The weaker island (lower median peak correlation) is withdrawn from
        the template and realigned frame by frame from the stronger side.
        Its old motion is scored against the same template; the solution
        with the higher mean peak correlation is kept.
        """
        pk = np.nanmedian(self.qc.peak[self.root == self.root[k]])
        pg = np.nanmedian(self.qc.peak[self.root == self.root[g]])
        strong, weak = (k, g) if pk >= pg else (g, k)
        members = np.flatnonzero(self.root == self.root[weak])
        order = members[np.argsort((1 if weak > strong else -1) * members)]
        saved = {
            "motion": self.motion[order].copy(),
            "anchor": self.qc.anchor[order].copy(),
            "kind": self.qc.kind[order].copy(),
            "round": self.qc.round[order].copy(),
            "root": self.root[order].copy(),
            "step": self.step[order].copy(),
        }
        self._withdraw(order)
        new = self._chain(order, self.motion[strong])
        old = self._match(
            [
                (int(f), *self._init_of(mo), 1)
                for f, mo in zip(order, saved["motion"])
            ]
        )
        better = np.nanmean(new[1]) > np.nanmean(old[1])
        self.qc.junctions.append(
            (int(k), int(g), jump, "regrown" if better else "kept")
        )
        if better:
            anchors = np.concatenate([[strong], order[:-1]])
            self._accept(
                order,
                new[0],
                new[1],
                new[2],
                anchors,
                [REGROWN] * order.size,
                [self.root[strong]] * order.size,
            )
            return
        self._accept(
            order,
            saved["motion"],
            old[1],
            old[2],
            saved["anchor"],
            saved["kind"],
            saved["root"],
        )
        self.qc.round[order] = saved["round"]
        self.step[order] = saved["step"]

    # ---------------------------------------------------------------- run
    def run(self, max_rounds: Optional[int] = None):
        """Align every frame.

        Returns
        -------
        motion : ndarray
            ``(frames, 2)`` [row, col] motion.
        qc : GrowthQC
            How each frame was registered.
        """
        self._seed_round()
        limit = max_rounds or 4 * self.n + 10
        while not self.aligned.all():
            if self.sub_round >= limit:
                raise RuntimeError("growth did not converge")
            self._sub_round()
        log.info(
            "grown in %d sub-rounds; %d of %d seed candidates accepted; "
            "%d junction conflicts",
            self.sub_round,
            self.qc.seeds_accepted,
            self.qc.seeds_tried,
            len(self.qc.junctions),
        )
        return self.motion, self.qc

    def polish(self, radius: int):
        """Rematch every frame near its motion to the template without it."""
        st = self.state
        h, w = self.cache.window.shape
        m = self.m
        win = self.cache.window
        for k in range(self.n):
            img = self.cache.registered_sum([k], self.motion[[k]], m)[0]
            sel = ~torch.isnan(img)
            total = st.total - torch.where(sel, img, 0.0).to(st.total.dtype)
            count = st.count - sel.to(st.count.dtype)
            loo = torch.sqrt(total / count).masked_fill(
                count < st.min_samples, float("nan")
            )
            both = (st.initial + loo) / 2
            tmpl = torch.where(torch.isnan(st.initial), loo, both)
            tmpl = torch.where(torch.isnan(loo), st.initial, tmpl)
            r, c = self._init_of(self.motion[k])
            mov, fresh = self.cache.moving([k])
            mot, pk, corr = match_maps(
                mov,
                fresh,
                tmpl[m - r : m - r + h, m - c : m - c + w],
                win.rows,
                win.cols,
                radius,
                self.dtype,
            )
            if torch.isfinite(mot).all():
                self.motion[k] = mot[0].cpu().numpy() + np.array([r, c])
                self.qc.peak[k] = float(pk[0])
                self.qc.margin[k] = float(peak_margin(corr)[0])
        return self.motion


def _runs(mask: np.ndarray):
    """``(start, stop)`` of the runs of True in a boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def seeds_from_consensus(cons, init, ds, reach=None):
    """Consensus seeds and other candidates as DS frame indices.

    Parameters
    ----------
    cons : ConsensusResult
        Pairwise matching of the initial frames.
    init, ds : ndarray of int
        Initial and DS frame times.
    reach : int, optional
        Lines within which a DS frame shares samples with an initial frame.

    Returns
    -------
    seeds : list of (int, int, int)
        DS frame nearest each consensus member, with the member's motion
        relative to the reference (``-motion[:, i, ref]``), rounded.
    candidates : list of int
        DS frames nearest the other initial frames.
    independent : ndarray of bool
        DS frames more than ``reach`` lines (two kernel half-widths) from
        every consensus member, so sharing none of its samples (all frames
        when ``reach`` is None).
    """
    near = np.clip(np.searchsorted(ds, init), 0, ds.size - 1)
    left = np.clip(near - 1, 0, ds.size - 1)
    near = np.where(
        np.abs(ds[left] - init) <= np.abs(ds[near] - init), left, near
    )
    ref = cons.reference
    seeds, taken = [], set()
    for i in cons.members:
        k = int(near[i])
        if k in taken:
            continue
        taken.add(k)
        mot = -cons.motion[:, i, ref]
        seeds.append((k, int(matlab_round(mot[0])), int(matlab_round(mot[1]))))
    members = set(int(i) for i in cons.members)
    candidates = [
        int(near[i])
        for i in range(init.size)
        if i not in members and int(near[i]) not in taken
    ]
    independent = np.ones(ds.size, dtype=bool)
    if reach is not None:
        t0 = init[cons.members]
        independent = np.abs(ds[:, None] - t0[None, :]).min(1) > reach
    return seeds, candidates, independent
