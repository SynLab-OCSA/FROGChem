"""
Quantum Spin Simulation and 2D NMR Module for paclab-lib.

Provides:
1. Exact isotropic spin-1/2 quantum Hamiltonian construction and M-block diagonalization
   for arbitrary coupled spin systems (accounting for scalar J-couplings, second-order
   strong coupling, and the roof effect).
2. Graph-based cluster partitioning to decompose large molecules into uncoupled
   spin clusters, preventing exponential Hilbert space scaling.
3. 1D NMR spectrum simulation with Lorentzian line broadening.
4. 2D COSY spectrum simulation (scalar through-bond correlation via J).
5. 2D NOESY spectrum simulation (spatial through-space cross-relaxation via r^-6).
6. Export and plotting utilities for 1D and 2D spectra.

Andrea Pellegrini, 2026
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from frogchem.nmr import NUCLEUS_DATA, get_nucleus_spin, normalize_nucleus_name


@dataclass
class SpinTransition:
    """Represents an allowed NMR dipole transition."""
    freq_hz: float
    intensity: float
    cluster_id: int = 0
    delta_ppm: float = 0.0


def solve_spin_cluster(
    nu: np.ndarray,
    J: np.ndarray,
    spins: Optional[Union[float, Sequence[float]]] = None,
    min_intensity: float = 1e-5,
) -> List[Tuple[float, float]]:
    """Solve the isotropic spin Hamiltonian for a cluster of n coupled spins with arbitrary
    spin quantum numbers S_k (e.g. S=0.5 for 1H/13C, S=1 for 2H/14N, S=1.5 for 33S/11B, S=2.5 for 17O)
    using block diagonalization by total magnetic quantum number M = sum(m_k).

    Parameters:
        nu: (n,) array of chemical shift offsets / resonance frequencies in Hz.
        J: (n, n) symmetric scalar coupling matrix in Hz.
        spins: Single spin quantum number (float, e.g. 2.5) or sequence of spin quantum numbers
               for each nucleus in the cluster. If None, defaults to 0.5 (spin-1/2).
        min_intensity: Minimum transition probability to keep.

    Returns:
        List of (frequency_hz, transition_probability) tuples, one per Delta M = +1
        transition (frequencies keep their sign: negative for delta < 0 ppm).
    """
    n = len(nu)
    if n == 0:
        return []

    if spins is None:
        spin_list = [0.5] * n
    elif isinstance(spins, (int, float)):
        spin_list = [float(spins)] * n
    else:
        spin_list = [float(s) for s in spins]

    if len(spin_list) != n:
        raise ValueError(f"Length of spins ({len(spin_list)}) must match number of frequencies ({n})")

    if n == 1:
        return [(float(nu[0]), 1.0)]

    # Product Zeeman basis states: for each spin k, m_k in [-S_k, -S_k+1, ..., +S_k]
    m_ranges = []
    for s in spin_list:
        d = int(round(2.0 * s + 1.0))
        m_ranges.append([-s + i for i in range(d)])

    all_states = [np.array(st, dtype=float) for st in product(*m_ranges)]

    # 1. Group Zeeman product basis states by total magnetic quantum number M = sum(m_k)
    blocks: Dict[float, List[int]] = defaultdict(list)
    for idx, st in enumerate(all_states):
        M = round(float(np.sum(st)), 4)
        blocks[M].append(idx)

    # 2. Construct and diagonalize each M block independently
    eigvals: Dict[float, np.ndarray] = {}
    eigvecs: Dict[float, np.ndarray] = {}

    for M, b_list in blocks.items():
        dim = len(b_list)
        H_block = np.zeros((dim, dim), dtype=float)

        for i, bi in enumerate(b_list):
            st_i = all_states[bi]
            # Diagonal element: sum nu_k m_k + sum_{k<l} J_kl m_k m_l
            diag = float(np.sum(nu * st_i))
            for k in range(n):
                for l in range(k + 1, n):
                    diag += J[k, l] * st_i[k] * st_i[l]
            H_block[i, i] = diag

            # Off-diagonal element: 1/2 J_kl (I+_k I-_l + I-_k I+_l)
            for j in range(i + 1, dim):
                bj = b_list[j]
                st_j = all_states[bj]
                diff = st_j - st_i
                non_zero = np.where(np.abs(diff) > 1e-4)[0]
                if len(non_zero) == 2:
                    k, l = non_zero[0], non_zero[1]
                    dk, dl = diff[k], diff[l]
                    if np.isclose(dk, 1.0) and np.isclose(dl, -1.0):
                        # state j has m_k = m_i + 1 and m_l = m_i - 1
                        val_k = math.sqrt(spin_list[k] * (spin_list[k] + 1) - st_i[k] * (st_i[k] + 1))
                        val_l = math.sqrt(spin_list[l] * (spin_list[l] + 1) - st_i[l] * (st_i[l] - 1))
                        val = 0.5 * J[k, l] * val_k * val_l
                        H_block[i, j] = val
                        H_block[j, i] = val
                    elif np.isclose(dk, -1.0) and np.isclose(dl, 1.0):
                        # state j has m_k = m_i - 1 and m_l = m_i + 1
                        val_k = math.sqrt(spin_list[k] * (spin_list[k] + 1) - st_i[k] * (st_i[k] - 1))
                        val_l = math.sqrt(spin_list[l] * (spin_list[l] + 1) - st_i[l] * (st_i[l] + 1))
                        val = 0.5 * J[k, l] * val_k * val_l
                        H_block[i, j] = val
                        H_block[j, i] = val

        w, v = np.linalg.eigh(H_block)
        eigvals[M] = w
        eigvecs[M] = v

    # 3. Calculate dipole transitions between adjacent blocks: Delta M = +1
    transitions: List[Tuple[float, float]] = []
    sorted_M = sorted(blocks.keys())

    for idx_M, M in enumerate(sorted_M[:-1]):
        M_next = sorted_M[idx_M + 1]
        if not np.isclose(M_next - M, 1.0):
            continue

        b_M = blocks[M]
        b_M1 = blocks[M_next]
        dim_M = len(b_M)
        dim_M1 = len(b_M1)

        # Build raising operator I+ matrix: connects |st_j> in M to |st_i> in M+1 if exactly one spin k has m_i = m_j + 1
        I_plus = np.zeros((dim_M1, dim_M), dtype=float)
        for j, bj in enumerate(b_M):
            st_j = all_states[bj]
            for i, bi in enumerate(b_M1):
                st_i = all_states[bi]
                diff = st_i - st_j
                non_zero = np.where(np.abs(diff) > 1e-4)[0]
                if len(non_zero) == 1:
                    k = non_zero[0]
                    if np.isclose(diff[k], 1.0):
                        I_plus[i, j] = math.sqrt(spin_list[k] * (spin_list[k] + 1) - st_j[k] * (st_j[k] + 1))

        # Transform to eigenbasis: <psi_b^(M+1) | I+ | psi_a^(M)>
        M_trans = eigvecs[M_next].T @ (I_plus @ eigvecs[M])
        w_M = eigvals[M]
        w_M1 = eigvals[M_next]

        for i in range(dim_M1):
            for j in range(dim_M):
                freq = float(w_M1[i] - w_M[j])
                prob = float(M_trans[i, j] ** 2)
                # Only Delta M = +1 is enumerated, so no line is double counted: keep
                # negative frequencies too (nuclei with delta < 0 ppm).
                if prob >= min_intensity:
                    transitions.append((freq, prob))

    return transitions


def partition_spin_network(
    n_spins: int,
    J: np.ndarray,
    cutoff: float = 0.3,
    max_cluster_size: int = 8,
    spins: Optional[Sequence[float]] = None,
    max_dim: int = 256,
    counts: Optional[Sequence[int]] = None,
    fallback: str = 'singles',
) -> List[List[int]]:
    """Partition spin network into connected components (clusters) based on J threshold.
    If a cluster exceeds max_cluster_size or total Hilbert space dimension exceeds max_dim,
    its internal threshold is iteratively increased to prune weaker couplings, preventing
    combinatorial Hilbert space explosion (crucial for high-spin nuclei like 17O where (2S+1)=6).

    Parameters:
        counts: Optional number of equivalent spins represented by each node (group). The
            cluster size is then sum(counts) and its dimension prod((2S+1)**count).
        fallback: What to do with a cluster that cannot be subdivided into valid clusters:
            'singles' (default) splits it into isolated nodes (couplings lost);
            'none' returns it unchanged, so the caller can treat it differently
            (e.g. first order, as simulate_quantum_transitions does).
    """
    spin_list = [0.5] * n_spins if spins is None else [float(s) for s in spins]
    count_list = [1] * n_spins if counts is None else [int(c) for c in counts]

    def cluster_dim(c_indices: List[int]) -> int:
        d = 1
        for idx in c_indices:
            d *= int(round(2.0 * spin_list[idx] + 1.0)) ** count_list[idx]
        return d

    def is_valid_cluster(c_indices: List[int]) -> bool:
        size = sum(count_list[idx] for idx in c_indices)
        return size <= max_cluster_size and cluster_dim(c_indices) <= max_dim

    adj = np.abs(J) >= cutoff
    visited = [False] * n_spins
    raw_clusters: List[List[int]] = []

    for i in range(n_spins):
        if not visited[i]:
            cluster: List[int] = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                cluster.append(curr)
                for neighbor in range(n_spins):
                    if adj[curr, neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            raw_clusters.append(sorted(cluster))

    final_clusters: List[List[int]] = []
    for c in raw_clusters:
        if is_valid_cluster(c):
            final_clusters.append(c)
        else:
            # Subdivide large cluster by progressively raising threshold
            c_J = np.zeros((len(c), len(c)), dtype=float)
            for i, a in enumerate(c):
                for j, b in enumerate(c):
                    c_J[i, j] = J[a, b]
            c_spins = [spin_list[idx] for idx in c]
            c_counts = [count_list[idx] for idx in c]

            upper_vals = sorted(list(set(np.abs(c_J[np.triu_indices(len(c), k=1)]))))
            upper_vals = [v for v in upper_vals if v >= cutoff]

            sub_assigned = False
            for thresh in upper_vals:
                sub_c = partition_spin_network(
                    len(c),
                    c_J,
                    cutoff=thresh + 1e-4,
                    max_cluster_size=max_cluster_size,
                    spins=c_spins,
                    max_dim=max_dim,
                    counts=c_counts,
                    fallback='none',
                )
                if all(is_valid_cluster([c[idx] for idx in sc]) for sc in sub_c):
                    for sc in sub_c:
                        final_clusters.append(sorted([c[idx] for idx in sc]))
                    sub_assigned = True
                    break
            if not sub_assigned:
                if fallback == 'none':
                    final_clusters.append(c)
                else:
                    # Fallback: single spins
                    for idx in c:
                        final_clusters.append([idx])

    return final_clusters


def _m_distribution(spin: float, n: int) -> Dict[float, float]:
    """Degeneracy of the total M of n uncoupled spins I (n spin-1/2 -> binomial / Pascal row)."""
    single = [-spin + i for i in range(int(round(2.0 * spin + 1.0)))]
    dist: Dict[float, float] = {0.0: 1.0}
    for _ in range(n):
        new: Dict[float, float] = defaultdict(float)
        for m_tot, w in dist.items():
            for m in single:
                new[round(m_tot + m, 6)] += w
        dist = dict(new)
    return dist


def _first_order_multiplets(
    cluster: Sequence[int],
    nu_hz: np.ndarray,
    J_arr: np.ndarray,
    counts: Sequence[int],
    spins: Sequence[float],
    j_cutoff: float,
    min_intensity: float,
) -> List[Tuple[float, float]]:
    """First-order (weak coupling, |J| << |dnu|) multiplets of every group in a cluster.

    Group k (n_k nuclei) gives lines at nu_k + sum_j J_kj * M_j, where M_j runs over the
    total magnetic quantum number of the n_j spins of each coupled group j, weighted by its
    degeneracy (Pascal triangle for spin 1/2). The lines of group k sum to n_k.
    """
    out: List[Tuple[float, float]] = []
    for k in cluster:
        lines: Dict[float, float] = {float(nu_hz[k]): 1.0}
        for j in cluster:
            if j == k or abs(J_arr[k, j]) < j_cutoff:
                continue
            dist = _m_distribution(spins[j], counts[j])
            new: Dict[float, float] = defaultdict(float)
            for f, w in lines.items():
                for m_tot, deg in dist.items():
                    new[round(f + J_arr[k, j] * m_tot, 6)] += w * deg
            lines = dict(new)
        total = sum(lines.values())
        for f, w in lines.items():
            inten = counts[k] * w / total
            if inten >= min_intensity:
                out.append((f, inten))
    return out


def _add_external_couplings(
    lines: List[Tuple[float, float]],
    cluster: Sequence[int],
    nu_hz: np.ndarray,
    J_arr: np.ndarray,
    counts: Sequence[int],
    spins: Sequence[float],
    j_cutoff: float,
    min_intensity: float,
) -> List[Tuple[float, float]]:
    """Apply to first order the couplings (|J| >= j_cutoff) between a cluster and the groups
    outside it (couplings dropped by the partitioning). Each line is attributed to the cluster
    group with the closest resonance frequency and split by the total-M distribution of every
    external group coupled to it; intensities are conserved."""
    outside = [j for j in range(len(nu_hz)) if j not in set(cluster)]
    if not outside or not lines:
        return lines
    pattern: Dict[int, Dict[float, float]] = {}
    for k in cluster:
        offs: Dict[float, float] = {0.0: 1.0}
        for j in outside:
            if abs(J_arr[k, j]) < j_cutoff:
                continue
            dist = _m_distribution(spins[j], counts[j])
            tot = sum(dist.values())
            new: Dict[float, float] = defaultdict(float)
            for o, w in offs.items():
                for m_tot, deg in dist.items():
                    new[round(o + J_arr[k, j] * m_tot, 6)] += w * deg / tot
            offs = dict(new)
        pattern[k] = offs
    if all(len(p) == 1 for p in pattern.values()):
        return lines
    out: List[Tuple[float, float]] = []
    for f, inten in lines:
        k = min(cluster, key=lambda c: abs(nu_hz[c] - f))
        for o, w in pattern[k].items():
            if inten * w >= min_intensity:
                out.append((f + o, inten * w))
    return out


def _auto_range(values: Sequence[float], pad: float, clamp: Optional[float] = None) -> Tuple[float, float]:
    """(min - pad, max + pad); the lower bound is raised to `clamp` only if no value
    lies within half a pad of it (so lines below the usual window are never cut off)."""
    lo = float(np.min(values)) - pad
    hi = float(np.max(values)) + pad
    if clamp is not None and float(np.min(values)) - 0.5 * pad >= clamp:
        lo = max(clamp, lo)
    return lo, hi


def simulate_quantum_transitions(
    shifts_ppm: Sequence[float],
    j_matrix: np.ndarray,
    group_counts: Optional[Sequence[int]] = None,
    freq_mhz: float = 500.0,
    j_cutoff: float = 0.3,
    max_cluster_size: int = 8,
    min_intensity: float = 1e-4,
    spins: Optional[Union[float, Sequence[float]]] = None,
    nucleus: Optional[str] = None,
    max_hilbert_dim: int = 256,
) -> List[SpinTransition]:
    """Simulate exact quantum mechanical transitions for a set of spins with arbitrary nuclear spin I.

    Parameters:
        shifts_ppm: Sequence of chemical shifts in ppm.
        j_matrix: (N, N) coupling matrix in Hz.
        group_counts: Optional sequence of number of equivalent nuclei per spin/group.
        freq_mhz: Spectrometer frequency in MHz (default: 500.0).
        j_cutoff: Minimum J (Hz) to treat two spins as coupled in the Hamiltonian.
        max_cluster_size: Maximum spin cluster size to diagonalize simultaneously.
        min_intensity: Minimum transition probability to keep.
        spins: Single spin quantum number (float) or sequence of spin quantum numbers.
        nucleus: Optional nucleus symbol (e.g. '17O', '1H', '33S') used to look up spin if spins is None.
        max_hilbert_dim: Maximum Hilbert space dimension per cluster (default: 256).

    Equivalent groups (group_counts[k] = n > 1, e.g. CH3) are expanded into n individual
    spins with the group shift and the group-group couplings (intra-group J is omitted: it
    does not affect the spectrum of magnetically equivalent spins). Clusters are partitioned
    at group level, so a group is never split. A cluster whose expanded size exceeds
    max_cluster_size or whose dimension prod((2I+1)**n) exceeds max_hilbert_dim, and that
    cannot be subdivided by raising the J threshold, is treated to FIRST ORDER: each group
    gives a multiplet nu_k + sum_j J_kj * M_j over the binomial (Pascal) distribution of
    the total M_j of the n_j spins of every coupled group j. Couplings between different
    sub-clusters (removed when a large network is split by raising the J threshold) are
    added to first order in the same way, each line being attributed to the group of its
    cluster with the closest resonance frequency.

    Intensities: each cluster is normalised to its number of nuclei (sum of counts); for an
    exactly solved cluster of like spins this gives each nucleus the same integrated weight,
    so the integrals of the groups are in the ratio of their counts.

    Returns:
        List of SpinTransition objects.
    """
    n_spins = len(shifts_ppm)
    if n_spins == 0:
        return []

    # Determine spin array
    if spins is not None:
        if isinstance(spins, (int, float)):
            spin_arr = [float(spins)] * n_spins
        else:
            spin_arr = [float(s) for s in spins]
    elif nucleus is not None:
        s_val = get_nucleus_spin(nucleus)
        spin_arr = [s_val] * n_spins
    else:
        spin_arr = [0.5] * n_spins

    count_list = [max(1, int(round(c))) for c in (group_counts if group_counts is not None else [1] * n_spins)]
    counts = np.array(count_list, dtype=float)
    nu_hz = np.array(shifts_ppm, dtype=float) * freq_mhz
    J_arr = np.array(j_matrix, dtype=float)

    def _dim(idx_list: Sequence[int]) -> int:
        d = 1
        for idx in idx_list:
            d *= int(round(2.0 * spin_arr[idx] + 1.0)) ** count_list[idx]
        return d

    def _exact_ok(idx_list: Sequence[int]) -> bool:
        return sum(count_list[i] for i in idx_list) <= max_cluster_size and _dim(idx_list) <= max_hilbert_dim

    clusters = partition_spin_network(
        n_spins,
        J_arr,
        cutoff=j_cutoff,
        max_cluster_size=max_cluster_size,
        spins=spin_arr,
        max_dim=max_hilbert_dim,
        counts=count_list,
        fallback='none',
    )
    all_transitions: List[SpinTransition] = []

    for cluster_id, cluster in enumerate(clusters):
        if len(cluster) == 1:
            # Isolated group: a single line carrying all its nuclei
            k = cluster[0]
            lines = [(float(nu_hz[k]), float(counts[k]))]
        elif not _exact_ok(cluster):
            # First-order fallback (cluster too large for exact diagonalization)
            lines = _first_order_multiplets(cluster, nu_hz, J_arr, count_list, spin_arr,
                                            j_cutoff, min_intensity)
        else:
            # Expand each group into its individual spins
            members = [k for k in cluster for _ in range(count_list[k])]
            c_size = len(members)
            c_nu = nu_hz[members]
            c_spins = [spin_arr[k] for k in members]
            c_J = np.zeros((c_size, c_size), dtype=float)
            for i, a in enumerate(members):
                for j, b in enumerate(members):
                    if a != b:
                        c_J[i, j] = J_arr[a, b]

            raw_trans = solve_spin_cluster(c_nu, c_J, spins=c_spins, min_intensity=min_intensity)

            # Scale transitions so that the sum of intensities in the cluster equals total spins (atoms)
            total_cluster_spins = float(np.sum(counts[cluster]))
            cluster_prob_sum = float(sum(prob for _, prob in raw_trans))
            c_scale = (total_cluster_spins / cluster_prob_sum) if cluster_prob_sum > 0 else 1.0
            lines = [(freq, prob * c_scale) for freq, prob in raw_trans]

        # Couplings to groups outside this cluster (dropped by the partitioning): first order
        lines = _add_external_couplings(lines, cluster, nu_hz, J_arr, count_list, spin_arr,
                                        j_cutoff, min_intensity)

        # Merge coincident lines (degenerate transitions of expanded equivalent spins)
        merged: Dict[float, float] = defaultdict(float)
        for freq, inten in lines:
            merged[round(freq, 6)] += inten
        for freq, inten in merged.items():
            all_transitions.append(
                SpinTransition(
                    freq_hz=freq,
                    intensity=inten,
                    cluster_id=cluster_id,
                    delta_ppm=freq / freq_mhz,
                )
            )

    return all_transitions


def _lorentzian_profiles(axis: np.ndarray, centers: np.ndarray, fwhm: float) -> np.ndarray:
    """Compute Lorentzian line profiles for multiple center frequencies/shifts.

    Parameters:
        axis: (P,) grid axis.
        centers: (N,) center positions.
        fwhm: Full width at half maximum.

    Returns:
        (N, P) array where profiles[i] is the normalized Lorentzian for centers[i].
    """
    if len(centers) == 0:
        return np.zeros((0, len(axis)), dtype=float)
    gamma = fwhm / 2.0
    gamma2 = gamma * gamma
    diff = axis[np.newaxis, :] - centers[:, np.newaxis]
    return (gamma / math.pi) / (diff * diff + gamma2)


def simulate_1d_spectrum(
    transitions: Sequence[SpinTransition],
    ppm_range: Optional[Tuple[float, float]] = None,
    n_points: int = 8192,
    freq_mhz: float = 500.0,
    fwhm_hz: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convolve quantum dipole transitions with a Lorentzian lineshape to generate
    a continuous 1D NMR spectrum.

    Parameters:
        transitions: List of SpinTransition objects.
        ppm_range: (min_ppm, max_ppm) tuple. If None, auto-determined from transitions.
        n_points: Number of points on the spectrum grid.
        freq_mhz: Spectrometer frequency in MHz.
        fwhm_hz: Full width at half maximum of Lorentzian line in Hz.

    Returns:
        (ppm_axis, intensity_array)
    """
    if not transitions:
        if ppm_range is None:
            ppm_range = (-0.5, 12.0)
        axis = np.linspace(ppm_range[0], ppm_range[1], n_points)
        return axis, np.zeros(n_points)

    deltas = np.array([t.delta_ppm for t in transitions], dtype=float)
    intensities = np.array([t.intensity for t in transitions], dtype=float)

    if ppm_range is None:
        ppm_range = _auto_range(deltas, pad=1.0, clamp=-1.0)

    axis = np.linspace(ppm_range[0], ppm_range[1], n_points)
    fwhm_ppm = fwhm_hz / freq_mhz
    profs = _lorentzian_profiles(axis, deltas, fwhm_ppm)
    spectrum = intensities @ profs

    return axis, spectrum


def simulate_fid(
    transitions: Sequence[SpinTransition],
    freq_mhz: float = 500.0,
    fwhm_hz: float = 1.0,
    sw_ppm: float = 14.0,
    carrier_ppm: float = 5.0,
    n_points: int = 16384,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the time-domain Free Induction Decay (FID) signal s(t).

    Parameters:
        transitions: List of SpinTransition objects.
        freq_mhz: Spectrometer frequency in MHz (default: 500.0).
        fwhm_hz: Line broadening FWHM in Hz (related to T2 by FWHM = 1 / (pi * T2)).
        sw_ppm: Spectral width in ppm (default: 14.0).
        carrier_ppm: Carrier / center frequency in ppm (default: 5.0).
        n_points: Number of complex FID acquisition points (default: 16384).

    Returns:
        (time_axis, fid_signal) where fid_signal is complex:
        s(t) = sum_k I_k * exp(i * 2*pi * (nu_k - nu_carrier) * t) * exp(-t / T2)
    """
    sw_hz = sw_ppm * freq_mhz
    dt = 1.0 / sw_hz
    t = np.arange(n_points) * dt

    if not transitions:
        return t, np.zeros(n_points, dtype=np.complex128)

    t2 = 1.0 / (math.pi * max(1e-4, fwhm_hz))
    carrier_hz = carrier_ppm * freq_mhz
    signal = np.zeros(n_points, dtype=np.complex128)

    decay = np.exp(-t / t2)
    for tr in transitions:
        nu_offset = tr.freq_hz - carrier_hz
        oscillation = np.exp(1j * (2.0 * math.pi * nu_offset) * t)
        signal += tr.intensity * oscillation

    signal *= decay
    return t, signal


def simulate_1d_from_fid(
    transitions: Sequence[SpinTransition],
    freq_mhz: float = 500.0,
    fwhm_hz: float = 1.0,
    sw_ppm: float = 14.0,
    carrier_ppm: float = 5.0,
    n_points: int = 16384,
    zero_fill_factor: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute continuous 1D absorption spectrum via Discrete Fourier Transform (FFT) of simulated FID."""
    t, fid = simulate_fid(
        transitions,
        freq_mhz=freq_mhz,
        fwhm_hz=fwhm_hz,
        sw_ppm=sw_ppm,
        carrier_ppm=carrier_ppm,
        n_points=n_points,
    )
    n_fft = n_points * zero_fill_factor
    dt = t[1] - t[0] if len(t) > 1 else 1e-4
    freq_axis_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=dt))
    ppm_axis = (freq_axis_hz + carrier_ppm * freq_mhz) / freq_mhz

    if not transitions:
        return ppm_axis, np.zeros(n_fft)

    fid_copy = fid.copy()
    fid_copy[0] *= 0.5
    spectrum_complex = np.fft.fftshift(np.fft.fft(fid_copy, n=n_fft))

    # Real absorption spectrum
    intensity = np.maximum(0.0, np.real(spectrum_complex) * dt)
    return ppm_axis, intensity



def simulate_cosy(
    group_shifts_ppm: Sequence[float],
    group_j: np.ndarray,
    group_counts: Optional[Sequence[int]] = None,
    ppm_range: Optional[Tuple[float, float]] = None,
    n_points: int = 512,
    fwhm_ppm: float = 0.04,
    j_cutoff: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a 2D Homonuclear COSY spectrum with diagonal peaks and
    scalar J-coupling cross-peaks.

    Parameters:
        group_shifts_ppm: List of chemical shifts (ppm) for each group of nuclei.
        group_j: (N, N) matrix of scalar couplings in Hz.
        group_counts: Number of equivalent nuclei in each group (e.g. 3 for methyl).
        ppm_range: (min_ppm, max_ppm) tuple for both F1 and F2 axes.
        n_points: Grid size (n_points x n_points).
        fwhm_ppm: Broadening in ppm for each peak.
        j_cutoff: Minimum J (Hz) to produce a visible cross-peak.

    Returns:
        (f1_axis, f2_axis, 2D matrix of shape (n_points, n_points))
    """
    n_groups = len(group_shifts_ppm)
    shifts = np.array(group_shifts_ppm, dtype=float)
    counts = np.array(group_counts if group_counts is not None else [1] * n_groups, dtype=float)

    if ppm_range is None:
        ppm_range = _auto_range(shifts, pad=0.8, clamp=-0.5) if n_groups > 0 else (0.0, 10.0)

    f1 = np.linspace(ppm_range[0], ppm_range[1], n_points)
    f2 = np.linspace(ppm_range[0], ppm_range[1], n_points)
    matrix = np.zeros((n_points, n_points), dtype=float)

    profiles_1 = _lorentzian_profiles(f1, shifts, fwhm_ppm)
    profiles_2 = _lorentzian_profiles(f2, shifts, fwhm_ppm)
    # 1. Diagonal peaks: intensity proportional to count_i
    for i in range(n_groups):
        diag_intensity = counts[i] * 1.5
        matrix += diag_intensity * np.outer(profiles_1[i], profiles_2[i])

    # 2. Cross-peaks: intensity proportional to |J_ij|
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            val_j = abs(group_j[i, j])
            if val_j >= j_cutoff:
                # Efficiency factor scaled by J (typical J ranges 1 to 15 Hz)
                cross_intensity = min(1.2, val_j / 6.0) * math.sqrt(counts[i] * counts[j])
                # Symmetric cross-peaks at (i, j) and (j, i)
                matrix += cross_intensity * np.outer(profiles_1[i], profiles_2[j])
                matrix += cross_intensity * np.outer(profiles_1[j], profiles_2[i])

    return f1, f2, matrix


def simulate_noesy(
    group_shifts_ppm: Sequence[float],
    group_dist: np.ndarray,
    group_counts: Optional[Sequence[int]] = None,
    ppm_range: Optional[Tuple[float, float]] = None,
    n_points: int = 512,
    fwhm_ppm: float = 0.04,
    freq_mhz: float = 500.0,
    mixing_time: float = 0.5,
    tau_c_ps: float = 100.0,
    r_cutoff: float = 5.0,
    nucleus: Optional[str] = None,
    spin: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a phase-sensitive 2D Homonuclear NOESY spectrum using the full
    Solomon relaxation matrix (dM/dt = -R M).

    Physical model:
    - Extreme narrowing / small molecule dipolar relaxation (omega0 * tau_c << 1).
    - Dipolar spectral densities: J(w) = tau_c / (1 + (w*tau_c)^2).
    - Cross-relaxation: sigma_ij = (K^2 / 10) * (6*J(2w0) - J(0)) / r_ij^6.
    - Auto-relaxation: rho_ij = (K^2 / 10) * (J(0) + 3*J(w0) + 6*J(2w0)) / r_ij^6.
    - Relaxation matrix (Solomon): R_ii = rho_i + leakage, R_ij = +sigma_ij.
    - Intensity matrix: I = expm(-R * mixing_time) * diag(M0).
    - Phase sensitivity: the spectrum is plotted as -I with ONE global sign, so the
      diagonal is NEGATIVE; the cross-peak sign follows from sigma:
        extreme narrowing (sigma > 0, small molecules): cross-peaks POSITIVE (opposite to diagonal);
        slow tumbling (sigma < 0, large molecules / viscous): cross-peaks NEGATIVE (same as diagonal).
    - freq_mhz is the 1H spectrometer frequency; the Larmor frequency of `nucleus` is
      scaled by |gamma_X / gamma_1H|.
    """
    import scipy.linalg

    n_groups = len(group_shifts_ppm)
    shifts = np.array(group_shifts_ppm, dtype=float)
    counts = np.array(group_counts if group_counts is not None else [1] * n_groups, dtype=float)

    if ppm_range is None:
        ppm_range = _auto_range(shifts, pad=0.8, clamp=-0.5) if n_groups > 0 else (0.0, 10.0)

    f1 = np.linspace(ppm_range[0], ppm_range[1], n_points)
    f2 = np.linspace(ppm_range[0], ppm_range[1], n_points)
    matrix = np.zeros((n_points, n_points), dtype=float)

    # Physical parameters for dipole-dipole relaxation
    mu0 = 4.0 * math.pi * 1e-7
    hbar = 1.054571817e-34

    s_val = spin if spin is not None else (get_nucleus_spin(nucleus) if nucleus else 0.5)
    gamma_gyro = 267.522e6
    if nucleus is not None:
        norm = normalize_nucleus_name(nucleus)
        if norm in NUCLEUS_DATA and 'gamma' in NUCLEUS_DATA[norm]:
            gamma_gyro = abs(float(NUCLEUS_DATA[norm]['gamma']))

    gamma_h = abs(float(NUCLEUS_DATA.get('1H', {}).get('gamma', 267.522e6)))
    omega0 = 2.0 * math.pi * freq_mhz * 1e6 * (gamma_gyro / gamma_h)
    tau_c = tau_c_ps * 1e-12
    spin_factor = (4.0 / 3.0) * s_val * (s_val + 1.0)
    K2_10 = (((mu0 / (4.0 * math.pi)) * hbar * (gamma_gyro ** 2)) ** 2) * spin_factor / 10.0

    def J_spec(w: float) -> float:
        return tau_c / (1.0 + (w * tau_c) ** 2)

    J0 = J_spec(0.0)
    Jw = J_spec(omega0)
    J2w = J_spec(2.0 * omega0)

    sigma_factor = K2_10 * (6.0 * J2w - J0)
    rho_factor = K2_10 * (J0 + 3.0 * Jw + 6.0 * J2w)

    R = np.zeros((n_groups, n_groups), dtype=float)
    for i in range(n_groups):
        for j in range(n_groups):
            if i != j:
                r_ang = float(group_dist[i, j])
                if 0.5 < r_ang <= r_cutoff:
                    r_m = r_ang * 1e-10
                    sig = sigma_factor / (r_m ** 6)
                    # Solomon: dMz_i/dt = -rho_i dMz_i - sigma_ij dMz_j  ->  R_ij = +sigma_ij
                    R[i, j] = math.copysign(min(10.0, abs(sig)), sig)

    rho_leak = 0.5  # s^-1 external leakage
    for i in range(n_groups):
        diag_rho = 0.0
        for j in range(n_groups):
            if i != j:
                r_ang = float(group_dist[i, j])
                if 0.5 < r_ang <= r_cutoff:
                    r_m = r_ang * 1e-10
                    diag_rho += rho_factor / (r_m ** 6)
        R[i, i] = min(20.0, diag_rho) + rho_leak

    M0 = np.diag(counts)
    I_mat = scipy.linalg.expm(-R * mixing_time) @ M0

    # One global phase (diagonal negative); the cross-peak sign comes from sigma itself:
    # positive in extreme narrowing, negative (like the diagonal) for slow tumbling.
    A_noesy = -I_mat

    profiles_1 = _lorentzian_profiles(f1, shifts, fwhm_ppm)
    profiles_2 = _lorentzian_profiles(f2, shifts, fwhm_ppm)
    for i in range(n_groups):
        for j in range(n_groups):
            val = A_noesy[i, j]
            if abs(val) > 1e-4:
                matrix += val * np.outer(profiles_1[i], profiles_2[j])

    return f1, f2, matrix


def simulate_1d_lines(
    shifts_ppm: Sequence[float],
    intensities: Optional[Sequence[float]] = None,
    ppm_range: Optional[Tuple[float, float]] = None,
    n_points: int = 16384,
    fwhm_ppm: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a 1D decoupled spectrum (e.g. 13C{1H} or single lines) with Lorentzian lineshapes."""
    n_lines = len(shifts_ppm)
    shifts = np.array(shifts_ppm, dtype=float)
    ints = np.array(intensities if intensities is not None else [1.0] * n_lines, dtype=float)

    if ppm_range is None:
        if n_lines > 0:
            pad = max(5.0, 0.1 * (float(np.max(shifts)) - float(np.min(shifts))))
            ppm_range = _auto_range(shifts, pad=pad, clamp=-5.0)
        else:
            ppm_range = (0.0, 220.0)

    ppm_axis = np.linspace(ppm_range[0], ppm_range[1], n_points)
    profs = _lorentzian_profiles(ppm_axis, shifts, fwhm_ppm)
    spectrum = ints @ profs if n_lines > 0 else np.zeros(n_points, dtype=float)
    return ppm_axis, spectrum


def simulate_heteronuclear_2d(
    shifts_f2_ppm: Sequence[float],
    shifts_f1_ppm: Sequence[float],
    correlations: Sequence[Tuple[int, int, float]],
    ppm_range_f2: Optional[Tuple[float, float]] = None,
    ppm_range_f1: Optional[Tuple[float, float]] = None,
    n_points_f2: int = 512,
    n_points_f1: int = 512,
    fwhm_f2_ppm: Optional[float] = None,
    fwhm_f1_ppm: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a 2D Heteronuclear correlation spectrum between arbitrary nuclei F2 (horizontal) and F1 (vertical).

    Parameters:
        shifts_f2_ppm: Chemical shifts along F2 (ppm).
        shifts_f1_ppm: Chemical shifts along F1 (ppm).
        correlations: List of (f2_idx, f1_idx, intensity_weight) tuples.
        ppm_range_f2: (min, max) range for F2 axis.
        ppm_range_f1: (min, max) range for F1 axis.
        n_points_f2: Grid points along F2.
        n_points_f1: Grid points along F1.
        fwhm_f2_ppm: Linewidth along F2 (ppm).
        fwhm_f1_ppm: Linewidth along F1 (ppm).

    Returns:
        (f1_axis, f2_axis, matrix_2d)
    """
    s_f2 = np.array(shifts_f2_ppm, dtype=float)
    s_f1 = np.array(shifts_f1_ppm, dtype=float)

    if ppm_range_f2 is None:
        pad_2 = max(0.8, 0.12 * (float(np.max(s_f2)) - float(np.min(s_f2)))) if len(s_f2) > 0 else 1.0
        min_2 = float(np.min(s_f2)) - pad_2 if len(s_f2) > 0 else 0.0
        max_2 = float(np.max(s_f2)) + pad_2 if len(s_f2) > 0 else 10.0
        ppm_range_f2 = (min_2, max_2)

    if ppm_range_f1 is None:
        pad_1 = max(1.0, 0.12 * (float(np.max(s_f1)) - float(np.min(s_f1)))) if len(s_f1) > 0 else 1.0
        min_1 = float(np.min(s_f1)) - pad_1 if len(s_f1) > 0 else 0.0
        max_1 = float(np.max(s_f1)) + pad_1 if len(s_f1) > 0 else 100.0
        ppm_range_f1 = (min_1, max_1)

    f2 = np.linspace(ppm_range_f2[0], ppm_range_f2[1], n_points_f2)
    f1 = np.linspace(ppm_range_f1[0], ppm_range_f1[1], n_points_f1)
    matrix = np.zeros((n_points_f1, n_points_f2), dtype=float)

    if fwhm_f2_ppm is None:
        span_f2 = abs(ppm_range_f2[1] - ppm_range_f2[0])
        fwhm_f2_ppm = max(0.02, span_f2 * 0.008)
    if fwhm_f1_ppm is None:
        span_f1 = abs(ppm_range_f1[1] - ppm_range_f1[0])
        fwhm_f1_ppm = max(0.05, span_f1 * 0.008)

    prof_f2 = _lorentzian_profiles(f2, s_f2, fwhm_f2_ppm)
    prof_f1 = _lorentzian_profiles(f1, s_f1, fwhm_f1_ppm)

    for i2, i1, weight in correlations:
        if 0 <= i2 < len(s_f2) and 0 <= i1 < len(s_f1):
            matrix += weight * np.outer(prof_f1[i1], prof_f2[i2])

    return f1, f2, matrix


def simulate_hsqc(
    h_shifts_ppm: Sequence[float],
    c_shifts_ppm: Sequence[float],
    ch_correlations: Sequence[Tuple[int, int, str]],
    ppm_range_f2: Optional[Tuple[float, float]] = None,
    ppm_range_f1: Optional[Tuple[float, float]] = None,
    n_points_f2: int = 512,
    n_points_f1: int = 512,
    fwhm_f2_ppm: float = 0.04,
    fwhm_f1_ppm: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a 2D Heteronuclear Single Quantum Coherence (HSQC / edited HSQC) spectrum.

    Correlates 1H (F2, horizontal) with directly bonded 13C (F1, vertical).
    In multiplicity-edited HSQC (HSQC-DEPT):
    - CH and CH3 peaks have POSITIVE phase (+1.0).
    - CH2 peaks have NEGATIVE phase (-1.0).
    - Quaternary carbons have no cross-peaks.
    """
    h_arr = np.asarray(h_shifts_ppm, dtype=float)
    c_arr = np.asarray(c_shifts_ppm, dtype=float)
    if ppm_range_f2 is None:
        ppm_range_f2 = _auto_range(h_arr, pad=0.8, clamp=-0.5) if len(h_arr) > 0 else (0.0, 10.0)
    if ppm_range_f1 is None:
        ppm_range_f1 = _auto_range(c_arr, pad=10.0, clamp=-5.0) if len(c_arr) > 0 else (0.0, 220.0)

    correlations = [
        (h_idx, c_idx, -1.0 if 'CH2' in mult.upper() else 1.0)
        for h_idx, c_idx, mult in ch_correlations
    ]
    return simulate_heteronuclear_2d(
        h_shifts_ppm, c_shifts_ppm, correlations,
        ppm_range_f2=ppm_range_f2, ppm_range_f1=ppm_range_f1,
        n_points_f2=n_points_f2, n_points_f1=n_points_f1,
        fwhm_f2_ppm=fwhm_f2_ppm, fwhm_f1_ppm=fwhm_f1_ppm,
    )


def simulate_hmbc(
    h_shifts_ppm: Sequence[float],
    c_shifts_ppm: Sequence[float],
    ch_correlations: Sequence[Tuple[int, int, float]],
    ppm_range_f2: Optional[Tuple[float, float]] = None,
    ppm_range_f1: Optional[Tuple[float, float]] = None,
    n_points_f2: int = 512,
    n_points_f1: int = 512,
    fwhm_f2_ppm: float = 0.04,
    fwhm_f1_ppm: float = 1.0,
    tau_lr: float = 0.065,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a 2D Heteronuclear Multiple Bond Correlation (HMBC) spectrum.

    Correlates 1H (F2, horizontal) with long-range coupled 13C across 2 or 3 bonds (F1, vertical).
    Intensity follows transfer efficiency I ~ sin^2(pi * |J_CH| * tau_lr).
    Direct 1-bond couplings are filtered out.
    """
    h_arr = np.asarray(h_shifts_ppm, dtype=float)
    c_arr = np.asarray(c_shifts_ppm, dtype=float)
    if ppm_range_f2 is None:
        ppm_range_f2 = _auto_range(h_arr, pad=0.8, clamp=-0.5) if len(h_arr) > 0 else (0.0, 10.0)
    if ppm_range_f1 is None:
        ppm_range_f1 = _auto_range(c_arr, pad=10.0, clamp=-5.0) if len(c_arr) > 0 else (0.0, 220.0)

    correlations = [
        (h_idx, c_idx, max(0.20, math.sin(math.pi * abs(j_val) * tau_lr) ** 2))
        for h_idx, c_idx, j_val in ch_correlations
    ]
    return simulate_heteronuclear_2d(
        h_shifts_ppm, c_shifts_ppm, correlations,
        ppm_range_f2=ppm_range_f2, ppm_range_f1=ppm_range_f1,
        n_points_f2=n_points_f2, n_points_f1=n_points_f1,
        fwhm_f2_ppm=fwhm_f2_ppm, fwhm_f1_ppm=fwhm_f1_ppm,
    )

def _finish_figure(fig, save_path: Optional[str], show: bool, dpi: int):
    """Save (and report) the figure when a path is given; show only when asked; return it."""
    from frogchem import plotter
    if save_path:
        plotter.save_figure(fig, save_path, dpi=dpi, show=show)
    elif show:
        plotter.show_figures()
    return fig


#: Colour of 1D spectra and of the 1D traces of 2D spectra (overridden by --colors)
SPECTRUM_COLOR = '#004b87'


def plot_1d_spectrum(
    ppm_axis: np.ndarray,
    intensity: np.ndarray,
    title: str = "Simulated 1H NMR Spectrum",
    save_path: Optional[str] = None,
    show: bool = False,
    nucleus: str = "1H",
    dpi: int = 300,
    opts=None,
):
    """Plot a 1D NMR spectrum with inverted ppm axis (NMR convention), in the FROG style.

    opts: frogchem.plotter.PlotOptions (title, colors, limits, ...); options set there win over
    the arguments above. Returns the figure. With `save_path` it is saved (and closed unless
    `show`); without it the caller saves it, e.g. with CLICommand.save_figure.
    """
    from frogchem import plotter

    nuc = normalize_nucleus_name(nucleus)
    x_label = r"$\delta$ / ppm ($^{13}\mathrm{C}\{^{1}\mathrm{H}\}$)" if nuc == '13C' else r"$\delta$ / ppm (" + nuc + ")"
    top = float(np.max(intensity)) * 1.05 if len(intensity) > 0 else 0.0
    opts = (opts or plotter.PlotOptions()).with_defaults(
        title=title, xlabel=x_label, ylabel="Intensity (a.u.)", colors=(SPECTRUM_COLOR,),
        linewidth=1.2, grid=True, legend=False, figsize=(10, 5),
        ylim=(0.0, top if top > 0 else 1.0),
    )
    # NMR convention: higher ppm on the left, lower ppm on the right
    fig, _ax = plotter.line(ppm_axis, intensity, fill=True, invert_x=True, opts=opts)
    return _finish_figure(fig, save_path, show, dpi)


def _trace_max(ppm: np.ndarray, intensity: np.ndarray, lo: float, hi: float) -> float:
    """Largest intensity of a 1D trace inside the plotted ppm window (1 when empty)."""
    visible = intensity[(ppm >= lo) & (ppm <= hi)]
    if len(visible) and np.max(visible) > 0:
        return float(np.max(visible))
    return float(np.max(intensity)) if len(intensity) and np.max(intensity) > 0 else 1.0


def plot_2d_spectrum(
    f1_axis: np.ndarray,
    f2_axis: np.ndarray,
    matrix: np.ndarray,
    title: str = "2D NMR Spectrum",
    spectrum_type: str = "COSY",
    save_path: Optional[str] = None,
    show: bool = False,
    nucleus: str = "1H",
    nucleus_f1: Optional[str] = None,
    nucleus_f2: Optional[str] = None,
    n_contours: int = 14,
    cmap: Optional[str] = None,
    trace_1d: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    trace_1d_top: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    trace_1d_left: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    show_diagonal: Optional[bool] = None,
    dpi: int = 300,
    opts=None,
):
    """Plot 2D NMR spectrum (COSY, NOESY, HSQC, or HMBC) contour plot with inverted axes.

    Features:
    - Integrated 1D Spectrum Traces: Plots the calculated 1D spectrum along the top (F2)
      and left (F1) margins, matching standard NMR practice (TopSpin / MestReNova).
    - Phase-sensitive NOESY & Edited HSQC:
      * NOESY: Negative diagonal in Navy, Positive cross-peaks in Crimson Red.
      * HSQC: Positive CH/CH3 in Navy, Negative CH2 in Crimson Red.
    - Magnitude COSY & HMBC: Uses geometric level progression with a vibrant colormap
      (default: 'turbo') that cleanly distinguishes weak cross-peaks from strong peaks.
    - Right-Side Y-Axis: Numbers and F1 label on the right side of the 2D plot box.

    opts: frogchem.plotter.PlotOptions; cmap applies to magnitude spectra, colors[0] to the
    1D traces, linewidth to the contours. Returns the figure; with `save_path` it is also
    saved (see plot_1d_spectrum).
    """
    from dataclasses import replace

    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from frogchem import plotter

    nuc_1 = normalize_nucleus_name(nucleus_f1 if nucleus_f1 is not None else nucleus)
    nuc_2 = normalize_nucleus_name(nucleus_f2 if nucleus_f2 is not None else nucleus)
    opts = (opts or plotter.PlotOptions()).with_defaults(
        title=title, cmap=cmap, figsize=(8.5, 8.5), linewidth=0.9, grid=True,
        xlabel=r"$F_2$ [ppm] (" + nuc_2 + ")", ylabel=r"$F_1$ [ppm] (" + nuc_1 + ")",
    )
    level_opts = plotter.PlotOptions(linewidth=opts.linewidth, linestyle=opts.linestyle)
    trace_opts = plotter.PlotOptions(colors=(opts.colors or (SPECTRUM_COLOR,))[:1], linewidth=1.0,
                                     legend=False, grid=False)

    if show_diagonal is None:
        show_diagonal = (nuc_1 == nuc_2)

    if trace_1d is not None:
        if trace_1d_top is None:
            trace_1d_top = trace_1d
        if trace_1d_left is None:
            trace_1d_left = trace_1d

    pos_max = float(np.max(matrix)) if np.max(matrix) > 0 else 1.0
    neg_min = float(np.min(matrix))
    spec_upper = spectrum_type.upper()
    has_negative = (neg_min < -0.005 * pos_max) or (spec_upper in ('NOESY', 'HSQC') and neg_min < -1e-4)

    with plotter.style(opts):
        fig, ax = plotter.new_figure(opts=opts)

        sm = None
        legend_handles = None
        if has_negative:
            neg_mag = abs(neg_min)

            # Negative contours. Colour choice: for NOESY, diagonal is navy; for HSQC, CH2 is red
            neg_base = max(1e-4, neg_mag * 0.015)
            neg_levels = -np.geomspace(neg_base, neg_mag * 0.90, n_contours)[::-1]
            if spec_upper == 'HSQC':
                neg_cmap = plotter.get_cmap('Reds')
            else:
                neg_cmap = plotter.get_cmap(opts.cmap if opts.cmap and 'Blues' in opts.cmap else 'Blues')
            plotter.contour(f2_axis, f1_axis, matrix, levels=neg_levels,
                            colors=neg_cmap(np.linspace(0.45, 0.95, len(neg_levels))), ax=ax, opts=level_opts)

            # Positive contours
            pos_base = max(1e-4, pos_max * 0.03)
            pos_levels = np.geomspace(pos_base, pos_max * 0.90, n_contours)
            pos_cmap = plotter.get_cmap('Blues' if spec_upper == 'HSQC' else 'Reds')
            plotter.contour(f2_axis, f1_axis, matrix, levels=pos_levels,
                            colors=pos_cmap(np.linspace(0.45, 0.95, len(pos_levels))),
                            ax=ax, opts=level_opts, linestyles='solid')

            if spec_upper == 'HSQC':
                legend_handles = [
                    Line2D([0], [0], color='#08519c', lw=2, label=r'$\mathrm{CH\ /\ CH_3\ (Positive)}$'),
                    Line2D([0], [0], color='#cb181d', lw=2, label=r'$\mathrm{CH_2\ (Negative)}$'),
                ]
            else:
                legend_handles = [
                    Line2D([0], [0], color='#08519c', lw=2, label='Diagonal (Negative)'),
                    Line2D([0], [0], color='#cb181d', lw=2, label='NOE Cross-peaks (Positive)'),
                ]
        else:
            # Magnitude spectrum (COSY / HMBC)
            levels = np.geomspace(pos_max * 0.015, pos_max * 0.90, n_contours)
            color_map = plotter.get_cmap(opts.cmap or 'turbo')
            plotter.contour(f2_axis, f1_axis, matrix, levels=levels,
                            colors=color_map(np.linspace(0.15, 0.95, len(levels))), ax=ax, opts=level_opts)
            sm = cm.ScalarMappable(norm=Normalize(vmin=1.5, vmax=100.0), cmap=color_map)
            sm.set_array([])

        min_f2, max_f2 = min(float(f2_axis[0]), float(f2_axis[-1])), max(float(f2_axis[0]), float(f2_axis[-1]))
        min_f1, max_f1 = min(float(f1_axis[0]), float(f1_axis[-1])), max(float(f1_axis[0]), float(f1_axis[-1]))

        # Diagonal reference line if homonuclear
        if show_diagonal:
            d_min = max(min_f1, min_f2)
            d_max = min(max_f1, max_f2)
            ax.plot([d_min, d_max], [d_min, d_max], color='gray', linestyle=':', lw=1.0, alpha=0.7)

        ax.set_xlim(max_f2, min_f2)
        ax.set_ylim(max_f1, min_f1)
        ax.set_aspect('equal' if nuc_1 == nuc_2 else 'auto')

        divider = None
        if trace_1d_top is not None or trace_1d_left is not None:
            divider = make_axes_locatable(ax)

        ax_top = None
        if trace_1d_top is not None:
            ppm_top, int_top = trace_1d_top
            ax_top = divider.append_axes('top', size='18%', pad=0.06, sharex=ax)
            plotter.line(ppm_top, int_top, fill=True, ax=ax_top, opts=trace_opts)
            ax_top.set_ylim(0.0, _trace_max(ppm_top, int_top, min_f2, max_f2) * 1.08)
            ax_top.tick_params(labelbottom=False, bottom=False, left=False, labelleft=False)
            for s in ['top', 'left', 'right']:
                ax_top.spines[s].set_visible(False)

        if trace_1d_left is not None:
            ppm_left, int_left = trace_1d_left
            ax_left = divider.append_axes('left', size='18%', pad=0.06, sharey=ax)
            plotter.line(ppm_left, int_left, fill=True, vertical=True, ax=ax_left, opts=trace_opts)
            ax_left.set_xlim(_trace_max(ppm_left, int_left, min_f1, max_f1) * 1.08, 0.0)
            ax_left.tick_params(labelleft=False, left=False, bottom=False, labelbottom=False)
            for s in ['top', 'bottom', 'left']:
                ax_left.spines[s].set_visible(False)

        # Move Y-axis numbers and label to the right side of the 2D spectrum
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.tick_params(labelright=True, right=True, labelleft=False, left=True)
        plotter.finish(ax, replace(opts, title=None), handles=legend_handles, loc='upper left', framealpha=0.9)
        if opts.grid:
            ax.grid(True, alpha=0.3)
        if opts.title:
            (ax_top if ax_top is not None else ax).set_title(opts.title, pad=10)

        if sm is not None:
            if divider is not None:
                plotter.colorbar(sm, cax=divider.append_axes('right', size='3.5%', pad=0.7),
                                 label="Relative Intensity (%)")
            else:
                plotter.colorbar(sm, ax=ax, label="Relative Intensity (%)")

    return _finish_figure(fig, save_path, show, dpi)
