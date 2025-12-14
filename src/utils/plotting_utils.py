import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter


def advanced_stats(window, std_thresh, grad_thresh, SA, T):
    if SA.dtype.byteorder == '>':  # Big-endian
        SA = SA.byteswap().newbyteorder()

    rolling_std = pd.Series(SA).rolling(window=window, center=True).std()
    grad = np.abs(np.gradient(SA, T))
    
    stable_mask = (rolling_std < std_thresh) & (grad < grad_thresh)
    return stable_mask


def find_longest_stable(stable_mask, SA, T):
    stable_indices = np.where(stable_mask)[0]

    # Group into contiguous stable segments
    segments = []
    for k, g in groupby(enumerate(stable_indices), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), g))
        segments.append(group)

    if not segments:
        return None, None, None, None  # no stable region found

    # Find the longest one
    longest_segment = max(segments, key=len)
    longest_mask = np.zeros_like(SA, dtype=bool)
    longest_mask[longest_segment] = True

    # Stats
    start_time = T[longest_segment[0]]
    end_time = T[longest_segment[-1]]
    mean_a = np.mean(SA[longest_segment])

    return longest_mask, mean_a, start_time, end_time

def get_stable_regions(stable_mask, T):
    """
    Analyze stable mask to find all stable regions, their durations, and stats.
    
    Parameters:
        stable_mask: boolean array, True where signal is stable
        T: time array (same length)

    Returns:
        total_duration: sum of durations of all stable segments
        longest_duration: duration of the longest stable segment
        all_segments: list of (start_time, end_time, duration)
    """
    indices = np.where(stable_mask)[0]

    if len(indices) == 0:
        return 0, 0, []

    segments = []
    for k, g in groupby(enumerate(indices), lambda x: x[0]-x[1]):
        group = list(map(itemgetter(1), g))
        start, end = group[0], group[-1]
        duration = T[end] - T[start]
        segments.append((T[start], T[end], duration))

    durations = [d for _, _, d in segments]
    total_duration = np.sum(durations)
    longest_duration = np.max(durations)

    return total_duration, longest_duration, segments

def format_to_1sf(x, pos=None):
    """Format a number to 1 significant figure for axis labels."""
    if x == 0:
        return "0"
    else:
        return f"{x:.1g}"

def rescale_ticks(x, pos):
    """Rescale ticks to percentage format."""
    return f"{x * 100:.1f}" 

def format_func(value, tick_number):
    """Format function for matplotlib ticks to handle special cases."""
    if value == 0:
        return "0"
    elif value == np.pi:
        return r"$\pi$"
    elif value == -np.pi:
        return r"$-\pi$"
    elif value == 2 * np.pi:
        return r"$2\pi$"
    elif value == -2 * np.pi:
        return r"$-2\pi$"
    else:
        return r"${0:.0f}\pi$".format(value / np.pi)
    
def sci_notation_latex(x, pos=None):
    """Format a number as 2.3 × 10^4 for axis labels (LaTeX style)."""
    if x == 0:
        return "0"
    if x >= 100:
        exponent = int(np.floor(np.log10(abs(x))))
        coeff = x / 10**exponent
        # Use LaTeX formatting for matplotlib
        return r"${:.2g} \times 10^{{{}}}$".format(coeff, exponent)
    else:
        # For small numbers, use standard formatting
        return r"${:.2g}$".format(x)