"""
Module: utils.py
Authors: Christian Bergler, Hendrik Schroeter
GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 27.12.2022
"""

"""Return `same` padding for a given kernel size."""
def get_padding(kernel_size):
    if isinstance(kernel_size, int):
        return kernel_size // 2
    else:
        return tuple(s // 2 for s in kernel_size)
