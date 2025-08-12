from typing import Tuple, Optional

from panqec.error_models import PauliErrorModel
from panqec.codes import StabilizerCode
from panqec.bpauli import pauli_to_bsf

import numpy as np
from scipy.special import erfinv


def get_std(p: float):
    """
    Find the standard deviation of a gauss distribution such that the likelihood
    of incorrect decision is p.

    "Incorrect decision" here means measuring a value outside the interval
    [-sqrt(pi)/2, sqrt(pi)/2], as described in
    
    https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.021054.

    Parameters
    ----------
    p : float
        Probability of making an incorrect decision based on sampling from
        the Gaussian distribution.
    """
    eps = 1e-10
    std = np.sqrt(np.pi/8.0) / (erfinv(1.0-p) + eps)
    return std


def sample_from_gauss(std_X:float, std_Z: float, rng: Optional[np.random.Generator]=None) -> Tuple:
    """
    Draw a sample from two normal distributions, one for X and one for Z, and return a tuple
    containing the measured values and what type of error occurs (if any).
    Each sample produces an error if the measured value is outside the interval
    [-sqrt(pi)/2, sqrt(pi)/2], as described in
    
    https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.021054.
    
    If both samples gives an error, this corresponds to a Y error, and if neither gives an
    error there is no error.

    Parameters
    ----------
    std_X : ndarray
        The standard deviation of the X distribution.
    std_Z : ndarray
        The standard deviation of the Z distribution.
    rng : np.random.Generator   
        Random number generator to use for sampling. If None: Defaults to
        np.random.default_rng().
    
    Returns
    -------
    str
        The error as a Pauli string: "I" for identity (no error), and
        "X", "Y" and "Z" for X, Y and Z errors respectively.
    ndarray
        The minimized deviation from the X distribution (Delta_X_m).
    ndarray
        The minimized deviation from the Z distribution (Delta_Z_m).
    """
    rng = np.random.default_rng() if rng is None else rng

    dist = np.sqrt(np.pi) # Distance between q-values
    th = dist / 2.0 # Threshold for producing an error

    x = rng.normal(loc=0.0, scale=std_X)
    z = rng.normal(loc=0.0, scale=std_Z)

    x_abs = np.abs(x)
    z_abs = np.abs(z)

    if x_abs > th and z_abs <= th:
        Delta_X_m = np.abs(dist - x_abs)
        Delta_Z_m = np.abs(z_abs)
        pauli_err = "X"
    elif x_abs <= th and z_abs > th:
        Delta_X_m = np.abs(x_abs)
        Delta_Z_m = np.abs(dist - z_abs)
        pauli_err = "Z"
    elif x_abs > th and z_abs > th:
        Delta_X_m = np.abs(dist - x_abs)
        Delta_Z_m = np.abs(dist - z_abs)
        pauli_err = "Y"
    else:
        Delta_X_m = np.abs(x_abs)
        Delta_Z_m = np.abs(z_abs)
        pauli_err = "I"
    
    return (pauli_err, Delta_X_m, Delta_Z_m)


class GaussPauliErrorModel(PauliErrorModel):
    """
    Error model for a QEC model with error channels produced by Gaussian likelihoods as
    described by Fukui, Tomita and Okamoto in
        
    "High-Threshold Fault-Tolerant Quantum Computation with Analog Quantum Error Correction"

    https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.021054
    """
    def generate(
            self,
            code: StabilizerCode,
            error_rate: float,
            rng=None
    ):
        """
        Generate errors. Copied and slightly modified from PanQEC.
        """
        rng = np.random.default_rng() if rng is None else rng

        r_x, r_y, r_z = self.direction

        ## Check whether we should generate X-errors, Z-errors or both.
        x_and_z_errors = r_x==0.5 and r_y == 0.0 and r_z == 0.5
        only_x_errors = r_x==1.0 and r_y == 0.0 and r_z == 0.0
        only_z_errors = r_x==0.0 and r_y == 0.0 and r_z == 1.0
        if x_and_z_errors:
            q = 1 - np.sqrt(1-error_rate)
            qx = q
            qz = q
        elif only_x_errors:
            qx = error_rate
            qz = 0
        elif only_z_errors:
            qx = 0
            qz = error_rate
        elif hasattr(self, "DEV_BYPASS"):
            ## This is a naive, non-working approach to dealing with arbitrary error distributions
            ## It does not work, but can be accessed for development/testing purposes by assigning
            ## a variable with the name DEV_BYPASS to the error model before generating errors.
            ## See plot_error_rates() where this has been used to showcase the methods invalidity.
            qx = error_rate*(r_x + r_y)
            qz = error_rate*(r_z + r_y)
        else:
            raise ValueError(f"Due to the way errors are sampled in GaussPauliErrorModel there are only three valid inputs for (r_x,r_y,r_z):\n  - (1.0, 0.0, 0.0)\n  - (0.0, 0.0, 1.0)\n  - (0.5, 0.0, 0.5)\nYour input: ({r_x}, {r_y}, {r_z})")

        std_X = get_std(qx)
        std_Z = get_std(qz)

        self.std_X = std_X
        self.std_Z = std_Z

        error_pauli = ""
        Delta_X_m_arr = np.zeros(code.n)
        Delta_Z_m_arr = np.zeros(code.n)
        for i in range(code.n):
            pauli, Delta_X_m, Delta_Z_m = sample_from_gauss(std_X, std_Z, rng)
            Delta_X_m_arr[i] = Delta_X_m
            Delta_Z_m_arr[i] = Delta_Z_m

            error_pauli = error_pauli + pauli

        self.Delta_X_m_arr = Delta_X_m_arr
        self.Delta_Z_m_arr = Delta_Z_m_arr
        error = pauli_to_bsf(error_pauli)

        return error


def plot_error_rates():
    """
    Check that GaussPauliErrorModel produces the error rates and distributions we expect it to.

    NOTE:
        So far the Gaussian error model only works (produces the correct rate and distribution
        of errors) for the "Only X", "Only Z" and "50% X, 50% Z" cases.
    """
    import matplotlib.pyplot as plt
    from panqec.codes import Toric2DCode
    from panqec.bpauli import bsf_to_pauli

    Err_mod = GaussPauliErrorModel
    r_names = ["Only X", "Only Z", "50/50 X/Z", "20/80 X/Z", "33/33/33 X/Y/Z"]
    rs = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.5],
        [0.2, 0.0, 0.8],
        [1/3, 1/3, 1/3]
    ]
    n_r = len(rs)

    code = Toric2DCode(6)
    n_p = 40
    p_vals = np.linspace(0.0, 1.0, n_p)

    n_iter = 1000

    p_computed = np.zeros((n_r, n_p))
    r_x_computed = np.zeros((n_r, n_p))
    r_y_computed = np.zeros((n_r, n_p))
    r_z_computed = np.zeros((n_r, n_p))

    for r_ind, r_xyz in enumerate(rs):
        r_x, r_y, r_z = r_xyz

        for row, p in enumerate(p_vals):
            error_model = Err_mod(r_x, r_y, r_z)
            error_model.DEV_BYPASS = True
            n_ixyz = np.zeros((n_iter, 4), dtype=int)
            for i in range(n_iter):
                progress_percent = 100 * (r_ind*n_iter*n_p + row*n_iter+i)/(n_p*n_iter*n_r)
                print(f"Simulation: {r_ind+1:>2}/{n_r:<2} | Error rate: {row+1:>3}/{n_p:<3} | Total progress: {progress_percent:>3.0f}%", end="\r")
                error = error_model.generate(code, p)
                err_pauli = bsf_to_pauli(error)
                
                n_ixyz[i,0] = err_pauli.count("I")
                n_ixyz[i,1] = err_pauli.count("X")
                n_ixyz[i,2] = err_pauli.count("Y")
                n_ixyz[i,3] = err_pauli.count("Z")
            
            n_tot_ixyz = np.sum(n_ixyz, axis=0)
            n_tot = np.sum(n_tot_ixyz)
            n_tot_xyz = np.sum(n_tot_ixyz[1:])
            p_ixyz = n_tot_ixyz / n_tot
            if n_tot_xyz > 0:
                r_xyz = n_tot_ixyz[1:] / n_tot_xyz
            else:
                r_xyz = np.empty(3)
                r_xyz[:] = np.nan

            r_x_computed[r_ind, row] = r_xyz[0]
            r_y_computed[r_ind, row] = r_xyz[1]
            r_z_computed[r_ind, row] = r_xyz[2]

            p_error = np.sum(p_ixyz[1:])
            p_computed[r_ind, row] = p_error
            
            p_error = np.sum(p_ixyz[1:])
            p_computed[r_ind, row] = p_error

    # create 4x1 subfigs
    plt.style.use("seaborn-v0_8")
    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f"Comparison of input and output of $p$ and $r$ for {Err_mod.__name__}")
    
    subfigs = fig.subfigures(nrows=1, ncols=n_r)
    
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f"{r_names[row]}")

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=3, ncols=1)
        
        ## Plot the computed error rate
        axs[0].axline((0,0),(1,1), linestyle="dashed", color="black", label="Expected p")
        axs[0].plot(p_vals, p_computed[row,:], "r", label="Computed p")
        axs[0].set_xlim(-0.1, 1.1)
        axs[0].set_ylim(-0.1, 1.1)
        if row==0:
            axs[0].set_ylabel("$p_{out}$")
            axs[0].legend()
        
        ## Plot the computed error distribution
        axs[1].plot(p_vals, r_x_computed[row,:], "r", label="$r_x$")
        axs[1].plot(p_vals, r_z_computed[row,:], "b", label="$r_z$")
        axs[1].plot(p_vals, r_y_computed[row,:], color="orange", linestyle="dashed", label="$r_y$")
        if row==0:
            axs[1].set_ylabel("$r_{out}$")
            axs[1].legend()
        
        ## Plot ratio of X errors to total X and Z errors (not including Y)
        r_ratio_true = rs[row][0] / (rs[row][0] + rs[row][2])
        r_xz_sum = r_x_computed[row,:] + r_z_computed[row,:]
        r_ratio_computed = np.divide(
            r_x_computed[row,:], r_xz_sum,
            out=np.ones_like(r_xz_sum)*r_ratio_true, # If r_x=r_z=0, the ratio is always correct
            where=r_xz_sum != 0.0 # Avoid zero division
        )
        axs[2].axhline(r_ratio_true, color="black", linestyle="dashed", label="Expected")
        axs[2].plot(p_vals, r_ratio_computed, color="red", label="Computed")
        if row==0:
            axs[2].set_ylabel("$\\frac{r_x}{r_x+r_z}$")
            axs[2].legend()

    fig.supxlabel("Error rate input $p_{in}$")
    fig.savefig("Plots/gauss_error_test.pdf", bbox_inches="tight")


if __name__ == "__main__":
    plot_error_rates()