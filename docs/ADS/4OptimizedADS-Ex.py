from inspect import getsourcefile
from pathlib import Path
from typing import Callable, List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from numpy.typing import NDArray
from daceypy import ADS, DA, array, RK
from daceypy import ADSintegrator, ADSintegrator_optimized
from daceypy import ADS_utils 
from daceypy._ADSintegrator import ADSstate
import time
import itertools


def CR3BP(x: array, t: float, mu: float) -> array:
    """
    Circular Restricted Three Body Problem dynamics in normalized coordinates.
    
    Args:
        x: State vector [x, y, z, vx, vy, vz]
        t: Time (unused in autonomous system)
        mu: Mass parameter (Earth-Moon: ~0.01215)
    
    Returns:
        State derivative [vx, vy, vz, ax, ay, az]
    """
    # Extract position and velocity
    pos: array = x[:3]
    vel: array = x[3:]
    
    x_pos, y_pos, z_pos = pos[0], pos[1], pos[2]
    
    # Distances from the two primaries
    r1 = ((x_pos + mu)**2 + y_pos**2 + z_pos**2)**0.5
    r2 = ((x_pos - 1 + mu)**2 + y_pos**2 + z_pos**2)**0.5
    
    # Pseudo-potential gradient
    Omega_x = x_pos - (1 - mu) * (x_pos + mu) / r1**3 - mu * (x_pos - 1 + mu) / r2**3
    Omega_y = y_pos - (1 - mu) * y_pos / r1**3 - mu * y_pos / r2**3
    Omega_z = -(1 - mu) * z_pos / r1**3 - mu * z_pos / r2**3
    
    # Accelerations (Coriolis + centrifugal + gravitational)
    ax = 2 * vel[1] + Omega_x
    ay = -2 * vel[0] + Omega_y
    az = Omega_z
    
    acc = array([ax, ay, az])
    
    # State derivative
    dx = vel.concat(acc)
    return dx


class AutomaticADS_CR3BP_integrator_optimized(ADSintegrator_optimized):
    """
    Custom child class of ADSintegrator_optimized for CR3BP dynamics.
    """
    def __init__(self, mu: float, RK: RK.RKCoeff = RK.RK78()):
        super(AutomaticADS_CR3BP_integrator_optimized, self).__init__(RK)
        self.mu = mu
    
    def f(self, x, t):
        return CR3BP(x, t, self.mu)


class AutomaticADS_CR3BP_integrator(ADSintegrator):
    """
    Custom child class of ADSintegrator for CR3BP dynamics.
    """
    def __init__(self, mu: float, RK: RK.RKCoeff = RK.RK78()):
        super(AutomaticADS_CR3BP_integrator, self).__init__(RK)
        self.mu = mu
    
    def f(self, x, t):
        return CR3BP(x, t, self.mu)


from matplotlib.cm import get_cmap

def plot_ADS_boxes_3D(
    domain_boxes,
    points,
    point_to_domain,
    title_prefix="ADS domains",
    ndim=6
):
    """
    Plot 3D delle box ADS e dei punti associati.
    Separato per posizione (0,1,2) e velocità (3,4,5).
    """

    cmap = mpl.colormaps.get_cmap("tab10")
    dim_sets = [(0,1,2), (3,4,5)] if ndim >= 6 else [(0,1,2)]

    for dims in dim_sets:
        if dims[2] >= ndim:
            continue

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
        i1, i2, i3 = dims

        for j_domain, d in enumerate(domain_boxes):
            indices = np.where(point_to_domain == j_domain)[0]
            has_pts = indices.size > 0
            color = cmap(j_domain % 10)

            # ---- Box ----
            min_v, max_v = d["min"], d["max"]
            v = np.array([
                [min_v[i1], min_v[i2], min_v[i3]],
                [max_v[i1], min_v[i2], min_v[i3]],
                [max_v[i1], max_v[i2], min_v[i3]],
                [min_v[i1], max_v[i2], min_v[i3]],
                [min_v[i1], min_v[i2], max_v[i3]],
                [max_v[i1], min_v[i2], max_v[i3]],
                [max_v[i1], max_v[i2], max_v[i3]],
                [min_v[i1], max_v[i2], max_v[i3]],
            ])

            faces = [
                [v[0],v[1],v[2],v[3]],
                [v[4],v[5],v[6],v[7]],
                [v[0],v[1],v[5],v[4]],
                [v[2],v[3],v[7],v[6]],
                [v[1],v[2],v[6],v[5]],
                [v[4],v[7],v[3],v[0]]
            ]

            edge_w = 2.0 if has_pts else 0.5
            edge_c = 'black' if has_pts else color
            face_a = 0.18 if has_pts else 0.04

            poly = Poly3DCollection(
                faces,
                facecolors=color,
                edgecolors=edge_c,
                linewidths=edge_w,
                alpha=face_a
            )
            ax.add_collection3d(poly)

            # ---- Punti ----
            if has_pts:
                ax.scatter(
                    points[indices, i1],
                    points[indices, i2],
                    points[indices, i3],
                    color=color,
                    s=70,
                    edgecolors='white',
                    linewidth=0.8,
                    alpha=1.0,
                    label=f"Box {j_domain} ({len(indices)} pts)"
                )

        ax.set_xlabel(f"Dim {i1+1}", fontweight='bold')
        ax.set_ylabel(f"Dim {i2+1}", fontweight='bold')
        ax.set_zlabel(f"Dim {i3+1}", fontweight='bold')

        space = "Position" if i1 == 0 else "Velocity"
        ax.set_title(f"{title_prefix} – {space} space", pad=20)

        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

        ax.grid(True, alpha=0.3)
        plt.tight_layout()


def plot_domain_evolution(listOut_opt: List[List[ADSstate]], t_eval: np.ndarray):
    """
    Plot number of domains evolution over time for OPTIMIZED integrator.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Number of domains over time
    n_domains_opt = [len(states) for states in listOut_opt]
    
    ax.plot(t_eval, n_domains_opt, 'b-o', linewidth=2, markersize=8, label='Optimized - Number of Domains')
    ax.set_xlabel('Time (normalized)', fontsize=12)
    ax.set_ylabel('Number of Domains', fontsize=12)
    ax.set_title('Domain Evolution Over Time - OPTIMIZED', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cr3bp_domain_evolution_optimized.png', dpi=300, bbox_inches='tight')


def compute_expansion_errors(
    sample_points: np.ndarray,
    assignments_opt: List[Dict],
    assignments_std: List[Dict]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute position and velocity errors between optimized and standard expansions.
    
    Args:
        sample_points: Array of sample points to evaluate (N x 6)
        assignments_opt: List of assignment dictionaries from OPTIMIZED method
        assignments_std: List of assignment dictionaries from STANDARD method
    
    Returns:
        error_pos: Array of position errors (%)
        error_vel: Array of velocity errors (%)
    """
    error_tot_pos = []
    error_tot_vel = []
    
    for idx, point in enumerate(sample_points):
        expansion_opt = None
        expansion_std = None
        
        # OPTIMIZED assignment
        for assignment in assignments_opt:
            if np.any(assignment['point_indices'] == idx):
                central_point_opt = assignment["box"]["center"]
                delta_state_opt = point - central_point_opt
                expansion_opt = assignment["DA_map"].eval(delta_state_opt)
                break
        
        # STANDARD assignment
        for assignment in assignments_std:
            if np.any(assignment['point_indices'] == idx):
                central_point_std = assignment["box"]["center"]
                delta_state_std = point - central_point_std
                expansion_std = assignment["DA_map"].eval(delta_state_std)
                break
        
        if expansion_opt is None or expansion_std is None:
            continue
        
        error_pos = (
            np.linalg.norm(expansion_opt[0:3] - expansion_std[0:3])
            / np.linalg.norm(expansion_std[0:3])
            * 100
        )
        error_vel = (
            np.linalg.norm(expansion_opt[3:6] - expansion_std[3:6])
            / np.linalg.norm(expansion_std[3:6])
            * 100
        )
        error_tot_pos.append(error_pos)
        error_tot_vel.append(error_vel)
    
    return np.array(error_tot_pos), np.array(error_tot_vel)


def compute_cumulative_distribution(errors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cumulative distribution function (CDF) of errors.
    
    Args:
        errors: Array of errors
    
    Returns:
        sorted_errors: Errors sorted in ascending order
        cumulative_percentage: Percentage of points below each error threshold
    """
    # Sort errors in ascending order
    sorted_errors = np.sort(errors)
    
    # Compute cumulative percentage (0-100%)
    n_points = len(sorted_errors)
    cumulative_percentage = np.arange(1, n_points + 1) / n_points * 100
    
    return sorted_errors, cumulative_percentage


def print_percentile_summary(error_pos: np.ndarray, error_vel: np.ndarray) -> None:
    """
    Print percentile summary of errors between OPTIMIZED and STANDARD.
    
    Args:
        error_pos: Array of position errors (%)
        error_vel: Array of velocity errors (%)
    """
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    
    print("\n" + "="*70)
    print("ERROR PERCENTILE SUMMARY (OPTIMIZED vs STANDARD)")
    print("="*70)
    
    print("\nPOSITION ERRORS:")
    print(f"{'Percentile':<15} {'Error Threshold (%)':<25} {'Interpretation'}")
    print("-" * 70)
    for pct in percentiles:
        threshold = np.percentile(error_pos, pct)
        print(f"{pct}th{'':<12} {threshold:<25.6f} {pct}% of points have error < {threshold:.6f}%")
    
    print("\nVELOCITY ERRORS:")
    print(f"{'Percentile':<15} {'Error Threshold (%)':<25} {'Interpretation'}")
    print("-" * 70)
    for pct in percentiles:
        threshold = np.percentile(error_vel, pct)
        print(f"{pct}th{'':<12} {threshold:<25.6f} {pct}% of points have error < {threshold:.6f}%")
    
    print("="*70 + "\n")


def main():
    """
    CR3BP propagation with ADS:
    - STANDARD vs OPTIMIZED integrator comparison
    - domain evolution in time
    - 3D visualization of ADS boxes (position & velocity)
    - analysis at final time to inspect domain deformation
    """

    # ======================================================================
    # 1. DA & PROBLEM SETUP
    # ======================================================================
    n_order_DA = 4
    state_dim = 6
    DA.init(n_order_DA, state_dim)
    DA.setEps(1e-16)

    mu = 0.01215  # Earth–Moon

    # L1 halo-like initial condition
    L1_x = 0.83691513
    x0, y0, z0 = L1_x + 0.01, 0.0, 0.04
    vx0, vy0, vz0 = 0.0, 0.1, 0.0

    # Uncertainties
    dx, dy, dz = 5e-6, 1e-6, 1e-6
    dvx, dvy, dvz = 1e-7, 1e-7, 1e-7

    # DA initial state
    XI = array.zeros(6)
    XI[0] = x0  + dx  * DA(1)
    XI[1] = y0  + dy  * DA(2)
    XI[2] = z0  + dz  * DA(3)
    XI[3] = vx0 + dvx * DA(4)
    XI[4] = vy0 + dvy * DA(5)
    XI[5] = vz0 + dvz * DA(6)

    init_domains = [ADS(XI, [])]

    # ======================================================================
    # 2. TIME & ADS PARAMETERS
    # ======================================================================
    T0, TF = 0.0, 3.75
    t_eval = np.linspace(T0, TF, 20)

    toll = 1e-9
    Nmax = 20

    print("=" * 70)
    print("CR3BP – ADS PROPAGATION COMPARISON")
    print("=" * 70)
    print(f"mu = {mu}")
    print(f"Initial state: [{x0}, {y0}, {z0}, {vx0}, {vy0}, {vz0}]")
    print(f"Uncertainty: position ~1e-6, velocity ~1e-7")
    print(f"Time span: {T0} → {TF}")
    print("=" * 70)

    # ======================================================================
    # 3. STANDARD ADS PROPAGATION
    # ======================================================================
    print("\n[1/2] STANDARD ADS integrator")

    t0_std = time.time()
    prop_std = AutomaticADS_CR3BP_integrator(mu, RK.RK78())
    prop_std.loadTime(T0, TF)
    prop_std.loadTol(1e-16, 1e-16)
    prop_std.loadStepSize()
    prop_std.loadADSopt(toll, Nmax)

    listOut_std = prop_std.propagate(init_domains, T0, TF)
    t_elapsed_std = time.time() - t0_std

    n_domains_std = len(listOut_std)
    
    print(f"✓ STANDARD - Time: {t_elapsed_std:.3f} s")
    print(f"✓ STANDARD - Final domains: {n_domains_std}")

    # ======================================================================
    # 4. OPTIMIZED ADS PROPAGATION
    # ======================================================================
    print("\n[2/2] OPTIMIZED ADS integrator")

    t0_opt = time.time()
    prop_opt = AutomaticADS_CR3BP_integrator_optimized(mu, RK.RK78())
    prop_opt.loadTime(T0, TF)
    prop_opt.loadTol(1e-16, 1e-16)
    prop_opt.loadStepSize()
    prop_opt.loadADSopt(toll, Nmax)

    listOut_opt = prop_opt.propagate(init_domains, t_eval)
    t_elapsed_opt = time.time() - t0_opt

    n_domains_opt = len(listOut_opt[-1])
    
    print(f"✓ OPTIMIZED - Time: {t_elapsed_opt:.3f} s")
    print(f"✓ OPTIMIZED - Final domains: {n_domains_opt}")
    print(f"✓ Speedup: {t_elapsed_std / t_elapsed_opt:.2f}x")

    # ======================================================================
    # 5. DOMAIN EVOLUTION (OPTIMIZED)
    # ======================================================================
    print("\nOPTIMIZED - Domain evolution:")
    for t, states in zip(t_eval, listOut_opt):
        print(f"  t = {t:6.3f} → {len(states):4d} domains")

    plot_domain_evolution(listOut_opt, t_eval)

    # ======================================================================
    # 6. SAMPLE POINTS FOR FINAL ANALYSIS
    # ======================================================================
    n_samples = 200
    sample_points = np.random.uniform(
        low=[x0-dx, y0-dy, z0-dz, vx0-dvx, vy0-dvy, vz0-dvz],
        high=[x0+dx, y0+dy, z0+dz, vx0+dvx, vy0+dvy, vz0+dvz],
        size=(n_samples, 6)
    )

    # ======================================================================
    # 7. FINAL TIME ANALYSIS - OPTIMIZED
    # ======================================================================
    print("\n" + "=" * 70)
    print("FINAL TIME DOMAIN ANALYSIS - OPTIMIZED")
    print("=" * 70)

    final_states_opt = listOut_opt[-1]

    assignments_opt = ADS_utils.assign_points_to_domains(
        final_states_opt,
        sample_points,
        order_DA=n_order_DA
    )

    viz_data_opt = ADS_utils.prepare_visualization_data(
        final_states_opt,
        sample_points,
        assignments_opt
    )

    print("\nOPTIMIZED - Assignment Statistics:")
    ADS_utils.print_assignment_statistics(viz_data_opt)
    print("\nOPTIMIZED - Validation Report:")
    ADS_utils.print_validation_report(viz_data_opt)

    # ======================================================================
    # 8. FINAL TIME ANALYSIS - STANDARD
    # ======================================================================
    print("\n" + "=" * 70)
    print("FINAL TIME DOMAIN ANALYSIS - STANDARD")
    print("=" * 70)

    final_states_std = listOut_std

    assignments_std = ADS_utils.assign_points_to_domains(
        final_states_std,
        sample_points,
        order_DA=n_order_DA
    )

    viz_data_std = ADS_utils.prepare_visualization_data(
        final_states_std,
        sample_points,
        assignments_std
    )

    print("\nSTANDARD - Assignment Statistics:")
    ADS_utils.print_assignment_statistics(viz_data_std)
    print("\nSTANDARD - Validation Report:")
    ADS_utils.print_validation_report(viz_data_std)

    # ======================================================================
    # 9. 3D VISUALIZATION – POSITION & VELOCITY
    # ======================================================================
    print("\n" + "=" * 70)
    print("GENERATING 3D VISUALIZATIONS")
    print("=" * 70)
    
    print("\nGenerating OPTIMIZED 3D ADS box plots (final time)...")
    plot_ADS_boxes_3D(
        domain_boxes=viz_data_opt["domain_boxes"],
        points=viz_data_opt["points"],
        point_to_domain=viz_data_opt["point_to_domain"],
        title_prefix="ADS domains at final time - OPTIMIZED"
    )

    print("Generating STANDARD 3D ADS box plots (final time)...")
    plot_ADS_boxes_3D(
        domain_boxes=viz_data_std["domain_boxes"],
        points=viz_data_std["points"],
        point_to_domain=viz_data_std["point_to_domain"],
        title_prefix="ADS domains at final time - STANDARD"
    )

    # ======================================================================
    # 10. ERROR ANALYSIS: OPTIMIZED VS STANDARD
    # ======================================================================
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS: OPTIMIZED vs STANDARD")
    print("=" * 70)
    
    # Compute errors
    error_pos, error_vel = compute_expansion_errors(
        sample_points=sample_points,
        assignments_opt=assignments_opt,
        assignments_std=assignments_std
    )
    
    # Print percentile summary
    print_percentile_summary(error_pos, error_vel)
    
    # ======================================================================
    # 11. TAYLOR MAP EXTRACTION
    # ======================================================================
    print("\n" + "=" * 70)
    print("TAYLOR MAP EXTRACTION")
    print("=" * 70)
    
    print("\nExtracting Taylor maps - OPTIMIZED...")
    taylor_maps_opt = ADS_utils.extract_all_taylor_maps(final_states_opt, order_DA=n_order_DA)
    print(f"✓ OPTIMIZED - Extracted {len(taylor_maps_opt)} Taylor maps")
    
    print("\nExtracting Taylor maps - STANDARD...")
    taylor_maps_std = ADS_utils.extract_all_taylor_maps(final_states_std, order_DA=n_order_DA)
    print(f"✓ STANDARD - Extracted {len(taylor_maps_std)} Taylor maps")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  STANDARD:  {n_domains_std} domains in {t_elapsed_std:.3f} s")
    print(f"  OPTIMIZED: {n_domains_opt} domains in {t_elapsed_opt:.3f} s")
    print(f"  Speedup:   {t_elapsed_std / t_elapsed_opt:.2f}x")
    print("=" * 70)
    
    plt.show()


if __name__ == "__main__":
    main()



# import daceypy_import_helper  # noqa: F401
# from inspect import getsourcefile
# from pathlib import Path
# from typing import Callable, List, Dict, Tuple
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import numpy as np
# from numpy.typing import NDArray
# from daceypy import ADS, DA, array, RK
# from daceypy import ADSintegrator, ADSintegrator_optimized
# from daceypy import ADS_utils 
# from daceypy._ADSintegrator import ADSstate
# import time
# import itertools


# def CR3BP(x: array, t: float, mu: float) -> array:
#     """
#     Circular Restricted Three Body Problem dynamics in normalized coordinates.
    
#     Args:
#         x: State vector [x, y, z, vx, vy, vz]
#         t: Time (unused in autonomous system)
#         mu: Mass parameter (Earth-Moon: ~0.01215)
    
#     Returns:
#         State derivative [vx, vy, vz, ax, ay, az]
#     """
#     # Extract position and velocity
#     pos: array = x[:3]
#     vel: array = x[3:]
    
#     x_pos, y_pos, z_pos = pos[0], pos[1], pos[2]
    
#     # Distances from the two primaries
#     r1 = ((x_pos + mu)**2 + y_pos**2 + z_pos**2)**0.5
#     r2 = ((x_pos - 1 + mu)**2 + y_pos**2 + z_pos**2)**0.5
    
#     # Pseudo-potential gradient
#     Omega_x = x_pos - (1 - mu) * (x_pos + mu) / r1**3 - mu * (x_pos - 1 + mu) / r2**3
#     Omega_y = y_pos - (1 - mu) * y_pos / r1**3 - mu * y_pos / r2**3
#     Omega_z = -(1 - mu) * z_pos / r1**3 - mu * z_pos / r2**3
    
#     # Accelerations (Coriolis + centrifugal + gravitational)
#     ax = 2 * vel[1] + Omega_x
#     ay = -2 * vel[0] + Omega_y
#     az = Omega_z
    
#     acc = array([ax, ay, az])
    
#     # State derivative
#     dx = vel.concat(acc)
#     return dx


# class AutomaticADS_CR3BP_integrator_optimized(ADSintegrator_optimized):
#     """
#     Custom child class of ADSintegrator_optimized for CR3BP dynamics.
#     """
#     def __init__(self, mu: float, RK: RK.RKCoeff = RK.RK78()):
#         super(AutomaticADS_CR3BP_integrator_optimized, self).__init__(RK)
#         self.mu = mu
    
#     def f(self, x, t):
#         return CR3BP(x, t, self.mu)


# class AutomaticADS_CR3BP_integrator(ADSintegrator):
#     """
#     Custom child class of ADSintegrator for CR3BP dynamics.
#     """
#     def __init__(self, mu: float, RK: RK.RKCoeff = RK.RK78()):
#         super(AutomaticADS_CR3BP_integrator, self).__init__(RK)
#         self.mu = mu
    
#     def f(self, x, t):
#         return CR3BP(x, t, self.mu)


# from matplotlib.cm import get_cmap

# def plot_ADS_boxes_3D(
#     domain_boxes,
#     points,
#     point_to_domain,
#     title_prefix="ADS domains",
#     ndim=6
# ):
#     """
#     Plot 3D delle box ADS e dei punti associati.
#     Separato per posizione (0,1,2) e velocità (3,4,5).
#     """

#     cmap = get_cmap("tab10")
#     dim_sets = [(0,1,2), (3,4,5)] if ndim >= 6 else [(0,1,2)]

#     for dims in dim_sets:
#         if dims[2] >= ndim:
#             continue

#         fig = plt.figure(figsize=(12, 9))
#         ax = fig.add_subplot(111, projection="3d")
#         i1, i2, i3 = dims

#         for j_domain, d in enumerate(domain_boxes):
#             indices = np.where(point_to_domain == j_domain)[0]
#             has_pts = indices.size > 0
#             color = cmap(j_domain % 10)

#             # ---- Box ----
#             min_v, max_v = d["min"], d["max"]
#             v = np.array([
#                 [min_v[i1], min_v[i2], min_v[i3]],
#                 [max_v[i1], min_v[i2], min_v[i3]],
#                 [max_v[i1], max_v[i2], min_v[i3]],
#                 [min_v[i1], max_v[i2], min_v[i3]],
#                 [min_v[i1], min_v[i2], max_v[i3]],
#                 [max_v[i1], min_v[i2], max_v[i3]],
#                 [max_v[i1], max_v[i2], max_v[i3]],
#                 [min_v[i1], max_v[i2], max_v[i3]],
#             ])

#             faces = [
#                 [v[0],v[1],v[2],v[3]],
#                 [v[4],v[5],v[6],v[7]],
#                 [v[0],v[1],v[5],v[4]],
#                 [v[2],v[3],v[7],v[6]],
#                 [v[1],v[2],v[6],v[5]],
#                 [v[4],v[7],v[3],v[0]]
#             ]

#             edge_w = 2.0 if has_pts else 0.5
#             edge_c = 'black' if has_pts else color
#             face_a = 0.18 if has_pts else 0.04

#             poly = Poly3DCollection(
#                 faces,
#                 facecolors=color,
#                 edgecolors=edge_c,
#                 linewidths=edge_w,
#                 alpha=face_a
#             )
#             ax.add_collection3d(poly)

#             # ---- Punti ----
#             if has_pts:
#                 ax.scatter(
#                     points[indices, i1],
#                     points[indices, i2],
#                     points[indices, i3],
#                     color=color,
#                     s=70,
#                     edgecolors='white',
#                     linewidth=0.8,
#                     alpha=1.0,
#                     label=f"Box {j_domain} ({len(indices)} pts)"
#                 )

#         ax.set_xlabel(f"Dim {i1+1}", fontweight='bold')
#         ax.set_ylabel(f"Dim {i2+1}", fontweight='bold')
#         ax.set_zlabel(f"Dim {i3+1}", fontweight='bold')

#         space = "Position" if i1 == 0 else "Velocity"
#         ax.set_title(f"{title_prefix} – {space} space", pad=20)

#         if ax.get_legend_handles_labels()[0]:
#             ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

#         ax.grid(True, alpha=0.3)
#         plt.tight_layout()



# def plot_domain_evolution(listOut: List[List[ADSstate]], t_eval: np.ndarray):
#     """
#     Plot number of domains evolution over time.
#     """
#     fig = plt.figure(figsize=(10, 6))
#     ax = plt.gca()
    
#     # Number of domains over time (no averaging, just count)
#     n_domains = [len(states) for states in listOut]
    
#     ax.plot(t_eval, n_domains, 'b-o', linewidth=2, markersize=8, label='Number of Domains')
#     ax.set_xlabel('Time (normalized)', fontsize=12)
#     ax.set_ylabel('Number of Domains', fontsize=12)
#     ax.set_title('Domain Evolution Over Time', fontsize=14)
#     ax.legend(fontsize=11)
#     ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('cr3bp_domain_evolution.png', dpi=300, bbox_inches='tight')

# def compute_expansion_errors(
#     sample_points: np.ndarray,
#     assignments: List[Dict],
#     assignments_std: List[Dict]
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Compute position and velocity errors between optimized and standard expansions.
    
#     Args:
#         sample_points: Array of sample points to evaluate (N x 6)
#         assignments: List of assignment dictionaries from optimized method
#         assignments_std: List of assignment dictionaries from standard method
    
#     Returns:
#         error_pos: Array of position errors (%)
#         error_vel: Array of velocity errors (%)
#     """
#     error_tot_pos = []
#     error_tot_vel = []
    
#     for idx, point in enumerate(sample_points):
#         expansion_optimized = None
#         expansion_standard = None
        
#         # Optimized assignment
#         for assignment in assignments:
#             if np.any(assignment['point_indices'] == idx):
#                 central_point = assignment["box"]["center"]
#                 delta_state = point - central_point
#                 expansion_optimized = assignment["DA_map"].eval(delta_state)
#                 break
        
#         # Standard assignment
#         for assignment in assignments_std:
#             if np.any(assignment['point_indices'] == idx):
#                 central_point_std = assignment["box"]["center"]
#                 delta_state_std = point - central_point_std
#                 expansion_standard = assignment["DA_map"].eval(delta_state_std)
#                 break
        
#         if expansion_optimized is None or expansion_standard is None:
#             continue
        
#         error_pos = (
#             np.linalg.norm(expansion_optimized[0:3] - expansion_standard[0:3])
#             / np.linalg.norm(expansion_standard[0:3])
#             * 100
#         )
#         error_vel = (
#             np.linalg.norm(expansion_optimized[3:6] - expansion_standard[3:6])
#             / np.linalg.norm(expansion_standard[3:6])
#             * 100
#         )
#         error_tot_pos.append(error_pos)
#         error_tot_vel.append(error_vel)
    
#     return np.array(error_tot_pos), np.array(error_tot_vel)


# def compute_cumulative_distribution(errors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Compute cumulative distribution function (CDF) of errors.
    
#     Args:
#         errors: Array of errors
    
#     Returns:
#         sorted_errors: Errors sorted in ascending order
#         cumulative_percentage: Percentage of points below each error threshold
#     """
#     # Sort errors in ascending order
#     sorted_errors = np.sort(errors)
    
#     # Compute cumulative percentage (0-100%)
#     n_points = len(sorted_errors)
#     cumulative_percentage = np.arange(1, n_points + 1) / n_points * 100
    
#     return sorted_errors, cumulative_percentage


# def plot_cumulative_error_distribution(
#     error_pos: np.ndarray,
#     error_vel: np.ndarray,
#     save_path: str = None,
#     show_percentiles: bool = True
# ) -> plt.Figure:
#     """
#     Plot cumulative distribution of errors.
    
#     Args:
#         error_pos: Array of position errors (%)
#         error_vel: Array of velocity errors (%)
#         save_path: Optional path to save the figure
#         show_percentiles: Whether to show percentile lines
    
#     Returns:
#         matplotlib Figure object
#     """
#     # Compute cumulative distributions
#     sorted_pos, cum_pct_pos = compute_cumulative_distribution(error_pos)
#     sorted_vel, cum_pct_vel = compute_cumulative_distribution(error_vel)
    
#     # Create figure
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#     fig.suptitle('Cumulative Error Distribution: Optimized vs Standard', 
#                  fontsize=16, fontweight='bold')
    
#     # Define percentile thresholds to highlight
#     percentiles = [50, 90, 95, 99]
#     colors_perc = ['green', 'orange', 'red', 'darkred']
    
#     # 1. Position errors CDF
#     ax = axes[0]
#     ax.plot(sorted_pos, cum_pct_pos, linewidth=2.5, color='#2E86AB', label='CDF')
#     ax.set_xlabel('Position Error (%)', fontsize=12, fontweight='bold')
#     ax.set_ylabel('Cumulative Percentage of Points (%)', fontsize=12, fontweight='bold')
#     ax.set_title('Position Error Distribution', fontsize=14, fontweight='bold')
#     ax.grid(True, alpha=0.3, linestyle='--')
#     ax.set_ylim(0, 100)
    
#     # Add percentile lines
#     if show_percentiles:
#         for pct, color in zip(percentiles, colors_perc):
#             # Find error value at this percentile
#             idx = np.searchsorted(cum_pct_pos, pct)
#             if idx < len(sorted_pos):
#                 error_at_pct = sorted_pos[idx]
#                 ax.axhline(y=pct, color=color, linestyle='--', alpha=0.5, linewidth=1)
#                 ax.axvline(x=error_at_pct, color=color, linestyle='--', alpha=0.5, linewidth=1)
#                 ax.plot(error_at_pct, pct, 'o', color=color, markersize=8, 
#                        label=f'{pct}%: {error_at_pct:.4f}%')
    
#     ax.legend(loc='lower right', fontsize=10)
    
#     # Add statistics box
#     stats_text = (f'Mean: {np.mean(error_pos):.4f}%\n'
#                  f'Median: {np.median(error_pos):.4f}%\n'
#                  f'Max: {np.max(error_pos):.4f}%')
#     ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
#             verticalalignment='top', 
#             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
#             fontsize=10)
    
#     # 2. Velocity errors CDF
#     ax = axes[1]
#     ax.plot(sorted_vel, cum_pct_vel, linewidth=2.5, color='#A23B72', label='CDF')
#     ax.set_xlabel('Velocity Error (%)', fontsize=12, fontweight='bold')
#     ax.set_ylabel('Cumulative Percentage of Points (%)', fontsize=12, fontweight='bold')
#     ax.set_title('Velocity Error Distribution', fontsize=14, fontweight='bold')
#     ax.grid(True, alpha=0.3, linestyle='--')
#     ax.set_ylim(0, 100)
    
#     # Add percentile lines
#     if show_percentiles:
#         for pct, color in zip(percentiles, colors_perc):
#             idx = np.searchsorted(cum_pct_vel, pct)
#             if idx < len(sorted_vel):
#                 error_at_pct = sorted_vel[idx]
#                 ax.axhline(y=pct, color=color, linestyle='--', alpha=0.5, linewidth=1)
#                 ax.axvline(x=error_at_pct, color=color, linestyle='--', alpha=0.5, linewidth=1)
#                 ax.plot(error_at_pct, pct, 'o', color=color, markersize=8,
#                        label=f'{pct}%: {error_at_pct:.4f}%')
    
#     ax.legend(loc='lower right', fontsize=10)
    
#     # Add statistics box
#     stats_text = (f'Mean: {np.mean(error_vel):.4f}%\n'
#                  f'Median: {np.median(error_vel):.4f}%\n'
#                  f'Max: {np.max(error_vel):.4f}%')
#     ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
#             verticalalignment='top',
#             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
#             fontsize=10)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Figure saved to: {save_path}")
    
#     return fig


# def print_percentile_summary(error_pos: np.ndarray, error_vel: np.ndarray) -> None:
#     """
#     Print percentile summary of errors.
    
#     Args:
#         error_pos: Array of position errors (%)
#         error_vel: Array of velocity errors (%)
#     """
#     percentiles = [10, 25, 50, 75, 90, 95, 99]
    
#     print("\n" + "="*70)
#     print("ERROR PERCENTILE SUMMARY")
#     print("="*70)
    
#     print("\nPOSITION ERRORS:")
#     print(f"{'Percentile':<15} {'Error Threshold (%)':<25} {'Interpretation'}")
#     print("-" * 70)
#     for pct in percentiles:
#         threshold = np.percentile(error_pos, pct)
#         print(f"{pct}th{'':<12} {threshold:<25.6f} {pct}% of points have error < {threshold:.6f}%")
    
#     print("\nVELOCITY ERRORS:")
#     print(f"{'Percentile':<15} {'Error Threshold (%)':<25} {'Interpretation'}")
#     print("-" * 70)
#     for pct in percentiles:
#         threshold = np.percentile(error_vel, pct)
#         print(f"{pct}th{'':<12} {threshold:<25.6f} {pct}% of points have error < {threshold:.6f}%")
    
#     print("="*70 + "\n")


# def main():
#     """
#     CR3BP propagation with ADS:
#     - standard vs optimized integrator
#     - domain evolution in time
#     - 3D visualization of ADS boxes (position & velocity)
#     - analysis at final time to inspect domain deformation
#     """

#     # ======================================================================
#     # 1. DA & PROBLEM SETUP
#     # ======================================================================
#     n_order_DA = 4
#     state_dim = 6
#     DA.init(n_order_DA, state_dim)
#     DA.setEps(1e-16)

#     mu = 0.01215  # Earth–Moon

#     # L1 halo-like initial condition
#     L1_x = 0.83691513
#     x0, y0, z0 = L1_x + 0.01, 0.0, 0.04
#     vx0, vy0, vz0 = 0.0, 0.1, 0.0

#     # Uncertainties
#     dx, dy, dz = 5e-6, 1e-6, 1e-6
#     dvx, dvy, dvz = 1e-7, 1e-7, 1e-7

#     # DA initial state
#     XI = array.zeros(6)
#     XI[0] = x0  + dx  * DA(1)
#     XI[1] = y0  + dy  * DA(2)
#     XI[2] = z0  + dz  * DA(3)
#     XI[3] = vx0 + dvx * DA(4)
#     XI[4] = vy0 + dvy * DA(5)
#     XI[5] = vz0 + dvz * DA(6)

#     init_domains = [ADS(XI, [])]

#     # ======================================================================
#     # 2. TIME & ADS PARAMETERS
#     # ======================================================================
#     T0, TF = 0.0, 3.8
#     t_eval = np.linspace(T0, TF, 20)

#     toll = 1e-9
#     Nmax = 20

#     print("=" * 70)
#     print("CR3BP – ADS PROPAGATION")
#     print("=" * 70)
#     print(f"mu = {mu}")
#     print(f"Initial state: [{x0}, {y0}, {z0}, {vx0}, {vy0}, {vz0}]")
#     print(f"Uncertainty: position ~1e-6, velocity ~1e-7")
#     print(f"Time span: {T0} → {TF}")
#     print("=" * 70)

#     # ======================================================================
#     # 3. STANDARD ADS PROPAGATION
#     # ======================================================================
#     # print("\n[1/2] Standard ADS integrator")

#     t0 = time.time()
#     prop_std = AutomaticADS_CR3BP_integrator(mu, RK.RK78())
#     prop_std.loadTime(T0, TF)
#     prop_std.loadTol(1e-16, 1e-16)
#     prop_std.loadStepSize()
#     prop_std.loadADSopt(toll, Nmax)

#     listOut_std = prop_std.propagate(init_domains, T0, TF)
#     t_std = time.time() - t0

#     print(f"✓ Time: {t_std:.3f} s")
#     print(f"✓ Final domains: {len(listOut_std)}")

#     # ======================================================================
#     # 4. OPTIMIZED ADS PROPAGATION
#     # ======================================================================
#     print("\n[2/2] Optimized ADS integrator")

#     t0 = time.time()
#     prop_opt = AutomaticADS_CR3BP_integrator_optimized(mu, RK.RK78())
#     prop_opt.loadTime(T0, TF)
#     prop_opt.loadTol(1e-16, 1e-16)
#     prop_opt.loadStepSize()
#     prop_opt.loadADSopt(toll, Nmax)

#     listOut_opt = prop_opt.propagate(init_domains, t_eval)
#     t_opt = time.time() - t0

#     print(f"✓ Time: {t_opt:.3f} s")
#     print(f"✓ Speedup: {t_std / t_opt:.2f}x")

#     # ======================================================================
#     # 5. DOMAIN EVOLUTION
#     # ======================================================================
#     print("\nDomain evolution:")
#     for t, states in zip(t_eval, listOut_opt):
#         print(f"t = {t:6.3f} → {len(states):4d} domains")

#     plot_domain_evolution(listOut_opt, t_eval)

#     # ======================================================================
#     # 6. SAMPLE POINTS FOR FINAL ANALYSIS
#     # ======================================================================
#     n_samples = 200
#     sample_points = np.random.uniform(
#         low=[x0-dx, y0-dy, z0-dz, vx0-dvx, vy0-dvy, vz0-dvz],
#         high=[x0+dx, y0+dy, z0+dz, vx0+dvx, vy0+dvy, vz0+dvz],
#         size=(n_samples, 6)
#     )

#     # ======================================================================
#     # 7. FINAL TIME ANALYSIS (DEFORMATION)
#     # ======================================================================
#     print("\n" + "=" * 70)
#     print("FINAL TIME DOMAIN ANALYSIS")
#     print("=" * 70)

#     final_states = listOut_opt[-1]

#     assignments = ADS_utils.assign_points_to_domains(
#         final_states,
#         sample_points,
#         order_DA=n_order_DA
#     )

#     viz_data = ADS_utils.prepare_visualization_data(
#         final_states,
#         sample_points,
#         assignments
#     )

#     ADS_utils.print_assignment_statistics(viz_data)
#     ADS_utils.print_validation_report(viz_data)

#     final_states_std = listOut_std

#     assignments_std = ADS_utils.assign_points_to_domains(
#         final_states_std,
#         sample_points,
#         order_DA=n_order_DA
#     )

#     viz_data_std = ADS_utils.prepare_visualization_data(
#         final_states_std,
#         sample_points,
#         assignments_std
#     )

#     ADS_utils.print_assignment_statistics(viz_data_std)
#     ADS_utils.print_validation_report(viz_data_std)

#     # ======================================================================
#     # 8. 3D VISUALIZATION – POSITION & VELOCITY
#     # ======================================================================
#     print("\nGenerating 3D ADS box plots (final time)...")

#     plot_ADS_boxes_3D(
#         domain_boxes=viz_data["domain_boxes"],
#         points=viz_data["points"],
#         point_to_domain=viz_data["point_to_domain"],
#         title_prefix="ADS domains at final time - Optimized"
#     )

#     plot_ADS_boxes_3D(
#         domain_boxes=viz_data_std["domain_boxes"],
#         points=viz_data_std["points"],
#         point_to_domain=viz_data_std["point_to_domain"],
#         title_prefix="ADS domains at final time - Standard"
#     )

#     # ======================================================================
#     # 9. ERROR ANALYSIS OPTIMIZED VS STANDARD
#     # ======================================================================
#     # Compute errors
#     error_pos, error_vel = compute_expansion_errors(
#             sample_points=sample_points,
#             assignments=assignments,
#             assignments_std=assignments_std
#         )
        
#         # Print percentile summary
#     print_percentile_summary(error_pos, error_vel)
        
#         # Create cumulative distribution plot
#     fig = plot_cumulative_error_distribution(
#         error_pos=error_pos,
#         error_vel=error_vel,
#         save_path='error_cdf.png',
#         show_percentiles=True
#     )
    
#     # ======================================================================
#     #  10. TAYLOR MAP EXTRACTION
#     # ======================================================================
#     print("\nExtracting Taylor maps...")
#     taylor_maps = ADS_utils.extract_all_taylor_maps(final_states, order_DA=n_order_DA)
#     print(f"✓ Extracted {len(taylor_maps)} Taylor maps")

#     print("\n" + "=" * 70)
#     print("ANALYSIS COMPLETE")
#     print("=" * 70)
#     plt.show()



if __name__ == "__main__":
    main()