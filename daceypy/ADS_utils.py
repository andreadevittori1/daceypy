import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from daceypy.DA_utils import extract_map
from scipy.spatial import KDTree
import itertools


def extract_all_taylor_maps(
    ADS_domains: Union[List[Any], List[List[Any]]], 
    order_DA: int
) -> Union[List[Dict], List[List[Dict]]]:
    """
    Extract Taylor maps from all ADS domains independently of any points.
    
    Parameters
    ----------
    ADS_domains : list[ADSstate] or list[list[ADSstate]]
        - If list: Single list of ADSstate objects (with ADSPatch attribute)
        - If list of lists: Multiple lists of ADSstate objects (one per time)
    order_DA : int
        Order for extracting the Taylor expansion (DA map)
    
    Returns
    -------
    taylor_maps : list[dict] or list[list[dict]]
        - If single ADS_domains list: Single list of Taylor map dicts
        - If multiple ADS_domains lists: List of lists of Taylor map dicts (one per time)
        
        Each dict contains:
            - 'taylor_map': np.ndarray - extracted DA map
            - 'DA_map': DA object - raw DA manifold
            - 'bounding_box': dict with 'min', 'max', 'center' (all np.ndarray)
            - 'manifold_dim': int - dimension of the manifold
    """
    # Check if input is a list of lists (multiple times) or single list
    is_multiple_times: bool = (
        isinstance(ADS_domains, list) and 
        len(ADS_domains) > 0 and 
        isinstance(ADS_domains[0], list)
    )
    
    # If single list, wrap it to use same logic
    ADS_domains_list: List[List[Any]] = ADS_domains if is_multiple_times else [ADS_domains]
    
    # Process each time step
    all_taylor_maps: List[List[Dict]] = []
    
    for _, ADS_domains_at_time in enumerate(ADS_domains_list):
        # Extract ADS patches from domain objects
        patches: List[Any] = [domain.ADSPatch for domain in ADS_domains_at_time]
        
        taylor_maps_at_time: List[Dict] = []
        for patch in patches:
            # Get dimension from manifold
            ndim: int = len(patch.manifold)
            
            # Compute bounding box in physical space
            min_corner: np.ndarray = np.array(patch.box.eval([-1] * ndim)).reshape(ndim)
            max_corner: np.ndarray = np.array(patch.box.eval([+1] * ndim)).reshape(ndim)
            center: np.ndarray = 0.5 * (min_corner + max_corner)
            
            # Extract Taylor map
            taylor_map: np.ndarray = extract_map(patch.manifold, order_DA)

            taylor_maps_at_time.append({
                'Taylor_map': taylor_map,
                'DA_map': patch.manifold,
                'bounding_box': {
                    'min': min_corner,
                    'max': max_corner,
                    'center': center
                },
                'manifold_dim': ndim
            })
        
        all_taylor_maps.append(taylor_maps_at_time)
    
    # Return single list if input was single list, otherwise return list of lists
    return all_taylor_maps[0] if not is_multiple_times else all_taylor_maps


"""
Improved assign_points_to_domains
──────────────────────────────────
Each assignment dict now carries two extra sections keyed with the
prefix  "basis_"  to make the local-coordinate data easy to spot:

    "basis_box"    : {"min", "max", "center"}
                     The axis-aligned bounds of the patch expressed in
                     parametric (u) space.  By construction the patch is
                     defined on  u ∈ [-1, 1]^n, so these are always
                     (-1…-1), (+1…+1), (0…0) – stored explicitly so
                     downstream code can treat physical and basis dicts
                     symmetrically.

    "basis_points" : np.ndarray, shape (n_assigned, ndim)
                     The assigned points converted to local coordinates:
                         u_i = Axes^{-1} @ (x_i - origin)
                     All entries satisfy |u_ij| ≤ 1 (up to the membership
                     tolerance), confirming correct assignment.
"""


def assign_points_to_domains(
    ADS_domains: Union[List[Any], List[List[Any]]],
    points: np.ndarray,
    order_DA: Optional[int] = None,
    basis: Optional[np.ndarray] = None,          # kept for API compatibility
) -> Union[List[Dict], List[List[Dict]]]:
    """
    Assign points to ADS sub-domains using an exact membership test.

    Physical-space parametrisation of each patch
    ─────────────────────────────────────────────
        x = origin + Axes @ u,   u ∈ [-1, 1]^n

    where
        origin       = patch.box.eval([0, 0, …, 0])
        Axes[:, k]   = patch.box.eval(e_k) - origin   (k-th column)

    A point x0 belongs to patch d iff ALL components of
        u0 = Axes_d^{-1} @ (x0 - origin_d)
    satisfy |u0_k| ≤ 1  (with a small tolerance).

    Parameters
    ──────────
    ADS_domains : list[ADSstate] or list[list[ADSstate]]
    points      : array-like, shape (N, ndim)
    order_DA    : int, optional
        When provided, Taylor maps are extracted and stored under
        "Taylor_map" in each assignment dict.
    basis       : np.ndarray, shape (ndim, ndim), optional
        Stored verbatim under "basis" in each assignment dict for
        downstream use (e.g. compute_stm_stt_ADS).  Does NOT influence
        domain assignment.

    Returns
    ───────
    list[dict] or list[list[dict]]

    Each dict contains
    ──────────────────
    "domain_index"  : int
    "point_indices" : np.ndarray[int]   – indices into *points*
    "n_points"      : int
    "DA_map"        : DA manifold object
    "box"           : {"min", "max", "center"}  – in physical space
    "patch_origin"  : np.ndarray, shape (ndim,)
    "patch_axes"    : np.ndarray, shape (ndim, ndim)  – columns = axis vecs
    "basis_box"     : {"min", "max", "center"}  – in local (u) space
                      always (-1…), (+1…), (0…) but explicit for symmetry
    "basis_points"  : np.ndarray, shape (n_assigned, ndim)
                      assigned points expressed in u-coordinates
    "Taylor_map"    : dict  (only when order_DA is not None)
    "basis"         : np.ndarray  (only when basis is not None)
    """

    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim == 1:
        points_array = points_array.reshape(1, -1)
    n_points, ndim = points_array.shape

    # ------------------------------------------------------------------ #
    # Normalise input to always be a list-of-lists                        #
    # ------------------------------------------------------------------ #
    is_multiple_times: bool = (
        isinstance(ADS_domains, list)
        and len(ADS_domains) > 0
        and isinstance(ADS_domains[0], list)
    )
    domains_list = ADS_domains if is_multiple_times else [ADS_domains]

    # Constant basis-space box – the same for every patch by definition
    basis_box_min    = np.full(ndim, -1.0)
    basis_box_max    = np.full(ndim,  1.0)
    basis_box_center = np.zeros(ndim)

    all_point_assignments: List[List[Dict]] = []

    # ================================================================== #
    #  Main loop over time steps                                          #
    # ================================================================== #
    for ADS_domains_at_time in domains_list:

        patches   = [d.ADSPatch for d in ADS_domains_at_time]
        n_domains = len(patches)

        # -------------------------------------------------------------- #
        # 1.  Compute patch geometry                                      #
        # -------------------------------------------------------------- #
        patch_origins = np.zeros((n_domains, ndim))
        patch_axes    = np.zeros((n_domains, ndim, ndim))
        patch_axes_inv = np.zeros((n_domains, ndim, ndim))   # pre-computed inverse

        phys_mins    = np.zeros((n_domains, ndim))
        phys_maxs    = np.zeros((n_domains, ndim))
        phys_centers = np.zeros((n_domains, ndim))

        for i, patch in enumerate(patches):
            origin = np.asarray(patch.box.eval([0] * ndim), dtype=float).ravel()
            patch_origins[i] = origin

            for k in range(ndim):
                ek = [1 if j == k else 0 for j in range(ndim)]
                patch_axes[i, :, k] = (
                    np.asarray(patch.box.eval(ek), dtype=float).ravel() - origin
                )

            # Pre-invert (used repeatedly in the membership tests and for
            # computing basis_points later – pay the cost once per patch)
            try:
                patch_axes_inv[i] = np.linalg.inv(patch_axes[i])
            except np.linalg.LinAlgError:
                patch_axes_inv[i] = np.full((ndim, ndim), np.nan)

            # Axis-aligned bounding box from all 2^n corners
            corners = np.array([
                np.asarray(patch.box.eval(list(s)), dtype=float).ravel()
                for s in itertools.product([-1, 1], repeat=ndim)
            ])
            phys_mins[i]    = corners.min(axis=0)
            phys_maxs[i]    = corners.max(axis=0)
            phys_centers[i] = 0.5 * (phys_mins[i] + phys_maxs[i])

        tree = KDTree(phys_centers)

        # -------------------------------------------------------------- #
        # 2.  Assign each point to its patch                             #
        # -------------------------------------------------------------- #
        MEMBERSHIP_TOL = 1e-10

        point_to_domain = np.full(n_points, -1, dtype=np.int64)

        # Vectorised AABB pre-filter  (n_points × n_domains)
        inside_aabb = np.all(
            (points_array[:, np.newaxis, :] >= phys_mins[np.newaxis]) &
            (points_array[:, np.newaxis, :] <= phys_maxs[np.newaxis]),
            axis=2,
        )

        def _exact_u(d: int, x0: np.ndarray) -> Optional[np.ndarray]:
            """Return u = Axes^{-1}(x0 - origin), or None if singular."""
            Ainv = patch_axes_inv[d]
            if np.any(np.isnan(Ainv)):
                return None
            return Ainv @ (x0 - patch_origins[d])

        def _in_patch(d: int, x0: np.ndarray) -> bool:
            u = _exact_u(d, x0)
            return (u is not None) and np.all(np.abs(u) <= 1.0 + MEMBERSHIP_TOL)

        for i in range(n_points):
            x0 = points_array[i]
            candidates = np.where(inside_aabb[i])[0]

            exact_inside = [d for d in candidates if _in_patch(d, x0)]

            if len(exact_inside) == 1:
                point_to_domain[i] = exact_inside[0]

            elif len(exact_inside) > 1:
                # Rare overlap – pick the patch with smallest |u|
                u_norms = []
                for d in exact_inside:
                    u = _exact_u(d, x0)
                    u_norms.append(np.linalg.norm(u) if u is not None else np.inf)
                point_to_domain[i] = exact_inside[int(np.argmin(u_norms))]

            else:
                # AABB pre-filter missed – try ALL patches exactly
                truly_inside = [d for d in range(n_domains) if _in_patch(d, x0)]

                if len(truly_inside) == 1:
                    point_to_domain[i] = truly_inside[0]
                elif len(truly_inside) > 1:
                    u_norms = []
                    for d in truly_inside:
                        u = _exact_u(d, x0)
                        u_norms.append(np.linalg.norm(u) if u is not None else np.inf)
                    point_to_domain[i] = truly_inside[int(np.argmin(u_norms))]
                else:
                    # Hard fallback: nearest physical centre
                    _, nearest = tree.query(x0)
                    point_to_domain[i] = nearest

        # -------------------------------------------------------------- #
        # 3.  Build per-domain assignment dicts                          #
        # -------------------------------------------------------------- #
        point_assignments: List[Dict] = []

        for d in range(n_domains):
            idx = np.where(point_to_domain == d)[0]
            if idx.size == 0:
                continue

            phys_pts = points_array[idx]          # (n_assigned, ndim)

            # ---- basis_points: transform assigned points → u-space ---- #
            # u_i = Axes_d^{-1} @ (x_i - origin_d)
            # shape: (n_assigned, ndim)
            basis_pts = (patch_axes_inv[d] @ (phys_pts - patch_origins[d]).T).T

            entry: Dict = {
                # ── physical-space geometry ──────────────────────────── #
                "domain_index":  d,
                "point_indices": idx,
                "n_points":      len(idx),
                "DA_map":        patches[d].manifold,
                "box": {
                    "min":    phys_mins[d],
                    "max":    phys_maxs[d],
                    "center": phys_centers[d],
                },
                "patch_origin": patch_origins[d],
                "patch_axes":   patch_axes[d],     # columns = axis vectors

                # ── basis / local (u) space ──────────────────────────── #
                # "basis_box" describes the parametric extent of the patch.
                # By definition this is always [-1,1]^n; stored explicitly
                # so callers can use "basis_box" and "box" symmetrically.
                "basis_box": {
                    "min":    basis_box_min.copy(),
                    "max":    basis_box_max.copy(),
                    "center": basis_box_center.copy(),
                },
                # Each assigned point expressed in local coordinates.
                # All entries should satisfy |u_ij| ≤ 1 + tol.
                "basis_points": basis_pts,
            }

            if order_DA is not None:
                entry["Taylor_map"] = extract_map(patches[d].manifold, order_DA)

            if basis is not None:
                entry["basis"] = basis

            point_assignments.append(entry)

        all_point_assignments.append(point_assignments)

    return all_point_assignments[0] if not is_multiple_times else all_point_assignments

def prepare_visualization_data(
    ADS_domains: List[Any], 
    points: np.ndarray, 
    point_assignments: Optional[List[Dict]] = None
) -> Dict:
    """
    Prepare data structures for visualization and analysis.
    
    Parameters
    ----------
    ADS_domains : list[ADSstate]
        List of ADSstate objects (with ADSPatch attribute)
    points : np.ndarray of shape (N, ndim)
        Points that were assigned to domains
    point_assignments : list[dict], optional
        Output from assign_points_to_domains. If None, points will be assigned automatically.
    
    Returns
    -------
    visualization_data : dict
        Dictionary containing:
            - 'points': np.ndarray - the original points array
            - 'domain_boxes': list[dict] - bounding boxes for each domain
            - 'point_to_domain': np.ndarray - mapping each point to its assigned domain index
            - 'n_domains': int - total number of domains
            - 'n_points': int - total number of points
    """
    # Convert points to numpy array and validate dimensions
    points_array: np.ndarray = np.asarray(points, dtype=np.float64)
    if points_array.ndim == 1:
        points_array = points_array.reshape(-1, 1)
    
    n_points: int
    ndim: int
    n_points, ndim = points_array.shape
    
    # Extract ADS patches from domain objects
    patches: List[Any] = [domain.ADSPatch for domain in ADS_domains]
    n_domains: int = len(patches)
    
    # Compute bounding boxes for all domains
    domain_boxes: List[Dict] = []
    for patch in patches:
        min_corner: np.ndarray = np.array(patch.box.eval([-1] * ndim)).reshape(ndim)
        max_corner: np.ndarray = np.array(patch.box.eval([+1] * ndim)).reshape(ndim)
        
        domain_boxes.append({
            "min": min_corner,
            "max": max_corner,
            "center": 0.5 * (min_corner + max_corner),
            "manifold": patch.manifold
        })
    
    # Build point-to-domain mapping
    point_to_domain: np.ndarray
    
    if point_assignments is None:
        # Compute assignments if not provided
        point_to_domain = np.full(n_points, -1, dtype=np.int64)
        
        for i, point in enumerate(points_array):
            is_inside: List[bool] = [
                np.all((point >= box["min"]) & (point <= box["max"])) 
                for box in domain_boxes
            ]
            
            if any(is_inside):
                candidate_indices: np.ndarray = np.where(is_inside)[0]
                distances: List[float] = [
                    np.linalg.norm(point - domain_boxes[idx]["center"]) 
                    for idx in candidate_indices
                ]
                point_to_domain[i] = candidate_indices[np.argmin(distances)]
            else:
                distances = [
                    np.linalg.norm(point - box["center"]) 
                    for box in domain_boxes
                ]
                point_to_domain[i] = np.argmin(distances)
    else:
        # Reconstruct from assignments
        point_to_domain = np.full(n_points, -1, dtype=np.int64)
        for assignment in point_assignments:
            domain_idx: int = assignment['domain_index']
            point_indices: np.ndarray = assignment['point_indices']
            point_to_domain[point_indices] = domain_idx
    
    # Prepare visualization data
    visualization_data: Dict = {
        "points": points_array,
        "domain_boxes": domain_boxes,
        "point_to_domain": point_to_domain,
        "n_domains": n_domains,
        "n_points": n_points
    }
    
    return visualization_data


def compute_assignment_statistics(visualization_data: Dict) -> Dict:
    """
    Calculate statistics about point-to-domain assignments.
    
    Parameters
    ----------
    visualization_data : dict
        Output from prepare_visualization_data
    
    Returns
    -------
    stats : dict
        Dictionary containing:
            - 'n_domains': int - total number of domains
            - 'n_points': int - total number of points
            - 'points_per_domain': np.ndarray - point counts per domain
            - 'empty_domains': int - number of domains with no assigned points
            - 'occupied_domains': int - number of domains with at least one point
            - 'mean_points_per_occupied_domain': float - average points per non-empty domain
            - 'min_points_per_occupied_domain': int - minimum points in non-empty domains
            - 'max_points_per_domain': int - maximum points in any domain
            - 'coverage_ratio': float - fraction of domains with assigned points
    """
    point_to_domain: np.ndarray = visualization_data['point_to_domain']
    n_domains: int = visualization_data['n_domains']
    n_points: int = visualization_data['n_points']
    
    # Count points per domain
    points_per_domain: np.ndarray = np.array([
        np.sum(point_to_domain == domain_idx) 
        for domain_idx in range(n_domains)
    ], dtype=np.int64)
    
    # Calculate statistics
    empty_domains: int = int(np.sum(points_per_domain == 0))
    occupied_domains: int = n_domains - empty_domains
    
    occupied_counts: np.ndarray = points_per_domain[points_per_domain > 0]
    mean_points_occupied: float = float(np.mean(occupied_counts)) if occupied_domains > 0 else 0.0
    min_points_occupied: int = int(np.min(occupied_counts)) if occupied_domains > 0 else 0
    max_points: int = int(np.max(points_per_domain))
    
    stats: Dict = {
        'n_domains': n_domains,
        'n_points': n_points,
        'points_per_domain': points_per_domain,
        'empty_domains': empty_domains,
        'occupied_domains': occupied_domains,
        'mean_points_per_occupied_domain': mean_points_occupied,
        'min_points_per_occupied_domain': min_points_occupied,
        'max_points_per_domain': max_points,
        'coverage_ratio': float(occupied_domains) / float(n_domains) if n_domains > 0 else 0.0
    }
    
    return stats


def print_assignment_statistics(visualization_data: Dict) -> None:
    """
    Print formatted statistics about point-to-domain assignments.
    
    Parameters
    ----------
    visualization_data : dict
        Output from prepare_visualization_data
    
    Returns
    -------
    None
    """
    stats: Dict = compute_assignment_statistics(visualization_data)
    
    print("\n" + "="*70)
    print("Point-to-Domain Assignment Statistics")
    print("="*70)
    print(f"Total domains:                    {stats['n_domains']:6d}")
    print(f"Total points:                     {stats['n_points']:6d}")
    print(f"Empty domains:                    {stats['empty_domains']:6d}")
    print(f"Occupied domains:                 {stats['occupied_domains']:6d}")
    print(f"Coverage ratio:                   {stats['coverage_ratio']:6.2%}")
    print(f"Mean points/occupied domain:      {stats['mean_points_per_occupied_domain']:6.2f}")
    print(f"Min points (occupied domains):    {stats['min_points_per_occupied_domain']:6d}")
    print(f"Max points (any domain):          {stats['max_points_per_domain']:6d}")
    print("="*70)
    
    # Show distribution histogram
    points_per_domain: np.ndarray = stats['points_per_domain']
    unique_counts: np.ndarray
    count_frequencies: np.ndarray
    unique_counts, count_frequencies = np.unique(points_per_domain, return_counts=True)
    
    print("\nDistribution of points per domain:")
    print("-" * 70)
    max_freq: int = int(max(count_frequencies))
    for count, freq in zip(unique_counts, count_frequencies):
        bar: str = '█' * int(freq / max_freq * 40)
        if count == 0:
            print(f"  {freq:4d} domains with    0 points  {bar}")
        else:
            print(f"  {freq:4d} domains with {count:4d} points  {bar}")
    print("="*70 + "\n")


def validate_point_assignments(
    visualization_data: Dict, 
    tolerance: float = 1e-10
) -> Dict:
    """
    Validate that all points are properly assigned to domains.
    
    Parameters
    ----------
    visualization_data : dict
        Output from prepare_visualization_data
    tolerance : float, default=1e-10
        Tolerance for checking if point is inside domain bounding box
    
    Returns
    -------
    validation : dict
        Dictionary containing:
            - 'all_assigned': bool - True if all points have valid domain assignments
            - 'n_unassigned': int - number of points with invalid assignments
            - 'n_outside_box': int - number of points outside their assigned domain box
            - 'total_points': int - total number of points
            - 'is_valid': bool - overall validation status
    """
    points: np.ndarray = visualization_data['points']
    point_to_domain: np.ndarray = visualization_data['point_to_domain']
    domain_boxes: List[Dict] = visualization_data['domain_boxes']
    
    n_unassigned: int = int(np.sum(point_to_domain == -1))
    n_outside_box: int = 0
    
    for i, (point, domain_idx) in enumerate(zip(points, point_to_domain)):
        if 0 <= domain_idx < len(domain_boxes):
            box: Dict = domain_boxes[domain_idx]
            # Check if point is inside assigned domain box (with tolerance)
            is_inside: bool = bool(np.all(
                (point >= box["min"] - tolerance) & 
                (point <= box["max"] + tolerance)
            ))
            if not is_inside:
                n_outside_box += 1
    
    validation: Dict = {
        'all_assigned': n_unassigned == 0,
        'n_unassigned': n_unassigned,
        'n_outside_box': n_outside_box,
        'total_points': len(points),
        'is_valid': (n_unassigned == 0) and (n_outside_box == 0)
    }
    
    return validation


def print_validation_report(
    visualization_data: Dict, 
    tolerance: float = 1e-10
) -> None:
    """
    Print a validation report for point-to-domain assignments.
    
    Parameters
    ----------
    visualization_data : dict
        Output from prepare_visualization_data
    tolerance : float, default=1e-10
        Tolerance for validation checks
    
    Returns
    -------
    None
    """
    validation: Dict = validate_point_assignments(visualization_data, tolerance)
    
    print("\n" + "="*70)
    print("Assignment Validation Report")
    print("="*70)
    print(f"Total points:                     {validation['total_points']:6d}")
    print(f"Unassigned points:                {validation['n_unassigned']:6d}")
    print(f"Points outside assigned box:      {validation['n_outside_box']:6d}")
    print(f"All points assigned:              {'✓ Yes' if validation['all_assigned'] else '✗ No'}")
    print(f"Overall validation:               {'✓ PASS' if validation['is_valid'] else '✗ FAIL'}")
    print("="*70 + "\n")


# Example usage and testing
if __name__ == "__main__":
    print("ADS Point Assignment Utility Functions")
    print("="*70)
    print("\nAvailable functions:")
    print("\n1. Core functions:")
    print("   - extract_all_taylor_maps: Extract all Taylor maps independently")
    print("   - assign_points_to_domains: Assign points and extract Taylor maps")
    print("   - prepare_visualization_data: Prepare data for plotting/analysis")
    print("\n2. Analysis functions:")
    print("   - compute_assignment_statistics: Calculate assignment statistics")
    print("   - print_assignment_statistics: Print formatted statistics")
    print("   - validate_point_assignments: Validate assignments")
    print("   - print_validation_report: Print validation report")
    print("\n3. Usage patterns:")
    print("\n   Pattern A: Assignment only (no visualization)")
    print("   >>> assignments = assign_points_to_domains(domains, points, order_DA=2)")
    print("\n   Pattern B: Assignment + visualization")
    print("   >>> assignments = assign_points_to_domains(domains, points, order_DA=2)")
    print("   >>> viz_data = prepare_visualization_data(domains, points, assignments)")
    print("   >>> print_assignment_statistics(viz_data)")
    print("\n   Pattern C: Visualization without pre-computed assignments")
    print("   >>> viz_data = prepare_visualization_data(domains, points)")
    print("   >>> print_assignment_statistics(viz_data)")
    print("="*70)
