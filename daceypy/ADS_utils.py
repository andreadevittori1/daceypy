import numpy as np
from daceypy.DA_utils import extract_map

def extract_all_taylor_maps(ADS_domains, order_DA):
    """
    Extract Taylor maps from all ADS domains independently of any points.
    
    Parameters
    ----------
    ADS_domains : list or list of lists
        - If list: Single list of ADSstate objects (with ADSPatch attribute)
        - If list of lists: Multiple lists of ADSstate objects (one per time)
    order_DA : int
        Order for extracting the Taylor expansion (DA map)
    
    Returns
    -------
    taylor_maps : list or list of lists
        - If single ADS_domains list: Single list of Taylor map dicts
        - If multiple ADS_domains lists: List of lists of Taylor map dicts (one per time)
        
        Each dict contains:
            - 'taylor_map': extracted DA map
            - 'DA_map': raw DA manifold
            - 'bounding_box': dict with 'min', 'max', 'center'
            - 'manifold_dim': dimension of the manifold
    """
    # Check if input is a list of lists (multiple times) or single list
    is_multiple_times = (
        isinstance(ADS_domains, list) and 
        len(ADS_domains) > 0 and 
        isinstance(ADS_domains[0], list)
    )
    
    # If single list, wrap it to use same logic
    ADS_domains_list = ADS_domains if is_multiple_times else [ADS_domains]
    
    # Process each time step
    all_taylor_maps = []
    
    for _, ADS_domains_at_time in enumerate(ADS_domains_list):
        # Extract ADS patches from domain objects
        patches = [domain.ADSPatch for domain in ADS_domains_at_time]
        
        taylor_maps_at_time = []
        for patch in patches:
            # Get dimension from manifold
            ndim = len(patch.manifold)
            
            # Compute bounding box in physical space
            min_corner = np.array(patch.box.eval([-1] * ndim)).reshape(ndim)
            max_corner = np.array(patch.box.eval([+1] * ndim)).reshape(ndim)
            center = 0.5 * (min_corner + max_corner)
            
            # Extract Taylor map
            taylor_map = extract_map(patch.manifold, order_DA)

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


def assign_points_to_domains(ADS_domains, points, order_DA=None):
    """
    Assign points to their corresponding ADS domains based on spatial proximity.
    
    Parameters
    ----------
    ADS_domains : list or list of lists
        - If list: Single list of ADSstate objects (with ADSPatch attribute)
        - If list of lists: Multiple lists of ADSstate objects (one per time)
    points : array (N, ndim)
        Points to be assigned to ADS subdomains
    order_DA : int, optional
        Order for extracting the Taylor expansion. If None, Taylor maps are not extracted.
    
    Returns
    -------
    point_assignments : list or list of lists
        - If single ADS_domains list: Single list of assignment dicts
        - If multiple ADS_domains lists: List of lists of assignment dicts (one per time)
        
        Each assignment dict contains:
            - 'domain_index': index of the domain
            - 'DA_map': raw DA manifold
            - 'taylor_map': extracted Taylor map (if order_DA is provided)
            - 'point_indices': indices of points assigned to this domain
            - 'n_points': number of points in this domain
    """
    # Convert points to numpy array and validate dimensions
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    
    n_points, ndim = points.shape
    
    # Check if input is a list of lists (multiple times) or single list
    is_multiple_times = (
        isinstance(ADS_domains, list) and 
        len(ADS_domains) > 0 and 
        isinstance(ADS_domains[0], list)
    )
    
    # If single list, wrap it to use same logic
    ADS_domains_list = ADS_domains if is_multiple_times else [ADS_domains]
    
    # Process each time step
    all_point_assignments = []
    
    for time_idx, ADS_domains_at_time in enumerate(ADS_domains_list):
        # Extract ADS patches from domain objects
        patches = [domain.ADSPatch for domain in ADS_domains_at_time]
        n_domains = len(patches)
        
        # --- 1. Compute bounding boxes for all domains ---
        domain_boxes = []
        for patch in patches:
            # Evaluate corners of the domain in physical space
            min_corner = np.array(patch.box.eval([-1] * ndim)).reshape(ndim)
            max_corner = np.array(patch.box.eval([+1] * ndim)).reshape(ndim)
            
            domain_boxes.append({
                "min": min_corner,
                "max": max_corner,
                "center": 0.5 * (min_corner + max_corner),
                "manifold": patch.manifold
            })
        
        # --- 2. Assign each point to its nearest domain ---
        point_to_domain = np.full(n_points, -1, dtype=int)
        
        for i, point in enumerate(points):
            # Check if point is inside any domain's bounding box
            is_inside = [
                np.all((point >= box["min"]) & (point <= box["max"])) 
                for box in domain_boxes
            ]
            
            if any(is_inside):
                # If inside multiple domains, choose the one with closest center
                candidate_indices = np.where(is_inside)[0]
                distances = [
                    np.linalg.norm(point - domain_boxes[idx]["center"]) 
                    for idx in candidate_indices
                ]
                point_to_domain[i] = candidate_indices[np.argmin(distances)]
            else:
                # If outside all domains, assign to domain with nearest center
                distances = [
                    np.linalg.norm(point - box["center"]) 
                    for box in domain_boxes
                ]
                point_to_domain[i] = np.argmin(distances)
        
        # --- 3. Build point assignments grouped by domain ---
        point_assignments = []
        for domain_idx in range(n_domains):
            # Find indices of points assigned to this domain
            assigned_point_indices = np.where(point_to_domain == domain_idx)[0]
            
            if assigned_point_indices.size > 0:
                assignment = {
                    "domain_index": domain_idx,
                    "point_indices": assigned_point_indices,
                    "n_points": len(assigned_point_indices),
                    "DA_map": domain_boxes[domain_idx]["manifold"],
                    "box": {
                        "min": domain_boxes[domain_idx]["min"],
                        "max": domain_boxes[domain_idx]["max"],
                        "center": domain_boxes[domain_idx]["center"]
                    }
                }
                
                # Extract Taylor map if order is specified
                if order_DA is not None:
                    assignment["Taylor_map"] = extract_map(
                        domain_boxes[domain_idx]["manifold"], 
                        order_DA
                    )
                
                point_assignments.append(assignment)
        
        all_point_assignments.append(point_assignments)
    
    # Return single list if input was single list, otherwise return list of lists
    return all_point_assignments[0] if not is_multiple_times else all_point_assignments


def prepare_visualization_data(ADS_domains, points, point_assignments=None):
    """
    Prepare data structures for visualization and analysis.
    
    Parameters
    ----------
    ADS_domains : list
        List of ADSstate objects (with ADSPatch attribute)
    points : array (N, ndim)
        Points that were assigned to domains
    point_assignments : list of dict, optional
        Output from assign_points_to_domains. If None, points will be assigned automatically.
    
    Returns
    -------
    visualization_data : dict
        Dictionary containing:
            - 'points': the original points array
            - 'domain_boxes': list of bounding boxes for each domain
            - 'point_to_domain': array mapping each point to its assigned domain index
            - 'n_domains': total number of domains
            - 'n_points': total number of points
    """
    # Convert points to numpy array and validate dimensions
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    
    n_points, ndim = points.shape
    
    # Extract ADS patches from domain objects
    patches = [domain.ADSPatch for domain in ADS_domains]
    n_domains = len(patches)
    
    # Compute bounding boxes for all domains
    domain_boxes = []
    for patch in patches:
        min_corner = np.array(patch.box.eval([-1] * ndim)).reshape(ndim)
        max_corner = np.array(patch.box.eval([+1] * ndim)).reshape(ndim)
        
        domain_boxes.append({
            "min": min_corner,
            "max": max_corner,
            "center": 0.5 * (min_corner + max_corner),
            "manifold": patch.manifold
        })
    
    # Build point-to-domain mapping
    if point_assignments is None:
        # Compute assignments if not provided
        point_to_domain = np.full(n_points, -1, dtype=int)
        
        for i, point in enumerate(points):
            is_inside = [
                np.all((point >= box["min"]) & (point <= box["max"])) 
                for box in domain_boxes
            ]
            
            if any(is_inside):
                candidate_indices = np.where(is_inside)[0]
                distances = [
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
        point_to_domain = np.full(n_points, -1, dtype=int)
        for assignment in point_assignments:
            domain_idx = assignment['domain_index']
            point_indices = assignment['point_indices']
            point_to_domain[point_indices] = domain_idx
    
    # Prepare visualization data
    visualization_data = {
        "points": points,
        "domain_boxes": domain_boxes,
        "point_to_domain": point_to_domain,
        "n_domains": n_domains,
        "n_points": n_points
    }
    
    return visualization_data


def compute_assignment_statistics(visualization_data):
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
            - 'n_domains': total number of domains
            - 'n_points': total number of points
            - 'points_per_domain': array of point counts per domain
            - 'empty_domains': number of domains with no assigned points
            - 'occupied_domains': number of domains with at least one point
            - 'mean_points_per_occupied_domain': average points per non-empty domain
            - 'min_points_per_occupied_domain': minimum points in non-empty domains
            - 'max_points_per_domain': maximum points in any domain
            - 'coverage_ratio': fraction of domains with assigned points
    """
    point_to_domain = visualization_data['point_to_domain']
    n_domains = visualization_data['n_domains']
    n_points = visualization_data['n_points']
    
    # Count points per domain
    points_per_domain = np.array([
        np.sum(point_to_domain == domain_idx) 
        for domain_idx in range(n_domains)
    ])
    
    # Calculate statistics
    empty_domains = np.sum(points_per_domain == 0)
    occupied_domains = n_domains - empty_domains
    
    occupied_counts = points_per_domain[points_per_domain > 0]
    mean_points_occupied = np.mean(occupied_counts) if occupied_domains > 0 else 0
    min_points_occupied = np.min(occupied_counts) if occupied_domains > 0 else 0
    max_points = np.max(points_per_domain)
    
    stats = {
        'n_domains': n_domains,
        'n_points': n_points,
        'points_per_domain': points_per_domain,
        'empty_domains': empty_domains,
        'occupied_domains': occupied_domains,
        'mean_points_per_occupied_domain': mean_points_occupied,
        'min_points_per_occupied_domain': min_points_occupied,
        'max_points_per_domain': max_points,
        'coverage_ratio': occupied_domains / n_domains if n_domains > 0 else 0
    }
    
    return stats


def print_assignment_statistics(visualization_data):
    """
    Print formatted statistics about point-to-domain assignments.
    
    Parameters
    ----------
    visualization_data : dict
        Output from prepare_visualization_data
    """
    stats = compute_assignment_statistics(visualization_data)
    
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
    points_per_domain = stats['points_per_domain']
    unique_counts, count_frequencies = np.unique(points_per_domain, return_counts=True)
    
    print("\nDistribution of points per domain:")
    print("-" * 70)
    for count, freq in zip(unique_counts, count_frequencies):
        bar = '█' * int(freq / max(count_frequencies) * 40)
        if count == 0:
            print(f"  {freq:4d} domains with    0 points  {bar}")
        else:
            print(f"  {freq:4d} domains with {count:4d} points  {bar}")
    print("="*70 + "\n")


def validate_point_assignments(visualization_data, tolerance=1e-10):
    """
    Validate that all points are properly assigned to domains.
    
    Parameters
    ----------
    visualization_data : dict
        Output from prepare_visualization_data
    tolerance : float
        Tolerance for checking if point is inside domain bounding box
    
    Returns
    -------
    validation : dict
        Dictionary containing:
            - 'all_assigned': True if all points have valid domain assignments
            - 'n_unassigned': number of points with invalid assignments
            - 'n_outside_box': number of points outside their assigned domain box
            - 'total_points': total number of points
            - 'is_valid': overall validation status
    """
    points = visualization_data['points']
    point_to_domain = visualization_data['point_to_domain']
    domain_boxes = visualization_data['domain_boxes']
    
    n_unassigned = np.sum(point_to_domain == -1)
    n_outside_box = 0
    
    for i, (point, domain_idx) in enumerate(zip(points, point_to_domain)):
        if 0 <= domain_idx < len(domain_boxes):
            box = domain_boxes[domain_idx]
            # Check if point is inside assigned domain box (with tolerance)
            is_inside = np.all(
                (point >= box["min"] - tolerance) & 
                (point <= box["max"] + tolerance)
            )
            if not is_inside:
                n_outside_box += 1
    
    validation = {
        'all_assigned': n_unassigned == 0,
        'n_unassigned': n_unassigned,
        'n_outside_box': n_outside_box,
        'total_points': len(points),
        'is_valid': (n_unassigned == 0) and (n_outside_box == 0)
    }
    
    return validation


def print_validation_report(visualization_data, tolerance=1e-10):
    """
    Print a validation report for point-to-domain assignments.
    
    Parameters
    ----------
    visualization_data : dict
        Output from prepare_visualization_data
    tolerance : float
        Tolerance for validation checks
    """
    validation = validate_point_assignments(visualization_data, tolerance)
    
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
    print("   - extract_taylor_map: Extract Taylor map from single manifold")
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