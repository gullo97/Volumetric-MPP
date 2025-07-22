import numpy as np
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor, as_completed


def smooth_spectrum_moving_average(spectrum, window_size):
    """Apply a simple moving average smoothing to the spectrum."""
    if window_size < 2:
        return spectrum.copy()
    kernel = np.ones(window_size) / window_size
    return np.convolve(spectrum, kernel, mode='same')


def aggregate_spectrum_channels(spectrum, bins_factor):
    """
    Aggregate the spectrum by averaging over bins_factor channels.
    Returns the aggregated spectrum and a mapping function to recover the
    original index.
    """
    n = len(spectrum)
    if bins_factor <= 1:
        return spectrum, lambda idx: idx  # Identity mapping
    trim = n % bins_factor
    if trim:
        spectrum = spectrum[:-trim]
    agg_spec = spectrum.reshape(-1, bins_factor).mean(axis=1)
    # Mapping: aggregated index i maps to original index 
    # ~ i*bins_factor + bins_factor/2.
    mapping = lambda i: int(round(i * bins_factor + bins_factor / 2))
    return agg_spec, mapping


def find_peaks_volumetric_persistence(spectrum, 
                                      smoothing_range, 
                                      bins_factor_range, 
                                      threshold_range, 
                                      width_range, 
                                      prominence_range,
                                      distance_range,   
                                      merging_range=5,
                                      tol=1,
                                      parallel=True,
                                      top_k=None,
                                      channel_ranges=None):
    """
    Detect peaks using a grid search over multiple parameters and 
    quantify each peak's volumetric persistence in the hyperparameter space.
    
    Parameters:
        spectrum (np.ndarray): 1D raw spectrum.
        smoothing_range (iterable): Range of smoothing window sizes.
        bins_factor_range (iterable): Range of bins aggregation factors.
        threshold_range (iterable): Range of threshold values.
        width_range (iterable): Range of width values.
        prominence_range (iterable): Range of prominence values.
        distance_range (iterable): Range of distance values for find_peaks.
        merging_range (int): Range for merging nearby detections.
        tol (int): Tolerance for clustering detections within a grid.
        parallel (bool): Whether to use parallel processing.
        top_k (int or None): If set, return only top k peaks.
        channel_ranges (list or None): List of (start, end) channel ranges.
            If provided, persistence analysis is applied only to these ranges.
    
    Returns:
        List[dict]: Each dictionary contains:
            - 'peak_index': Peak position in original spectrum.
            - 'persistence': Aggregated persistence.
            - 'grid_params': Parameter combinations detecting the peak.
    """
    # For simplicity, assume equal contribution to cell volume
    inner_cell_volume = 1.0

    # If channel ranges are provided, process each range separately
    if channel_ranges:
        return _find_peaks_in_ranges(
            spectrum, channel_ranges, smoothing_range, bins_factor_range,
            threshold_range, width_range, prominence_range, distance_range,
            merging_range, tol, parallel, top_k
        )

    # Original full-spectrum analysis
    candidates = {}
    
    # Outer grid: loop over smoothing and bins_factor combinations
    for smooth in smoothing_range:
        for bins in bins_factor_range:
            # Preprocess the spectrum once per outer combination
            smoothed = smooth_spectrum_moving_average(spectrum, smooth)
            proc_spec, mapping_fn = aggregate_spectrum_channels(smoothed, bins)
            
            # Build inner grid of parameters (now 4D)
            inner_grid = []
            for thr in threshold_range:
                for wid in width_range:
                    for prom in prominence_range:
                        for dist in distance_range:
                            # Include outer parameters in parameter set
                            params = {
                                'smoothing': smooth,
                                'bins_factor': bins,
                                'threshold': thr,
                                'width': wid,
                                'prominence': prom,
                                'distance': dist
                            }
                            inner_grid.append(params)
            
            # Function to process one inner grid parameter set.
            def process_inner(params):
                peaks, _ = find_peaks(proc_spec, threshold=params['threshold'], 
                                      width=params['width'], prominence=params['prominence'],
                                      distance=params['distance'])
                return params, peaks

            # Process the inner grid (optionally in parallel).
            if parallel:
                futures = []
                with ThreadPoolExecutor() as executor:
                    for params in inner_grid:
                        futures.append(executor.submit(process_inner, params))
                    inner_results = [future.result() for future in as_completed(futures)]
            else:
                inner_results = [process_inner(params) for params in inner_grid]
            
            # Accumulate detections.
            for params, peaks in inner_results:
                for p in peaks:
                    key = (smooth, bins, p)
                    # Cluster with an existing candidate within tol in aggregated index space.
                    found = False
                    for existing_key in list(candidates.keys()):
                        if existing_key[0] == smooth and existing_key[1] == bins and abs(p - existing_key[2]) <= tol:
                            candidates[existing_key]['grid_params'].append(params)
                            candidates[existing_key]['count'] += 1
                            found = True
                            break
                    if not found:
                        candidates[key] = {
                            'grid_params': [params],
                            'count': 1,
                            'outer': (smooth, bins),
                            'agg_index': p,
                            'mapping_fn': mapping_fn  # same for this outer set
                        }
    
    # Compute persistence for each candidate.
    candidate_list = []
    for cand in candidates.values():
        persistence = cand['count'] * inner_cell_volume
        candidate_list.append({
            'outer': cand['outer'],
            'agg_index': cand['agg_index'],
            'persistence': persistence,
            'grid_params': cand['grid_params'],
            'mapping_fn': cand['mapping_fn']
        })
    
    # Merge nearby candidates across all outer grid settings.
    candidate_list.sort(key=lambda c: c['mapping_fn'](c['agg_index']))
    merged = []
    for cand in candidate_list:
        orig_idx = cand['mapping_fn'](cand['agg_index'])
        merged_found = False
        for m in merged:
            if abs(orig_idx - m['peak_index']) <= merging_range:
                total_pers = m['persistence'] + cand['persistence']
                m['peak_index'] = int(round((m['peak_index'] * m['persistence'] + orig_idx * cand['persistence']) / total_pers))
                m['persistence'] = total_pers
                m['grid_params'].extend(cand['grid_params'])
                merged_found = True
                break
        if not merged_found:
            merged.append({
                'peak_index': orig_idx,
                'persistence': cand['persistence'],
                'grid_params': cand['grid_params']
            })
    
    # Sort merged candidates by persistence in descending order.
    merged.sort(key=lambda m: m['persistence'], reverse=True)
    
    # Return only top_k peaks if requested.
    if top_k is not None:
        merged = merged[:top_k]
    
    return merged


def _find_peaks_in_ranges(spectrum, channel_ranges, smoothing_range, 
                         bins_factor_range, threshold_range, width_range,
                         prominence_range, distance_range, merging_range,
                         tol, parallel, top_k):
    """
    Apply persistence analysis only to specified channel ranges.
    
    This function extracts sub-spectra for each range, applies persistence
    analysis, and carefully maps the results back to the original spectrum
    coordinates, accounting for bin aggregation.
    """
    inner_cell_volume = 1.0
    all_candidates = []
    
    for range_start, range_end in channel_ranges:
        # Extract sub-spectrum for this range
        sub_spectrum = spectrum[range_start:range_end + 1]
        if len(sub_spectrum) == 0:
            continue
            
        # Track candidates for this range
        range_candidates = {}
        
        # Apply persistence analysis to sub-spectrum
        for smooth in smoothing_range:
            for bins in bins_factor_range:
                # Process sub-spectrum
                smoothed = smooth_spectrum_moving_average(sub_spectrum, smooth)
                proc_spec, mapping_fn = aggregate_spectrum_channels(
                    smoothed, bins)
                
                # Build inner grid
                inner_grid = []
                for thr in threshold_range:
                    for wid in width_range:
                        for prom in prominence_range:
                            for dist in distance_range:
                                params = {
                                    'smoothing': smooth,
                                    'bins_factor': bins,
                                    'threshold': thr,
                                    'width': wid,
                                    'prominence': prom,
                                    'distance': dist
                                }
                                inner_grid.append(params)
                
                # Process inner parameters
                def process_inner(params):
                    peaks, _ = find_peaks(
                        proc_spec, 
                        threshold=params['threshold'],
                        width=params['width'], 
                        prominence=params['prominence'],
                        distance=params['distance']
                    )
                    return params, peaks

                # Execute grid search
                if parallel:
                    futures = []
                    with ThreadPoolExecutor() as executor:
                        for params in inner_grid:
                            futures.append(executor.submit(process_inner, params))
                        inner_results = [future.result() 
                                       for future in as_completed(futures)]
                else:
                    inner_results = [process_inner(params) 
                                   for params in inner_grid]
                
                # Accumulate detections for this range
                for params, peaks in inner_results:
                    for p in peaks:
                        key = (smooth, bins, p)
                        # Cluster with existing candidates
                        found = False
                        for existing_key in list(range_candidates.keys()):
                            if (existing_key[0] == smooth and 
                                existing_key[1] == bins and 
                                abs(p - existing_key[2]) <= tol):
                                range_candidates[existing_key][
                                    'grid_params'].append(params)
                                range_candidates[existing_key]['count'] += 1
                                found = True
                                break
                        if not found:
                            range_candidates[key] = {
                                'grid_params': [params],
                                'count': 1,
                                'outer': (smooth, bins),
                                'agg_index': p,
                                'mapping_fn': mapping_fn,
                                'range_start': range_start  # Track range
                            }
        
        # Convert range candidates to candidates with original indices
        for cand in range_candidates.values():
            # Map from sub-spectrum aggregated index to original spectrum
            sub_orig_idx = cand['mapping_fn'](cand['agg_index'])
            orig_idx = range_start + sub_orig_idx
            
            # Ensure we don't go beyond range bounds
            orig_idx = min(orig_idx, range_end)
            orig_idx = max(orig_idx, range_start)
            
            persistence = cand['count'] * inner_cell_volume
            all_candidates.append({
                'peak_index': orig_idx,
                'persistence': persistence,
                'grid_params': cand['grid_params']
            })
    
    # Merge nearby candidates across all ranges
    all_candidates.sort(key=lambda c: c['peak_index'])
    merged = []
    for cand in all_candidates:
        orig_idx = cand['peak_index']
        merged_found = False
        for m in merged:
            if abs(orig_idx - m['peak_index']) <= merging_range:
                total_pers = m['persistence'] + cand['persistence']
                # Weighted average of positions
                new_pos = ((m['peak_index'] * m['persistence'] + 
                           orig_idx * cand['persistence']) / total_pers)
                m['peak_index'] = int(round(new_pos))
                m['persistence'] = total_pers
                m['grid_params'].extend(cand['grid_params'])
                merged_found = True
                break
        if not merged_found:
            merged.append({
                'peak_index': orig_idx,
                'persistence': cand['persistence'],
                'grid_params': cand['grid_params']
            })
    
    # Sort by persistence (descending)
    merged.sort(key=lambda m: m['persistence'], reverse=True)
    
    # Apply top_k filtering - select k most persistent, then sort by channel
    if top_k is not None and len(merged) > top_k:
        # Step 1: Get top k most persistent peaks
        top_persistent = merged[:top_k]
        # Step 2: Sort by channel position for consistency
        top_persistent.sort(key=lambda m: m['peak_index'])
        merged = top_persistent
    
    return merged