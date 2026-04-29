def filter_admet(predictions: dict, thresholds: dict = None) -> list:
    if thresholds is None:
        thresholds = {'herg': 0.3, 'hepatotox': 0.1, 'qed': 0.5}  # Strict hERG CiPA (p<0.3 low risk)
    filtered_indices = []
    for i in range(len(predictions['herg'])):
        herg_ok = predictions['herg'][i] < thresholds['herg']
        hepat_ok = predictions['hepatotox'][i] < thresholds['hepatotox']
        qed_ok = predictions['qed'][i] > thresholds['qed']
        if herg_ok and hepat_ok and qed_ok:
            filtered_indices.append(i)
    return filtered_indices