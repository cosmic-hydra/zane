def filter_admet(predictions: dict, thresholds: dict = None) -> list:
    if thresholds is None:
        thresholds = {'herg': 0.1, 'hepatotox': 0.05, 'qed': 0.5}
    filtered_indices = []
    for i in range(len(predictions['herg'])):
        if (predictions['herg'][i] < thresholds['herg'] and 
            predictions['hepatotox'][i] < thresholds['hepatotox'] and
            predictions['qed'][i] > thresholds['qed']):
            filtered_indices.append(i)
    return filtered_indices