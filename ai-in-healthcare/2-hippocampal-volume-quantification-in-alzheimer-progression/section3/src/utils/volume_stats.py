"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    intersection = np.logical_and(a, b).sum()
    volumes = np.sum(a>0) + np.sum(b>0)
    
    if volumes == 0:
        return -1
    
    return 2. * float(intersection) / float(volumes)

def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    intersection = np.sum((a>0)*(b>0))
    union = np.sum(a>0) + np.sum(b>0) - intersection

    if union == 0:
        return -1

    return intersection / union

def Sensitivity(tp, fn):
    """
    Sensitivity = TP / (TP+FP) or TP / Overall Positives
    """
    if (tp + fn) == 0:
        return -1
 
    return tp / (tp + fn)

def Specificity(tn, fp):
    """
    Specificity = TN / (TN+FP) or TN / Overall Negatives
    """
    if (tn + fp) == 0:
        return -1
    
    return tn / (tn + fp)


def get_performance_metrics(a, b):
    """
    a = pred
    b = seg
    
    Bundling the above functions into one to get all desired performance metrics.
    - Dice
    - Jaccard
    - Sensitivity
    - Specificity
    """
    negative = b == 0
    positive = b > 0
    
    tp = np.sum(positive[a==b]) # b > 0 and a == gt
    tn = np.sum(negative[a==b]) # b == 0 and a == gt
    fp = np.sum(negative[a!=b]) # b == 0 but a != gt
    fn = np.sum(positive[a!=b]) # b > 0 but a != gt
    
    ss = Sensitivity(tp, fn)
    sp = Specificity(tn, fp)
    dc = Dice3d(a, b)
    jc = Jaccard3d(a, b) 
    
    results = {'tp': tp, 'tn': tn, 
               'fp': fp, 'fn': fn,
               'sensitivity': ss, 'specificity': sp,
               'dice': dc, 'jaccard': jc}
    
    return results