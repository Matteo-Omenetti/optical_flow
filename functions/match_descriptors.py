import numpy as np
from scipy.spatial.distance import cdist

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function
    distances = cdist(desc1, desc2, metric='sqeuclidean')

    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)

    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        nn = np.argmin(distances, axis=1)
        matches = np.array([[i, nn[i]] for i in range(len(nn))])

        return matches

    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        nn = np.argmin(distances, axis=0)
        nn_inversed = np.argmin(distances, axis=1)
        matches = np.array([[nn[i], i] for i in range(len(nn))])
        matches_inversed = np.array([[i, nn_inversed[i]] for i in range(len(nn_inversed))])

        matches_to_return = []
        for i in range(len(matches)):
            if matches[i][1] == matches_inversed[matches[i][0]][1]:
                matches_to_return.append(matches[i])

        return np.array(matches_to_return)

    elif method == "ratio":
        # TODO: implement the ratio test matching here
        nn = np.argmin(distances, axis=1)
        matches = np.array([[i, nn[i]] for i in range(len(nn))])

        partitioned = np.partition(distances, (0,1), axis=1)
        ratio = partitioned[:,0] / partitioned[:,1]
        mask = ratio < ratio_thresh
        matches = matches[mask]

        return matches

    else:
        raise NotImplementedError
