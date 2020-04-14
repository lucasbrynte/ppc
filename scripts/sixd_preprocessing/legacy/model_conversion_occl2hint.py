import numpy as np

# Columns of R_occl2sixd are the occluded-LINEMOD unit vectors (ordered x,y,z), but expressed in the SIXD hinterstoisser coordinate system.

model_conversion_meta = {
    'ape': {
        'idx_occl': 1,
        'idx_sixd': 1,
        'R_occl2sixd': np.array([
            [0,-1,0],
            [0,0,1],
            [-1,0,0],
        ]).T,
    },
    'can': {
        'idx_occl': 4,
        'idx_sixd': 5,
        'R_occl2sixd': np.array([
            [0,-1,0],
            [0,0,1],
            [-1,0,0],
        ]).T,
    },
    'cat': {
        'idx_occl': 5,
        'idx_sixd': 6,
        'R_occl2sixd': np.array([
            [0,1,0],
            [0,0,1],
            [1,0,0],
        ]).T,
    },
    'driller': {
        'idx_occl': 6,
        'idx_sixd': 8,
        'R_occl2sixd': np.array([
            [0,-1,0],
            [0,0,1],
            [-1,0,0],
        ]).T,
    },
    'duck': {
        'idx_occl': 7,
        'idx_sixd': 9,
        'R_occl2sixd': np.array([
            [0,1,0],
            [0,0,1],
            [1,0,0],
        ]).T,
    },
    'eggbox': {
        'idx_occl': 8,
        'idx_sixd': 10,
        'R_occl2sixd': np.array([
            [0,-1,0],
            [0,0,1],
            [-1,0,0],
        ]).T,
    },
    'glue': {
        'idx_occl': 9,
        'idx_sixd': 11,
        'R_occl2sixd': np.array([
            [0,-1,0],
            [0,0,1],
            [-1,0,0],
        ]).T,
    },
    'holepuncher': {
        'idx_occl': 10,
        'idx_sixd': 12,
        'R_occl2sixd': np.array([
            [0,1,0],
            [0,0,1],
            [1,0,0],
        ]).T,
    },
}
