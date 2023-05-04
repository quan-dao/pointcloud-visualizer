import numpy as np
from utils.visualization_toolsbox import PointsPainter


def main():
    points = np.load('artifact/35e4b30bfa6a42bd87ce79ce9248240e.npy')
    print('points: ', points.shape)

    unq_sweep_ids = np.unique(points[:, -1].astype(int))
    print('unq_sweep_ids: ', unq_sweep_ids)

    painter = PointsPainter(points[:, :3])
    painter.show()
    


if __name__ == '__main__':
    main() 
