# vis pose 2d

import cv2

class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.link_pairs)):
            self.link_pairs[i].append(tuple(np.array(self.color[i])/255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i])/255.))


color2 = [(252,176,243),(252,176,243),(252,176,243),
    (0,176,240), (0,176,240), (0,176,240),
    (255,255,0), (255,255,0),(169, 209, 142),
    (169, 209, 142),(169, 209, 142),
    (240,2,127),(240,2,127),(240,2,127), (240,2,127), (240,2,127)]

link_pairs2 = [
        [15, 13], [13, 11], [11, 5],
        [12, 14], [14, 16], [12, 6],
        [9, 7], [7,5], [5, 6], [6, 8], [8, 10],
        [3, 1],[1, 2],[1, 0],[0, 2],[2,4],
        ]


point_color2 = [(240,2,127),(240,2,127),(240,2,127),
            (240,2,127), (240,2,127),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (252,176,243),(0,176,240),(252,176,243),
            (0,176,240),(252,176,243),(0,176,240),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142)]

chunhua_style = ColorStyle(color2, link_pairs2, point_color2)


def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)

    return joints_dict


def draw_pose_cv2(image, keypoints_list, thickness):
    data_numpy = image.copy()
    h = data_numpy.shape[0]
    w = data_numpy.shape[1]

    for i, dt_joints in enumerate(keypoints_list[:]):
        dt_joints = dt_joints.reshape(17,-1)
        joints_dict = map_joint_dict(dt_joints)

        # stick
        for k, link_pair in enumerate(chunhua_style.link_pairs):
            if k in range(11,16):
                lw = thickness
            else:
                lw = thickness * 2

            cv2.line(
                data_numpy,
                joints_dict[link_pair[0]],
                joints_dict[link_pair[1]],
                tuple([int(c*255) for c in link_pair[2]]),
                lw
            )

        # black ring
        for k in range(dt_joints.shape[0]):
            if k in range(5):
                radius = thickness
            else:
                radius = thickness * 2

            cv2.circle(
                data_numpy,
                tuple(joints_dict[k]),
                radius,
                [0,0,0],
                # tuple([int(c*255) for c in chunhua_style.ring_color[k]]),
                2
            )

    return data_numpy

