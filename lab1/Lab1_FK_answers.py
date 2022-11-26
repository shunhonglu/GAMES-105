import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []

    index = 0
    indexStack = []
    with open(bvh_file_path) as file_object:
        for line in file_object:
            words = line.split()
            if words[0] == "MOTION":
                assert len(indexStack) == 0
                break
            elif words[0] == "ROOT" or words[0] == "JOINT" or words[0] == "End":
                if words[0] == "End":
                    joint_name.append(joint_name[indexStack[-1]] + "_end")
                else:
                    joint_name.append(words[1])

                if words[0] == "ROOT":
                    joint_parent.append(-1)
                else:
                    joint_parent.append(indexStack[-1])

                indexStack.append(index)
                index += 1
            elif words[0] == "OFFSET":
                offset = [float(x) for x in words[1 : 4]]
                joint_offset.append(offset)
            elif words[0] == "}":
                indexStack.pop()

    joint_offset = np.array(joint_offset)
    assert joint_offset.shape == (len(joint_name), 3)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = np.zeros((len(joint_name), 3))
    joint_orientations = np.zeros((len(joint_name), 4))

    frame = motion_data[frame_id]

    joint_positions[0] = frame[ : 3]
    joint_orientations[0] = R.from_euler('XYZ', frame[3 : 6], degrees=True).as_quat()

    j = 0
    for i in range(1, len(joint_name)):
        parent_postition = joint_positions[joint_parent[i]]
        parent_orientation = R.from_quat(joint_orientations[joint_parent[i]])
    
        joint_positions[i] = parent_postition + parent_orientation.apply(joint_offset[i])
        if not joint_name[i].endswith("_end"):
            rotation = R.from_euler('XYZ', frame[6 + 3*j : 9 + 3*j], degrees=True)
            joint_orientations[i] = (parent_orientation * rotation).as_quat()
            j += 1
        else:
            joint_orientations[i] = joint_orientations[joint_parent[i]]

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = load_motion_data(A_pose_bvh_path)
    motion_data_dict = {}

    joint_name_T, joint_parent_T, joint_offset_T = part1_calculate_T_pose(T_pose_bvh_path)
    joint_name_A, joint_parent_A, joint_offset_A = part1_calculate_T_pose(A_pose_bvh_path)

    # T_pose → A_pose
    lshoulder = R.from_euler('XYZ', [0, 0, -45], degrees=True)
    lshoulder_T = R.from_matrix(lshoulder.as_matrix().T)
    rshoulder = R.from_euler('XYZ', [0, 0, 45], degrees=True)
    rshoulder_T = R.from_matrix(rshoulder.as_matrix().T)

    j = 0
    for joint_name in joint_name_A:
        if joint_name == "RootJoint":
            motion_data_dict[joint_name] = motion_data[:, 0 : 6]
        elif joint_name.endswith("_end"):
            continue
        else:
            if joint_name == "lShoulder":
                motion_data_dict[joint_name] = (R.from_euler('XYZ', motion_data[:, 6 + 3*j : 9 + 3*j], degrees=True) * lshoulder).as_euler('XYZ', degrees=True)
            elif joint_name == "lElbow" or joint_name == "lWrist":
                motion_data_dict[joint_name] = (lshoulder_T * R.from_euler('XYZ', motion_data[:, 6 + 3*j : 9 + 3*j], degrees=True) * lshoulder).as_euler('XYZ', degrees=True)
            elif joint_name == "rShoulder":
                motion_data_dict[joint_name] = (R.from_euler('XYZ', motion_data[:, 6 + 3*j : 9 + 3*j], degrees=True) * rshoulder).as_euler('XYZ', degrees=True)
            elif joint_name == "rElbow" or joint_name == "rWrist":
                motion_data_dict[joint_name] = (rshoulder_T * R.from_euler('XYZ', motion_data[:, 6 + 3*j : 9 + 3*j], degrees=True) * rshoulder).as_euler('XYZ', degrees=True)
            else:
                motion_data_dict[joint_name] = motion_data[:, 6 + 3*j : 9 + 3*j]
            j += 1

    j = 0
    for joint_name in joint_name_T:
        if joint_name == "RootJoint":
            motion_data[:, 0 : 6] = motion_data_dict[joint_name]
        elif joint_name.endswith("_end"):
            continue
        else:
            motion_data[:, 6 + 3*j : 9 + 3*j] = motion_data_dict[joint_name]
            j += 1

    return motion_data
