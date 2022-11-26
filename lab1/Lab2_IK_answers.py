import numpy as np
from scipy.spatial.transform import Rotation as R

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end() 
    joint_name = meta_data.joint_name
    # 这里不能直接给列表赋值。
    joint_parent = meta_data.joint_parent.copy()
    joint_initial_position = meta_data.joint_initial_position
    end_joint = meta_data.end_joint
    
    # 计算各个 joint 的偏移。
    joint_offset = [joint_initial_position[i] - joint_initial_position[joint_parent[i]] for i in range(1, len(joint_name))]
    joint_offset.insert(0, joint_initial_position[0])

    # 调整 path2 上 joint 的父节点。
    i = 0
    joint_parent[path[i]] = -1
    while path[i] != 0:
        joint_parent[path[i + 1]] = path[i]
        i += 1

    dis = float("inf")
    iter = 0
    
    path = list(reversed(path))
    while iter < 30 and dis >= 0.01:
        for index in path:
            if joint_name[index] == end_joint:
                continue
            
            rotIndex = index
            if index in path2 and joint_parent[index] != -1:
                rotIndex = joint_parent[index]
            dirtSrc = joint_positions[joint_name.index(end_joint)] - joint_positions[rotIndex]
            dirtDist = target_pose - joint_positions[rotIndex]
            unit_dirtSrc = dirtSrc / np.linalg.norm(dirtSrc)
            unit_dirtDist = dirtDist / np.linalg.norm(dirtDist)
            rotAngle = np.arccos(np.dot(unit_dirtSrc, unit_dirtDist))
            rotAxis = np.cross(dirtSrc, dirtDist)
            rotAxis = rotAxis / np.linalg.norm(rotAxis)
            rot = R.from_rotvec(rotAngle * rotAxis)

            joint_orientations[index] = (rot * R.from_quat(joint_orientations[index])).as_quat()
            if index in path2 and joint_parent[index] != -1:
                joint_positions[index] = joint_positions[joint_parent[index]] - R.from_quat(joint_orientations[index]).apply(joint_offset[joint_parent[index]])
                for j in path2[path2.index(index) + 1 : ]:
                    joint_orientations[j] = (rot * R.from_quat(joint_orientations[j])).as_quat()
                    joint_positions[j] = joint_positions[joint_parent[j]] - R.from_quat(joint_orientations[j]).apply(joint_offset[joint_parent[j]]) 
            
            for j in range(len(joint_name)):
                if j in path2:
                    continue
                parent = joint_parent[j]
                while parent != -1:
                    if parent != index:
                        parent = joint_parent[parent]
                    else:
                        joint_orientations[j] = (rot * R.from_quat(joint_orientations[j])).as_quat()
                        joint_positions[j] = joint_positions[joint_parent[j]] + R.from_quat(joint_orientations[joint_parent[j]]).apply(joint_offset[j])
                        break

        dis = np.linalg.norm(target_pose - joint_positions[path[0]])
        iter += 1
    
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end() 
    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_initial_position = meta_data.joint_initial_position
    end_joint = meta_data.end_joint

    joint_offset = [joint_initial_position[i] - joint_initial_position[joint_parent[i]] for i in range(1, len(joint_name))]
    joint_offset.insert(0, joint_initial_position[0])

    # 是从上一个位置开始！
    target_position = joint_positions[joint_name.index("RootJoint")] + [relative_x, 0, relative_z]
    target_position[1] = target_height

    dis = float("inf")
    iter = 0
    
    path = list(reversed(path))
    while iter < 10 and dis >= 0.01:
        for index in path:
            if joint_name[index] == end_joint:
                continue
            
            rotIndex = index
            if index in path2 and joint_parent[index] != -1:
                rotIndex = joint_parent[index]
            dirtSrc = joint_positions[joint_name.index(end_joint)] - joint_positions[rotIndex]
            dirtDist = target_position - joint_positions[rotIndex]
            unit_dirtSrc = dirtSrc / np.linalg.norm(dirtSrc)
            unit_dirtDist = dirtDist / np.linalg.norm(dirtDist)
            rotAngle = np.arccos(np.dot(unit_dirtSrc, unit_dirtDist))
            rotAxis = np.cross(dirtSrc, dirtDist)
            rotAxis = rotAxis / np.linalg.norm(rotAxis)
            rot = R.from_rotvec(rotAngle * rotAxis)

            joint_orientations[index] = (rot * R.from_quat(joint_orientations[index])).as_quat()
            if joint_parent[index] != -1:
                joint_positions[index] = joint_positions[joint_parent[index]] + R.from_quat(joint_orientations[joint_parent[index]]).apply(joint_offset[index])
            
            for j in range(len(joint_name)):
                parent = joint_parent[j]
                while parent != -1:
                    if parent != index:
                        parent = joint_parent[parent]
                    else:
                        joint_orientations[j] = (rot * R.from_quat(joint_orientations[j])).as_quat()
                        joint_positions[j] = joint_positions[joint_parent[j]] + R.from_quat(joint_orientations[joint_parent[j]]).apply(joint_offset[j])
                        break

        dis = np.linalg.norm(target_position - joint_positions[path[0]])
        iter += 1

    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations