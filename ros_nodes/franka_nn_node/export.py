import rospy
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryActionGoal, GripperCommandActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PointStamped
from collections import deque, defaultdict


import onnxruntime as ort
import numpy as np
#import moveit_commander
import sys

class ExportHelper:

    def __init__(self, argv):
        experiment = argv[1]
        point = argv[2]
        method = argv[3]
        self.output_filename = "{}_{}_{}.csv".format(experiment, point, method)
        #self.output_filename = "e3_p3_baseline.csv"

        # p1   0.4,  -0.6,   0.5
        # p2   0.3,   2.0,   0.7
        # p3  -2.0,  -1.0,   0.4
        # set target position dynamically based on argument "point"
        if point == "p1":
            self.target_pos = np.array([0.4, -0.6, 0.5])
        elif point == "p2":
            self.target_pos = np.array([0.3, 2.0, 0.7])
        elif point == "p3":
            self.target_pos = np.array([-2.0, -1.0, 0.4])
        else:
            print("invalid point")
            return

        self.base_position = None
        self.base_yaw = None
        self.left_finger_position = None
        
        self.first_position = None

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.dt = 1 / 60.0 # 60 Hz

        self.start_time = None

        rospy.Timer(rospy.Duration(1/30.0), self.update_base_pose)

        self.data = defaultdict(list)

        rospy.on_shutdown(self.shutdown_hook)
    
    def shutdown_hook(self):
        import pandas as pd
        print("\nexporting data to csv file: ", self.output_filename)     
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_filename, index=False)

        # Q: what is the command to set sim time parameter in ros
        # A: rosparam set use_sim_time true

        # Q: what do i need to add to rosbag play to make it work
        # A: rosbag play --clock <bagfile>

    
    def update_base_pose(self, timer_event):
        # get the first position as the origin
        if self.first_position is None:
            try:
                optitrack_trans = self.tfBuffer.lookup_transform('universe', 'husky_link', rospy.get_rostime(), rospy.Duration(1.0))
                self.first_position = np.array([optitrack_trans.transform.translation.x, optitrack_trans.transform.translation.y, optitrack_trans.transform.translation.z])
                self.start_time = rospy.Time.now()
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print("first position tf error")
                return
        
        # get the current position and rotation relative to the origin 
        try:
            husky_trans = self.tfBuffer.lookup_transform('universe', 'husky_link', rospy.get_rostime(), rospy.Duration(1.0))
            self.base_position = np.array([husky_trans.transform.translation.x, husky_trans.transform.translation.y, husky_trans.transform.translation.z]) - self.first_position
            base_quat = np.array([husky_trans.transform.rotation.x, husky_trans.transform.rotation.y, husky_trans.transform.rotation.z, husky_trans.transform.rotation.w])
            roll, pitch, yaw = euler_from_quaternion(base_quat)
            #print("yaw % 2*np.pi", yaw % (2*np.pi))
            self.base_yaw = yaw % (2*np.pi)
            left_finger_trans = self.tfBuffer.lookup_transform('universe', 'panda_leftfinger', rospy.get_rostime(), rospy.Duration(1.0))
            left_finger_position = np.array([left_finger_trans.transform.translation.x, left_finger_trans.transform.translation.y, left_finger_trans.transform.translation.z])
            self.left_finger_position = left_finger_position - self.first_position
            #print("self.base_position ", self.base_position)
            #print("self.left_finger_position", self.left_finger_position)

            current_time = rospy.Time.now()

            elapsed_time = current_time - self.start_time

            distance_from_target = np.linalg.norm([self.target_pos - self.left_finger_position])
            print("base x:", '{:.3f}'.format(self.base_position[0]))
            print("base y:", '{:.3f}'.format(self.base_position[1]))
            print("base yaw:", '{:.3f}'.format(self.base_yaw))
            print("left finger x:", '{:.3f}'.format(self.left_finger_position[0]))
            print("left finger y:", '{:.3f}'.format(self.left_finger_position[1]))
            print("left finger z:", '{:.3f}'.format(self.left_finger_position[2]))
            print("target x:", '{:.3f}'.format(self.target_pos[0]))
            print("target y:", '{:.3f}'.format(self.target_pos[1]))
            print("target z:", '{:.3f}'.format(self.target_pos[2]))
            print("distance from target:", distance_from_target)
            print("elapsed time:", elapsed_time.to_sec())
            print("-----------------")

            self.data["base_x"].append(self.base_position[0])
            self.data["base_y"].append(self.base_position[1])
            self.data["base_yaw"].append(self.base_yaw)
            self.data["left_finger_x"].append(self.left_finger_position[0])
            self.data["left_finger_y"].append(self.left_finger_position[1])
            self.data["left_finger_z"].append(self.left_finger_position[2])
            self.data["target_x"].append(self.target_pos[0])
            self.data["target_y"].append(self.target_pos[1])
            self.data["target_z"].append(self.target_pos[2])
            self.data["distance_from_target"].append(distance_from_target)
            self.data["elapsed_time"].append(elapsed_time.to_sec())
                
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("lookup tf error")


if __name__ == '__main__':
    rospy.init_node('rl_node', anonymous=True)
    ExportHelper(sys.argv)
    rospy.spin()

    