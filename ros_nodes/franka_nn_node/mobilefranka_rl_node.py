import rospy
from std_msgs.msg import Int64
from std_srvs.srv import SetBool
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryActionGoal, GripperCommandActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import onnxruntime as ort
import numpy as np
#import moveit_commander
import sys


class MobileFrankaRLNode:

    def __init__(self):

        self.arm_joint_sub = rospy.Subscriber('/franka_state_controller/joint_states', JointState, self.arm_callback)
        #self.gripper_joint_sub = rospy.Subscriber('/franka_gripper/joint_states', JointState, self.gripper_callback)
        self.optitrack_pose_sub = rospy.Subscriber('/vrpn_client_node/bighusky/pose', PoseStamped, self.optitrack_callback)
        self.trajectory_goal_pub = rospy.Publisher("/position_joint_trajectory_controller/follow_joint_trajectory/goal", FollowJointTrajectoryActionGoal, queue_size=20)
        #self.gripper_goal_pub = rospy.Publisher("/franka_gripper/gripper_action/goal", GripperCommandActionGoal, queue_size=20)
        
        self.ort_model = ort.InferenceSession("mobilefranka.onnx")

        self.joint_positions = np.zeros(9)
        self.joint_velocities = np.zeros(9)

        # arm joint dof limits from isaac
        self.lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,  0.0000, 0.0000])
        self.upper_limits = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,  0.0400, 0.0400])

        default_joint_pos = [0.0, -0.7856, 0.0, -2.356, 0.0, 1.572, 0.7854, 0.035, 0.035]
        self.joint_targets = None

        self.base_position = None
        self.base_yaw = None
        self.left_finger_position = None

        self.last_base_position = None
        self.last_base_yaw = None
        self.base_velocity = None
        self.base_yaw_velocity = None
        
        self.first_position = None

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        #moveit_commander.roscpp_initialize(sys.argv)
        #rospy.init_node("move_group_python_interface_tutorial", anonymous=True)
        #robot = moveit_commander.RobotCommander()
        #self.arm_move_group = moveit_commander.MoveGroupCommander("panda_arm")
        #self.gripper_move_group = moveit_commander.MoveGroupCommander("panda_hand")

        self.dt = 1 / 60.0 # 60 Hz

        rospy.Timer(rospy.Duration(self.dt), self.send_control)

        rospy.Timer(rospy.Duration(1/30.0), self.update_base_pose)

        #rospy.Timer(rospy.Duration(1/10.0), self.update_base_velocity)

    def update_base_velocity(self, timer_event):
        """Calculate the base velocity in the x and y directions and the base yaw velocity"""
        if self.base_position is not None and self.last_base_position is not None:
            self.base_velocity = (self.base_position - self.last_base_position) / (1/10.0)
            self.last_base_position = self.base_position
        elif self.base_position is not None:
            self.base_velocity = np.zeros(3)
            self.last_base_position = self.base_position
        else:
            self.base_velocity = None

        if self.base_yaw is not None and self.last_base_yaw is not None:
            base_yaw_velocity = (self.base_yaw - self.last_base_yaw) / (1/10.0)
            if np.abs(base_yaw_velocity) < 0.1:
                self.base_yaw_velocity = 0
            else:
                self.base_yaw_velocity = base_yaw_velocity
            self.last_base_yaw = self.base_yaw
        elif self.base_yaw is not None:
            self.base_yaw_velocity = 0
            self.last_base_yaw = self.base_yaw
        else:
            self.base_yaw_velocity = None
        print("basevel x", self.base_velocity[0])
        print("basevel y", self.base_velocity[1])
        print("baseyawvel", self.base_yaw_velocity)

    
    def update_base_pose(self, timer_event):
        # get the first position as the origin
        if self.first_position is None:
            optitrack_trans = self.tfBuffer.lookup_transform('universe', 'husky_link', rospy.Time())
            self.first_position = np.array([optitrack_trans.transform.translation.x, optitrack_trans.transform.translation.y, optitrack_trans.transform.translation.z])
        
        # get the current position and rotation relative to the origin 
        try:
            husky_trans = self.tfBuffer.lookup_transform('universe', 'husky_link', rospy.Time())
            self.base_position = np.array([husky_trans.transform.translation.x, husky_trans.transform.translation.y, husky_trans.transform.translation.z]) - self.first_position
            base_quat = np.array([husky_trans.transform.rotation.x, husky_trans.transform.rotation.y, husky_trans.transform.rotation.z, husky_trans.transform.rotation.w])
            roll, pitch, yaw = euler_from_quaternion(base_quat)
            #print("yaw % 2*np.pi", yaw % (2*np.pi))
            self.base_yaw = yaw % (2*np.pi)
            left_finger_trans = self.tfBuffer.lookup_transform('husky_link', 'panda_leftfinger', rospy.Time())
            left_finger_position = np.array([left_finger_trans.transform.translation.x, left_finger_trans.transform.translation.y, left_finger_trans.transform.translation.z])
            self.left_finger_position = self.base_position + left_finger_position
            #print("self.base_position ", self.base_position)
            #print("self.left_finger_position", self.left_finger_position)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("tf2_ros error")
        
        # tf.transformations.euler_from_quaternion
        # tf.transformations.quaternion_from_euler
    
    def optitrack_callback(self, msg):
        # get the first position as the origin
        # if self.first_position is None:
        #     self.first_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        # # get the current position relative to the origin
        # self.base_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]) - self.first_position
        # self.base_rotation = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        # print(self.base_position)
        # try:
        #     trans = self.tfBuffer.lookup_transform('universe', 'husky_link', rospy.Time())
        #     print("trans", trans)
        # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        #     print("tf2_ros error")
        pass

    def arm_callback(self, msg):
        self.joint_positions[:7] = msg.position
        self.joint_velocities[:7] = msg.velocity

    def gripper_callback(self, msg):
        self.joint_positions[7:] = msg.position
        self.joint_velocities[7:] = msg.velocity

    def send_control(self, timer_event):

        if self.joint_targets is None:
            # if self.joint_targets initializes to zeros it can make big movement which could break the real robot
            # so we check that joint_positions is not all zeros before initializing joint targets to it
            if self.joint_positions.sum() == 0:
                print("here self.joint_positions.sum() == 0:")
                return
            else:
                self.joint_targets = self.joint_positions
        
        #self.arm_move_group.get_current_joint_values()

        # scale position and velocities accordingly
        pos_scaled = 2.0 * (self.joint_positions - self.lower_limits) / (self.upper_limits - self.lower_limits) - 1.0
        vel_scaled = self.joint_velocities * 0.1

        to_target = np.array([0.5, 0.5, -0.5])

        # if self.task == "ground":
        #     observation = np.concatenate((pos_scaled, vel_scaled, to_target)).astype(np.float32)
        # else:
        #     observation = np.concatenate((pos_scaled, vel_scaled)).astype(np.float32)
        
        observation = np.zeros((2,)+(32,)).astype(np.float32)

        observation = np.array([
            [ 1.2193e-01, -9.2775e-02,  1.5526e-02,  1.8618e-03, -3.9557e-01,
            8.0547e-03, -5.0866e-01,  3.3527e-02,  1.1909e-01,  2.7887e-01,
            1.0000e+00,  8.3886e-01, -1.1650e-06, -1.9033e-07,  3.9978e-09,
            -2.2243e-05, -2.7235e-03,  2.2866e-04, -1.2848e-02, -2.1203e-05,
            -7.6256e-04, -2.3298e-05,  4.3954e-06, -3.4108e-06,  7.5987e-01,
            -9.5171e-02,  8.5100e-01,  3.0000e+00,  1.0000e+00,  5.0000e-01,
            1.0000e+00,  0.0000e+00],
            [ 1.2193e-01, -9.2775e-02,  1.5526e-02,  1.8618e-03, -3.9557e-01,
            8.0547e-03, -5.0866e-01,  3.3527e-02,  1.1909e-01,  2.7887e-01,
            1.0000e+00,  8.3886e-01, -1.1650e-06, -1.9033e-07,  3.9978e-09,
            -2.2243e-05, -2.7235e-03,  2.2866e-04, -1.2848e-02, -2.1203e-05,
            -7.6256e-04, -2.3298e-05,  4.3954e-06, -3.4108e-06,  7.5987e-01,
            -9.5171e-02,  8.5100e-01,  3.0000e+00,  1.0000e+00,  5.0000e-01,
            0.0000e+00,  1.0000e+00]]).astype(np.float32)
        #observation = observation.reshape((1,-1))

        # isaac code for observations
        # prop_pos = self._props.get_world_poses(clone=False)[0]
        # self.to_target = prop_pos - hand_pos
        
        # self.obs_buf = torch.cat(
        #     (
        #         dof_pos_scaled,
        #         franka_dof_vel * self.dof_vel_scale,
        #         self.to_target,
        #     ),
        #     dim=-1,
        # )
        
        outputs = self.ort_model.run(None, {"obs": observation})
        mu = outputs[0]
        sigma = np.exp(outputs[1])
        action = np.random.normal(mu, sigma)
        action = action[1] # arm action

        # isaac code for setting joint position targets          
        # targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        # self.franka_dof_targets[:] = torch.clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        # speed scales are changed from isaac sim to be 0.1 times the isaac sim values
        dof_speed_scales = np.array([0.1, 0.1 ,0.1 ,0.1, 0.1, 0.1, 0.1, 0.01, 0.01])
        #dof_speed_scales = np.array([0.01, 0.01 ,0.01 ,0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        #dof_speed_scales = np.array([1, 1 ,1 ,1, 1, 1, 1, 0.1, 0.1])
        targets = self.joint_targets + dof_speed_scales * self.dt * action * 7.5
        self.joint_targets = np.clip(targets, self.lower_limits, self.upper_limits)

        # set the goal for the arm joints (not gripper)
        #print("self.joint_positions", self.joint_positions)
        joint_goal = self.joint_targets[:7]
        
        # joint_goal = self.joint_positions[:7]
        # joint_goal = [
        #     0.00125, 
        #     -0.78564, 
        #     -0.00131, 
        #     -2.35635,
        #     -0.00366,
        #     1.57296,
        #     0.79711
        # ]

        #print(joint_goal)

        goal = FollowJointTrajectoryActionGoal()

        point = JointTrajectoryPoint()
        point.positions = joint_goal
        # this time_from_start is important, otherwise it won't work
        point.time_from_start.nsecs = 500000000
        goal.goal.trajectory.points.append(point)

        joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6", 
            "panda_joint7"
        ]
        goal.goal.trajectory.joint_names = joint_names
        
        self.trajectory_goal_pub.publish(goal)


if __name__ == '__main__':
    rospy.init_node('rl_node', anonymous=True)
    MobileFrankaRLNode()
    rospy.spin()