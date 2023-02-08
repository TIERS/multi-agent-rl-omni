import rospy
from std_msgs.msg import Int64
from std_srvs.srv import SetBool
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryActionGoal, GripperCommandActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import onnxruntime as ort
import numpy as np
#import moveit_commander
import sys


class RLNode:

    def __init__(self):

        self.arm_joint_sub = rospy.Subscriber('/franka_state_controller/joint_states', JointState, self.arm_callback)
        self.gripper_joint_sub = rospy.Subscriber('/franka_gripper/joint_states', JointState, self.gripper_callback)
        self.trajectory_goal_pub = rospy.Publisher("/effort_joint_trajectory_controller/follow_joint_trajectory/goal", FollowJointTrajectoryActionGoal, queue_size=20)
        self.gripper_goal_pub = rospy.Publisher("/franka_gripper/gripper_action/goal", GripperCommandActionGoal, queue_size=20)
        
        self.task = ""
        if len(sys.argv) > 1:
            self.task = sys.argv[1]
        if self.task == "ground":
            self.ort_model = ort.InferenceSession("franka_reachground.onnx")
        else:
            self.ort_model = ort.InferenceSession("franka_reachup.onnx")

        self.joint_positions = np.zeros(9)
        self.joint_velocities = np.zeros(9)

        # arm joint dof limits from isaac
        self.lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,  0.0000, 0.0000])
        self.upper_limits = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,  0.0400, 0.0400])

        default_joint_pos = [0.0, -0.7856, 0.0, -2.356, 0.0, 1.572, 0.7854, 0.035, 0.035]
        self.joint_targets = None

        #moveit_commander.roscpp_initialize(sys.argv)
        #rospy.init_node("move_group_python_interface_tutorial", anonymous=True)
        #robot = moveit_commander.RobotCommander()
        #self.arm_move_group = moveit_commander.MoveGroupCommander("panda_arm")
        #self.gripper_move_group = moveit_commander.MoveGroupCommander("panda_hand")

        self.dt = 1 / 60.0

        rospy.Timer(rospy.Duration(self.dt), self.send_control)

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

        if self.task == "ground":
            observation = np.concatenate((pos_scaled, vel_scaled, to_target)).astype(np.float32)
        else:
            observation = np.concatenate((pos_scaled, vel_scaled)).astype(np.float32)
        
        observation = observation.reshape((1,-1))

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
        mu = outputs[0].squeeze(0)
        sigma = np.exp(outputs[1].squeeze(0))
        action = np.random.normal(mu, sigma)

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
        print("self.joint_positions", self.joint_positions)
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

        print(joint_goal)

        # slow python interface, maybe useful for some other tasks?
        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        #self.arm_move_group.go(joint_goal, wait=False)

        # Calling ``stop()`` ensures that there is no residual movement
        #self.arm_move_group.stop()


        goal = FollowJointTrajectoryActionGoal()
        
        # timestamps not needed i guess
        # now = rospy.get_rostime()
        # stamp = goal.header.stamp
        # stamp.secs = now.secs
        # stamp.nsecs = now.nsecs
        # goal.header.stamp = stamp
        # goal.goal_id.stamp = stamp

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
        
        #print(goal)
        self.trajectory_goal_pub.publish(goal)
        
        # publish gripper goal also
        # gripper_goal = GripperCommandActionGoal()
        # gripper_goal.goal.command.position = self.joint_targets[7]
        # self.gripper_goal_pub.publish(gripper_goal)


if __name__ == '__main__':
    rospy.init_node('rl_node', anonymous=True)
    RLNode()
    rospy.spin()