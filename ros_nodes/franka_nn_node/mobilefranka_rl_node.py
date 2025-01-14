import rospy
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryActionGoal, GripperCommandActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PointStamped


import onnxruntime as ort
import numpy as np
#import moveit_commander
import sys
from collections import defaultdict


class MobileFrankaRLNode:

    def __init__(self, argv):

        experiment = argv[1]
        point = argv[2]
        method = argv[3]
        run_number = argv[4]
        self.output_filename = "{}_{}_{}_{}.csv".format(experiment, point, method, run_number)

        self.arm_joint_sub = rospy.Subscriber('/franka_state_controller/joint_states', JointState, self.arm_callback)
        #self.gripper_joint_sub = rospy.Subscriber('/franka_gripper/joint_states', JointState, self.gripper_callback)
        self.optitrack_pose_sub = rospy.Subscriber('/vrpn_client_node/bighusky/pose', PoseStamped, self.optitrack_callback)
        self.trajectory_goal_pub = rospy.Publisher("/position_joint_trajectory_controller/follow_joint_trajectory/goal", FollowJointTrajectoryActionGoal, queue_size=20)
        #self.gripper_goal_pub = rospy.Publisher("/franka_gripper/gripper_action/goal", GripperCommandActionGoal, queue_size=20)
        self.base_cmd_vel_pub = rospy.Publisher("/husky/raw_cmd_vel", Twist, queue_size=20)

        self.target_pub = rospy.Publisher("/target", PointStamped, queue_size=20)
        
        #self.ort_model = ort.InferenceSession("models/single_agent_mobilefranka.onnx")
        #self.ort_model = ort.InferenceSession("models/mobilefrankaMARL_cv.onnx")
        #self.ort_model = ort.InferenceSession("models/m1_exp_reward_mobilefranka.onnx")

        #self.ort_model = ort.InferenceSession("mobilefranka_no_base_vel_yaw_fix_easier_target.onnx")

        if experiment == "e1" or experiment == "e4":
            self.arm_control = True
        else:
            self.arm_control = False
        
        if experiment == "e4":
            self.base_control = False
        else:
            self.base_control = True

        if method == "baseline":
            #self.ort_model = ort.InferenceSession("models/single_agent_mobilefranka.onnx")
            self.ort_model = ort.InferenceSession("models/newmodels/baselinebest.onnx")
            self.single_agent = True
            self.use_cv = False
        elif method == "m1":
            self.ort_model = ort.InferenceSession("models/m1_exp_reward_mobilefranka.onnx")
            #self.ort_model = ort.InferenceSession("models/newmodels/m1best.onnx")
            self.single_agent = False
            self.use_cv = False
        elif method == "m2":
            #self.ort_model = ort.InferenceSession("models/mobilefrankaMARL_cv.onnx")
            self.ort_model = ort.InferenceSession("models/newmodels/m2best.onnx")
            self.single_agent = False
            self.use_cv = True

        self.joint_positions = np.zeros(9)
        self.joint_velocities = np.zeros(9)

        # arm joint dof limits from isaac
        self.lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,  0.0000, 0.0000])
        self.upper_limits = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,  0.0400, 0.0400])

        default_joint_pos = [0.0, -0.7856, 0.0, -2.356, 0.0, 1.572, 0.7854, 0.035, 0.035]

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
        
        self.start_time = None
        self.data = defaultdict(list)
        rospy.on_shutdown(self.shutdown_hook)

        #self.target_pos = np.array([-2.0, -1.0, 0.5])
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

        self.dt = 1 / 60.0 # 60 Hz

        rospy.Timer(rospy.Duration(self.dt), self.send_control)

        rospy.Timer(rospy.Duration(1/30.0), self.update_base_pose)

        #rospy.Timer(rospy.Duration(1/10.0), self.update_base_velocity)
    
    def shutdown_hook(self):
        import pandas as pd
        print("\nExporting data to csv file: ", self.output_filename)     
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_filename, index=False)
    
    def publish_target(self):
        target = PointStamped()
        target.header.frame_id = "universe"
        target.header.stamp = rospy.Time.now()
        target_point = self.target_pos + self.first_position
        target.point.x = target_point[0]
        target.point.y = target_point[1]
        target.point.z = target_point[2]
        self.target_pub.publish(target)

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
            try:
                optitrack_trans = self.tfBuffer.lookup_transform('universe', 'husky_link', rospy.get_rostime(), rospy.Duration(1.0))
                self.first_position = np.array([optitrack_trans.transform.translation.x, optitrack_trans.transform.translation.y, optitrack_trans.transform.translation.z])
                self.start_time = rospy.Time.now()
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print("tf error")
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
            distance_from_target = np.linalg.norm([self.target_pos - self.left_finger_position])
            current_time = rospy.Time.now()
            elapsed_time = current_time - self.start_time

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
            # if distance_from_target < 0.08:
            #     print("Goal reached!", "distance:", distance_from_target)
            #     rospy.signal_shutdown()
            self.publish_target()

            current_time = rospy.Time.now()
            elapsed_time = current_time - self.start_time

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
            print("tf error")
        
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
        #     trans = self.tfBuffer.lookup_transform('universe', 'husky_link', rospy.get_rostime())
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
        
        if self.first_position is None or self.base_position is None or self.left_finger_position is None:
            print("position not available")
            return
        # Isaac sim obs
        # obs = torch.hstack((
        #     base_pos_xy, 
        #     base_yaw, 
        #     arm_dof_pos_scaled,
        #     #base_vel_xy, 
        #     #base_angvel_z, 
        #     franka_dof_vel[:, 3:] * self.dof_vel_scale,
        #     self.franka_lfinger_pos,
        #     self.target_positions
        # )).to(dtype=torch.float32)
        #
        # base_id = torch.tensor([1.0, 0.0], device=self._device)
        # arm_id = torch.tensor([0.0, 1.0], device=self._device)

        #self.arm_move_group.get_current_joint_values()
        
        try:
            base_pos_xy = self.base_position[:2]
            base_yaw = np.array([self.base_yaw])
            
            # scale position and velocities accordingly
            pos_scaled = 2.0 * (self.joint_positions - self.lower_limits) / (self.upper_limits - self.lower_limits) - 1.0
            vel_scaled = self.joint_velocities * 0.1

            left_finger_pos = self.left_finger_position
            target_pos = self.target_pos

            if self.single_agent:
                observation = np.concatenate((base_pos_xy, base_yaw, pos_scaled, vel_scaled, left_finger_pos, target_pos)).astype(np.float32)
                observation = observation.reshape((1,-1))
                outputs = self.ort_model.run(None, {"obs": observation})
                mu = outputs[0].squeeze()
                sigma = np.exp(outputs[1].squeeze())
                action = np.random.normal(mu, sigma)
                action = np.clip(mu, -1.0, 1.0)

                #print(action.shape)
                base_action = action[:2]
                arm_action = action[2:]
            else:
                base_id = np.array([1.0, 0.0])
                arm_id = np.array([0.0, 1.0])

                if self.use_cv:
                    base_observation = np.concatenate((base_pos_xy, base_yaw, left_finger_pos, target_pos, np.zeros(15), base_id)).astype(np.float32)
                    arm_observation = np.concatenate((pos_scaled, vel_scaled, left_finger_pos, target_pos, arm_id)).astype(np.float32)
                    observation = np.vstack((base_observation, arm_observation))

                    # outputs = self.ort_model.run(None, {"obs": base_observation.reshape((1,-1))})
                    # mu = outputs[0]
                    # sigma = np.exp(outputs[1])
                    # #action = np.random.normal(mu, sigma)
                    # base_action = np.clip(mu, -1.0, 1.0)

                    # outputs = self.ort_model.run(None, {"obs": arm_observation.reshape((1,-1))})
                    # mu = outputs[0]
                    # sigma = np.exp(outputs[1])
                    # #action = np.random.normal(mu, sigma)
                    # arm_action = np.clip(mu, -1.0, 1.0)
                else:
                    base_observation = np.concatenate((base_pos_xy, base_yaw, pos_scaled, vel_scaled, left_finger_pos, target_pos, base_id)).astype(np.float32)
                    arm_observation = np.concatenate((base_pos_xy, base_yaw, pos_scaled, vel_scaled, left_finger_pos, target_pos, arm_id)).astype(np.float32)

                    observation = np.vstack((base_observation, arm_observation))

                outputs = self.ort_model.run(None, {"obs": observation})
                mu = outputs[0]
                sigma = np.exp(outputs[1])
                action = np.random.normal(mu, sigma)
                action = np.clip(mu, -1.0, 1.0)

                base_action = action[0]
                arm_action = action[1]

            

            # if self.task == "ground":
            #     observation = np.concatenate((pos_scaled, vel_scaled, to_target)).astype(np.float32)
            # else:
            #     observation = np.concatenate((pos_scaled, vel_scaled)).astype(np.float32)
            
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
            


            # isaac code for setting joint position targets          
            # targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
            # self.franka_dof_targets[:] = torch.clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)

            # speed scales are changed from isaac sim to be 0.1 times the isaac sim values
            dof_speed_scales = np.array([0.1, 0.1 ,0.1 ,0.1, 0.1, 0.1, 0.1, 0.01, 0.01]) * 0.2
            #dof_speed_scales = np.array([0.01, 0.01 ,0.01 ,0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
            #dof_speed_scales = np.array([1, 1 ,1 ,1, 1, 1, 1, 0.1, 0.1])
            targets = self.joint_targets + dof_speed_scales * self.dt * arm_action * 7.5
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
            
            #print("joint_positions", self.joint_positions[:7])
            #print("joint_goal", joint_goal)
            #print("scaled_action", dof_speed_scales * self.dt * arm_action * 7.5)

            if self.arm_control:
                self.trajectory_goal_pub.publish(goal)

            #self.base_vel_queue.append(base_action)

            #base_action = np.mean(self.base_vel_queue, axis=0)

            # publish base actions as twist message, base_action[0] is the linear velocity, base_action[1] is the angular velocity
            twist = Twist()
            twist.linear.x = base_action[0] * 0.5 * 0.5 # check the speeds 0.2 is safe
            twist.angular.z = base_action[1] * 0.375 * 0.3 # check the speeds 0.1 is safe
            #twist.linear.x = 0.1
            #twist.angular.z = 0.0

            #print("base cmd", twist.linear.x, twist.angular.z)

            #print("base_action:", base_action[:2])
            if self.base_control:
                self.base_cmd_vel_pub.publish(twist)

            # TODO Need to publish the base action as moving average of the last 10 actions
            # self.base_action_buffer.append(base_action)
            # if len(self.base_action_buffer) > 10:
            #     self.base_action_buffer.pop(0)
            # base_action = np.mean(self.base_action_buffer, axis=0)

            
            #print(self.left_finger_position)


        except Exception as e:
            print(type(e))
            print(e)
            pass


if __name__ == '__main__':
    rospy.init_node('rl_node', anonymous=True)
    MobileFrankaRLNode(sys.argv)
    rospy.spin()