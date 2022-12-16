import rclpy
from rclpy.node import Node
import sys
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion, quaternion_from_euler

import onnx
import onnxruntime as ort
import numpy as np
from matplotlib import pyplot as plt


class RLNode(Node):

    def __init__(self):
        super().__init__('rl_node')
        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                          history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                          depth=1)
        self.subscription = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, 'sim_ground_truth_pose', self.odom_callback, qos_profile=qos_policy)
        self.action_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        #onnx_model = onnx.load("jetbot_norm.onnx")
        # Check that the model is well formed
        #onnx.checker.check_model(onnx_model)
        self.ort_model = ort.InferenceSession("jetbot_bigradius3.onnx")

        self.position = None
        self.orientation = None
        self.heading = None
        self.goal_distance = None

        self.target_position = np.array([1.5, 1.5, 0.0])

    def odom_callback(self, msg):
        self.position = np.array([
            msg.pose.pose.position.x, 
            msg.pose.pose.position.y, 
            msg.pose.pose.position.z, 
        ])
        quaternion = np.array([
            msg.pose.pose.orientation.x, 
            msg.pose.pose.orientation.y, 
            msg.pose.pose.orientation.z, 
            msg.pose.pose.orientation.w, 
        ])
        self.orientation = np.array(euler_from_quaternion(quaternion))

        goal_angle = np.arctan2(self.target_position[1] - self.position[1], self.target_position[0] - self.position[0])

        self.heading = goal_angle - self.orientation[2]
        self.heading = np.where(self.heading > np.pi, self.heading - 2 * np.pi, self.heading)
        self.heading = np.where(self.heading < -np.pi, self.heading + 2 * np.pi, self.heading)

        self.goal_distance = np.linalg.norm(self.position - self.target_position)

        #print("pos", self.position)

    def scan_callback(self, msg):
        cmd = Twist()
        cmd.linear.x = 0.15

        #temp = np.array(msg.ranges)
        # print("max", temp.max())
        #print("min", temp.min())
        # print("mean", temp.mean())
        # exit()

        # heading is calculated, it starts as None
        if self.heading:
            #print(len(msg.ranges), msg.angle_min, msg.angle_max)
            observation = np.append(np.flip(np.array(msg.ranges)) - 0.1, (self.heading, self.goal_distance))
            #observation = np.append(np.array(msg.ranges), (self.heading, self.goal_distance))

            #self.polar_to_cartesian_coordinate(observation[:36], -np.pi, 0)
            #print(observation)

            outputs = self.ort_model.run(None, {"obs": observation.astype(np.float32).reshape((1,74))})
            mu = outputs[0].squeeze(1)
            sigma = np.exp(outputs[1].squeeze(1))
            action = np.random.normal(mu, sigma)
            
            cmd.angular.z = action.item() * 0.3
            #cmd.angular.z = mu.item() * 0.3
            #print("output", outputs)
            print("heading, goal distance", (self.heading, self.goal_distance))
        
        self.action_pub.publish(cmd)
    
    def polar_to_cartesian_coordinate(self, ranges, angle_min, angle_max):
        angle_step = (angle_max - angle_min) / len(ranges)
        angle = 180
        points = []
        for range in ranges:
            x = range * np.cos(angle)
            y = range * np.sin(angle)
            angle += angle_step
            points.append([x,y])

        points_np = np.array(points)
        #print(points_np)
        plt.figure()
        colors = np.linspace(0, 1, 36)
        sizes = np.linspace(1, 20, 36)
        plt.scatter(points_np[:,0], points_np[:,1], c=colors, s=sizes)
        plt.show()
    
        return points
        


def main(args=None):
    rclpy.init(args=args)

    rl_node = RLNode()

    rclpy.spin(rl_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    rl_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
