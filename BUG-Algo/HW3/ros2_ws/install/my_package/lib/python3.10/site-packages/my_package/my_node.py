import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class Bug0Navigation(Node):
    def __init__(self):
        super().__init__('bug0_navigation')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.obstacle_detected = False

    def scan_callback(self, msg):
        # Check if obstacle detected
        if min(min(msg.ranges[288:431]), 10) < 0.4: # You may need to adjust this threshold
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False

    def navigate(self):
        while rclpy.ok():
            if not self.obstacle_detected:
                # Move forward
                twist = Twist()
                twist.linear.x = 0.2
                self.cmd_vel_pub.publish(twist)
            else:
                # Stop forward motion
                twist = Twist()
                self.cmd_vel_pub.publish(twist)
                # Rotate until no obstacle detected
                while self.obstacle_detected:
                    twist.angular.z = 0.5  # Rotate left (adjust angular velocity as needed)
                    self.cmd_vel_pub.publish(twist)
                    rclpy.spin_once(self)
                twist.angular.z = 0.0  # Stop rotation once no obstacle detected
                self.cmd_vel_pub.publish(twist)
            rclpy.spin_once(self)

def main(args=None):
    rclpy.init(args=args)
    nav = Bug0Navigation()
    nav.navigate()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

