#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class Bug0Navigation:
    def __init__(self):
        rospy.init_node('bug0_navigation')

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        self.obstacle_detected = False

        self.rate = rospy.Rate(10)

    def scan_callback(self, msg):
        # Check if obstacle detected
        if min(msg.ranges) < 1.0: # You may need to adjust this threshold
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False

    def navigate(self):
        while not rospy.is_shutdown():
            if not self.obstacle_detected:
                # Move forward
                self.cmd_vel_pub.publish(Twist(linear=Vector3(x=0.2), angular=Vector3(z=0.0)))
            else:
                # Turn left
                self.cmd_vel_pub.publish(Twist(linear=Vector3(x=0.0), angular=Vector3(z=0.5)))
            self.rate.sleep()

if __name__ == '__main__':
    try:
        nav = Bug0Navigation()
        nav.navigate()
    except rospy.ROSInterruptException:
        pass

