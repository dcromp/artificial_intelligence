#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None  # Current posistion of the car
        self.base_waypoints = None  # Base waypoint object
        self.waypoints_2d = None  # X Y coords of the waypoints
        self.waypoint_tree = None  # Tree object for quick waypoint lookup
        self.stopline_wp_idx = -1

        self.loop()

    def loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:  # The member variables have been initilized
                closest_waypoint_idx = self.get_closest_waypoinyt_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_waypoinyt_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        closest_idx = self.waypoint_tree.query([x,y], 1)[1]  # Return 1 waypoint, 2nd element is the index, first is the distance
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Hyperplane
        cl_vect = np.array(closest_coord)  # Closest waypoint vector
        prev_vect = np.array(prev_coord)  # Previous waypoint vector
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)  # Modulo incase we run out of waypoints
        return closest_idx

    def publish_waypoints(self, closest_idx):
        final_lane = self.generate_lane(closest_idx)  # List of waypoints with updated velocities
        #lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS]
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self, closest_idx):
        lane = Lane()  # New empty lane message

        # Constant number of waypoints ahead of the car
        future_waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS]

        # No traffic light detected or it is too far away to care about
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= closest_idx + LOOKAHEAD_WPS):
            lane.waypoints = future_waypoints

        # Slow the car down because of the detected traffic light
        else:
            lane.waypoints = self.decelerate_waypoints(future_waypoints, closest_idx)

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []  # Place to store new waypoints to we dont overide basewaypoints

        for index, wp in enumerate(waypoints):

            p = Waypoint()
            p.pose = wp.pose

            # Stop a little bit before the stop line
            stop_idx = max(self.stopline_wp_idx - closest_idx -2, 0)
            dist = self.distance(waypoints, index, stop_idx)
            vel = math.sqrt(2*MAX_DECEL*dist)
            if vel < 1.:
                vel = 0
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        return temp

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # Save the waypoints in the object
        self.base_waypoints = waypoints

        if not self.waypoints_2d:
            # Convert waypoints to a list of [x,y] posistions
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            # Construct kd tree
            self.waypoint_tree = KDTree(self.waypoints_2d)

    # Traffic light callback function
    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
