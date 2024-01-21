import pyrealsense2 as rs
import numpy as np

class RealSense:
    def __init__(self, width, height, name):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        if (name == "Box"):
            self.config.enable_device("941322070581")
        elif (name == "Flag"):
            self.config.enable_device("920312072969")
        self.profile = self.pipeline.start(self.config)
        self.color_sensor = self.profile.get_device().query_sensors()[1]
        self.color_sensor.set_option(rs.option.enable_auto_exposure, False)
        self.color_sensor.set_option(rs.option.enable_auto_white_balance, True)

        self.light_level = 59
        self.light_level = 250
        self.color_sensor.set_option(rs.option.exposure, self.light_level)

    def get_frame(self):
        # Wait for a new frame
        frames = self.pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_data = np.asanyarray(depth_frame.get_data())
        color_data = np.asanyarray(color_frame.get_data())
        return depth_data, color_data
    def light_add(self):
        self.light_level += 1
        print(self.light_level)
        self.color_sensor.set_option(rs.option.enable_auto_exposure, False)
        self.color_sensor.set_option(rs.option.exposure, self.light_level)
        self.color_sensor.set_option(rs.option.enable_auto_white_balance, True)
    def light_sub(self):
        self.light_level -= 1
        print(self.light_level)
        self.color_sensor.set_option(rs.option.enable_auto_exposure, False)
        self.color_sensor.set_option(rs.option.exposure, self.light_level)
        self.color_sensor.set_option(rs.option.enable_auto_white_balance, True)
