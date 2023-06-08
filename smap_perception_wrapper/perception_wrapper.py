#!/usr/bin/env python3

import rclpy
import traceback
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from smap_interfaces.msg import SmapData, SmapDetections

from smap_interfaces.srv import AddPerceptionModule

from sensor_msgs.msg import PointCloud2, PointField
import time
import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import PointCloud2

import struct
import ctypes

xyzrgba_struct = struct.Struct('<ffff')


class timer:
    # Timer based on https://github.com/ultralytics/yolov5/blob/master/utils/general.py
    def __init__(self):
        self.t = 0

    def __enter__(self):
        self.t = 0
        self.start = time.time()
        return self
    
    def __exit__(self, type, value, traceback):
        self.end = time.time()
        self.t = (self.end - self.start)*1E3
    
    def reset(self):
        self.t = 0

a = timer()

class perception_wrapper(Node):

    module_id = 0
    classes = [] # must be a dictionary of class names. (e.g.) {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle'}
    parameters = []

    imgsz=None
    model=None
    device=None
    model_file=None
    model_description_file=None


    _cv_bridge=CvBridge()
    _img_original=None
    _img_processed=None
    _predictions=None
    _callback_time=0
    __initialization_state=0
    __state_busy=False
    

    def __init__(self,detector_name='Cl1',detector_type='object',detector_architecture='no_defined'):
        super().__init__("perception_{}".format(detector_name))

        self.get_logger().info("Initializing smap wrapper...")

        # Wrapper parameters
        self.declare_parameter('model',value=None)
        self.declare_parameter('model_description',value=None)

        # Wrapper private variables
        self._reentrant_cb_group=ReentrantCallbackGroup()
        self._mutuallyexclusive_cb_group=MutuallyExclusiveCallbackGroup()
        self.__ret_valid=10
        self.__shutdown_node=False

        # Timer
        self.__node_timer=self.create_timer(1.0,self.__states_next__)

        # Wrapper public variables
        self.detector_name=detector_name
        self.detector_type=detector_type
        self.detector_architecture=detector_architecture
        self.model_file = self.get_parameter('model').value
        self.model_description_file = self.get_parameter('model_description').value

        self.pre_processing_tim=timer()
        self.inference_tim=timer()
        self.nms_tim=timer()
        self.post_processing_tim=timer()

    def __states_next__(self):
        # Initialization:
        #   __states_next__ -> 0
        #       > setup_detector()
        #           initialize detector object
        #   __states_next__ -> 1
        #       > __add_detector__()
        #           add detector to the server
        #   __states_next__ -> 2
        #       > __validate_detector__()
        #           wait for server validation
        #   __states_next__ -> 3
        #       > init detector messages
        #   __states_next__ -> 4
        #       > initialization()
        #           initialize topics

        if self.__shutdown_node==True or self.__state_busy==True:
            return
        
        self.__state_busy=True
        if self.__initialization_state == 0:
            if self.setup_detector(): # Return True when done!
                self.__initialization_state+=1
        elif self.__initialization_state == 1:
            if self.__add_detector__(): # Return True when done!
                self.__initialization_state+=1
        elif self.__initialization_state == 2:
            if self.__validate_detector__(): # Return True when done!
                self.__initialization_state+=1
        elif self.__initialization_state == 3:
            self.get_logger().info("smap perception detector:")
            self.get_logger().info("\t name: {}".format(self.detector_name))
            self.get_logger().info("\t id: {}".format(self.module_id))
            self.get_logger().info("\t type: {}".format(self.detector_type))
            self.get_logger().info("\t architecture: {}".format(self.detector_architecture))
            self.get_logger().info("smap wrapper launched!")
            self.__initialization_state+=1
        elif self.__initialization_state == 4:
            if self.initialization(): # Return True when done!
                self.__initialization_state+=1
        else:
            pass

        self.__state_busy=False
        
    def __validate_detector__(self): # Return True when done!
        #self.__ret_valid=10
        if self.fut_val.done():
            try:
                resp=self.fut_val.result()
            except Exception as e:
                self.get_logger().error("Add detector request failed!")
                self.__shutdown_node=True
                return False
        else:
            if self.__ret_valid==10:
                self.get_logger().warning('Waiting for service response...')
                self.__ret_valid-=1
                return False
            elif self.__ret_valid==-1:
                self.get_logger().error('No service response!')
                self.__shutdown_node=True
                return False
            else:
                self.get_logger().warning('Waiting for service response [{}/10]...'.format(10-self.__ret_valid))
                self.__ret_valid=self.__ret_valid-1
                return False
            
        self.get_logger().info("Processing service response.")

        if not resp.success:
            self.get_logger().error("Add detector request failed!")
            self.__shutdown_node=True
            return False
        
        self.module_id=resp.module_id

        return True

    def __add_detector__(self): # Return True when done!
        """Send a request to the server to include a new perception module"""
        self.cli = self.create_client(
            AddPerceptionModule,
            self.get_namespace()+"/perception_server/add_perception_module"
        )
        ret=10
        self.get_logger().info("smap wrapper wainting for service \'{}\'".format(self.detector_name,self.get_namespace()+"/perception_server/add_perception_module"))
        while not self.cli.wait_for_service(timeout_sec=1.0):
            if ret==10:
                self.get_logger().warning('Service not available, trying again...')
            else:
                if ret==-1:
                    self.get_logger().error("Service not available!")
                    self.__shutdown_node=True
                    return False
                else:
                    self.get_logger().warning('Service not available [{}/10], waiting again...'.format(10-ret))
            ret=ret-1
        self.get_logger().info("Successful connection to service \'{}\'.".format(self.get_namespace()+"/perception_server/add_perception_module"))
        req=AddPerceptionModule.Request()
        req.name=self.detector_name
        req.type=self.detector_type
        req.architecture=self.detector_architecture
        req.n_classes=len(self.classes.keys())
        req.classes=list(self.classes.values())
        req.class_ids = list(self.classes.keys())


        self.get_logger().info("Sending request...")
        self.fut_val=self.cli.call_async(req)
        return True

    def get_callback_time(self):
        return self.pre_processing_tim.t + self.inference_tim.t + self.nms_tim.t + self.post_processing_tim.t

    def model_warmup(self):
        """Pipeline warmup"""
        self.get_logger().info('Model warming up...')

    def load_dataloader(self):
        """Load the dataloader"""
        self.get_logger().info('Initializing dataloader...')
        return False

    def __shutdown__(self):
        self.get_logger().fatal("Shutting down node!")
        #self.destroy_node()

    def setup_detector(self): # Return True when done!
        """
            Detector initialization: \n
            1. Checks the runtime device [check_device()] \n   
            2. Load the model [load_model()] \n
            3. Load the dataloader [load_dataloader()] \n
            4. Model warmup [model_warmup()] \n
        """

        self.get_logger().info("Setting up detector...")
        
        # Check device
        if self.check_device(): # Return True when an error occurs
            self.get_logger().fatal("Device not defined!")
            self.__shutdown_node=True
            return False
        self.get_logger().info("Runtime device ok")
        
        # Load model
        if self.load_model(): # Return True when an error occurs
            self.get_logger().fatal("Model file not defined!")
            self.__shutdown_node=True
            return False
        self.get_logger().info("Model ok")
        
        # Load dataloader
        if self.load_dataloader(): # Return True when an error occurs
            self.get_logger().fatal("Dataloader Error!")
            self.__shutdown_node=True
            return False
        self.get_logger().info('Dataloader: ok')
        
        # Model Warmup
        self.model_warmup()
        self.get_logger().info('Warm up complete')

        self.get_logger().info('Detector setup complete')

        return True

    def check_device(self): # Return True when an error occurs
        if self.device is None:
            return True
        return False

    def load_model(self): # Return True when an error occurs
        if (self.model_file is None):
            return True
        if (self.model_description_file is None):
            return True
        self.get_logger().info("Loading Model {}\n{}...".format(self.model_file,self.model_description_file))
        return False
    
    def initialization(self):
        self.get_logger().info("Initializing topics")
        self.subscription=self.create_subscription(SmapData, self.get_namespace()+'/sampler/data', self.__predict, 10,callback_group=self._mutuallyexclusive_cb_group)
        self.detections=self.create_publisher(SmapDetections, self.get_namespace()+'/perception/predictions', 10,callback_group=self._reentrant_cb_group)
        return True

    def on_process(self): # Pooling
        if self.__shutdown_node:
            self.__shutdown__()
            return True

    def train(self,data):
        pass

    def predict(self,data):
        self.get_logger().info('Model prediction')
        msg = SmapDetections()
        msg.module_id=0
        self.publisher.publish(msg)

    def letterbox(self,im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleUp=True, stride=32):
        # https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleUp:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)
    
    def __predict(self,input):
        
        try:
            resp_msg = self.predict(input)
        except Exception as e:
            self.get_logger().error("perception_wrapper/predict")
            print(e)
            return
        
        if resp_msg:
            for obj in resp_msg.objects:
                if obj.bb_2d.keypoint_1[0] < 0:
                    obj.bb_2d.keypoint_1[0] = 0
                if obj.bb_2d.keypoint_1[1] < 0:
                    obj.bb_2d.keypoint_1[1] = 0
                
                if obj.bb_2d.keypoint_2[0] > input.rgb_image.width-1:
                    obj.bb_2d.keypoint_2[0] = input.rgb_image.width-1
                if obj.bb_2d.keypoint_2[1] > input.rgb_image.height-1:
                    obj.bb_2d.keypoint_2[1] = input.rgb_image.height-1
            resp_msg.module_id = self.module_id
            resp_msg.rgb_image = input.rgb_image
            resp_msg.pointcloud = input.pointcloud
            resp_msg.stamped_pose = input.stamped_pose
            resp_msg.robot_to_map = input.robot_to_map 
            self.detections.publish(resp_msg)
        

        # Segmentation
        # with a:

        #     # Set module id
        #     resp_msg.module_id = self.module_id

        #     # Set reference image
        #     resp_msg.rgb_image = input.rgb_image

        #     # Extract segments of the pointcloud
        #     for obj in resp_msg.objects:
        #         obj.pointcloud.header = input.pointcloud.header
        #         obj.pointcloud.fields = input.pointcloud.fields
        #         obj.pointcloud.is_bigendian = input.pointcloud.is_bigendian
        #         obj.pointcloud.is_dense = input.pointcloud.is_dense

        #         obj.pointcloud.width = int(obj.bb_2d.keypoint_2[0] - obj.bb_2d.keypoint_1[0]+1)
        #         obj.pointcloud= int(obj.bb_2d.keypoint_2[1] - obj.bb_2d.keypoint_1[1]+1)
        #         obj.pointcloud.point_step = input.pointcloud.point_step
        #         obj.pointcloud.row_step = obj.pointcloud.width * obj.pointcloud.point_step
                
        #         buff = ctypes.create_string_buffer(xyzrgba_struct.size * obj.pointcloud.width * obj.pointcloud.height)

        #         offset=0
        #         for h in range(obj.bb_2d.keypoint_1[1],obj.bb_2d.keypoint_2[1]+1):
        #             for w in range(obj.bb_2d.keypoint_1[0],obj.bb_2d.keypoint_2[0]+1):
        #                 xyzrgba_struct.pack_into(
        #                     buff,
        #                     offset,
        #                     *xyzrgba_struct.unpack_from(input.pointcloud.data, (input.pointcloud.row_step * h) + (input.pointcloud.point_step * w))
        #                 )
        #                 offset+=xyzrgba_struct.size

        #         obj.pointcloud.data = buff.raw


        #         self.obj1.publish(obj.pointcloud)


        #     self.detections.publish(resp_msg)
        # self.get_logger().debug(f'Seg time: %.1fms' % a.t)


def main(args=None,detector_class=perception_wrapper,detector_args={'name': 'perception_wrapper'}):

    rclpy.init(args=args)

    detector_obj = detector_class(detector_name=detector_args['name'])

    executor = MultiThreadedExecutor(4)
    executor.add_node(detector_obj)

    while(rclpy.ok()):
        try:
            executor.spin_once()
            if detector_obj.on_process():
                executor.shutdown(0)
                rclpy.shutdown()
                return
        except Exception as exception:
            traceback_logger_node = Node('node_class_traceback_logger')
            traceback_logger_node.get_logger().error(traceback.format_exc())
            raise exception

    detector_obj.destroy_node()
    executor.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()