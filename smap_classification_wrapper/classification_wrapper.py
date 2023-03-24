#!/usr/bin/env python3

#import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from smap_interfaces.msg import SmapData, SmapPrediction


import rclpy
import traceback
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from smap_interfaces.msg import SmapPrediction
from smap_interfaces.srv import AddPerceptionModule

class classification_wrapper(Node):

    module_id = 0
    classes = []
    parameters = []

    device=None
    model=None
    model_file=None
    model_description_file=None
    imgsz=None

    __initialization_state=0
    __state_busy=False

    def __init__(self,detector_name='Cl1',detector_type='object',detector_architecture='no_defined'):
        super().__init__("smap_perception_{}".format(detector_name))

        self.get_logger().info("Initializing smap wrapper...")

        self.detector_name=detector_name
        
        self.detector_type=detector_type

        self.detector_architecture=detector_architecture

        self.reentrant_cb_group=ReentrantCallbackGroup()

        self.__ret_valid=10

        self.__shutdown_node=False

        self.node_timer=self.create_timer(1.0,self.__states_next__)

        self.declare_parameter('model',value=None)

        self.declare_parameter('model_description',value=None)

        self.model_file = self.get_parameter('model').value

        self.model_description_file = self.get_parameter('model_description').value

    def __states_next__(self):
        # Initialization:
        #   __states_next__ -> 0
        #       > setup_detector()
        #           initialize detector object
        #   __states_next__ -> 1
        #       > add_detector()
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
            if self.add_detector(): # Return True when done!
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
        

    def add_detector(self): # Return True when done!
        self.cli = self.create_client(
            AddPerceptionModule,
            "add_perception_module"
        )
        ret=10
        self.get_logger().info("smap wrapper wainting for service \'{}\'".format(self.detector_name,'add_perception_module'))
        while not self.cli.wait_for_service(timeout_sec=1.0):
            if ret==10:
                self.get_logger().warning('Service not available, waiting again...')
            else:
                if ret==-1:
                    self.get_logger().error("Service not available!")
                    self.__shutdown_node=True
                    return False
                else:
                    self.get_logger().warning('Service not available [{}/10], waiting again...'.format(10-ret))
            ret=ret-1
        self.get_logger().info("Successful connection to service \'{}\'.".format('add_perception_module'))

        req=AddPerceptionModule.Request()
        req.name=self.detector_name
        req.type=self.detector_type
        req.architecture=self.detector_architecture
        self.get_logger().info("Sending request...")
        self.fut_val=self.cli.call_async(req)
        return True

    def setup_detector(self): # Return True when done!

        self.get_logger().info("Setting up detector...")
        

        # Check device
        if self._chech_device(): # Return True when an error occurs
            self.get_logger().fatal("Device not defined!")
            self.__shutdown_node=True
            return False
        

        # Load model
        if self._load_model(): # Return True when an error occurs
            self.get_logger().fatal("Model file not defined!")
            self.__shutdown_node=True
            return False
        
        if self._dataloader(): # Return True when an error occurs
            self.get_logger().fatal("Dataloader Error!")
            self.__shutdown_node=True
            return False
        
        # Model Warmup
        self._model_warmup()

        self.get_logger().info('Detector setup complete.')

        return True

    def _chech_device(self): # Return True when an error occurs
        if self.device is None:
            return True
        return False

    def _load_model(self): # Return True when an error occurs
        if (self.model_file is None):
            return True
        if (self.model_description_file is None):
            return True
        self.get_logger().info("Loading Model {}\n{}...".format(self.model_file,self.model_description_file))
        return False
    
    def _model_warmup(self):
        self.get_logger().info('Model warming up...')

    def _dataloader(self):
        self.get_logger().info('Initializing dataloader...')
        return False

    def __validate_detector__(self): # Return True when done!
        #self.__ret_valid=10
        if self.fut_val.done():
            try:
                resp=self.fut_val.result()
            except Exception as e:
                self.get_logger().error("Add detector request faild!")
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
            self.get_logger().error("Add detector request faild!")
            self.__shutdown_node=True
            return False
        
        self.module_id=resp.module_id

        return True
    
    def initialization(self):
        self.get_logger().info("Initializing topics")
        self.subscription=self.create_subscription(SmapData, '/smap/sampler/data', self.predict, 10,callback_group= self.reentrant_cb_group)
        self.publisher=self.create_publisher(SmapPrediction, '/smap/perception/predictions', 10,callback_group= self.reentrant_cb_group)
        return True

    def train(self,data):
        pass

    def predict(self,data):
        #print(data)
        msg = SmapPrediction()
        msg.module_id=0
        self.publisher.publish(msg)

    def _shutdown(self):
        self.get_logger().fatal("Shutting down node!")
        #self.destroy_node()

    def on_process(self): # Pooling
        if self.__shutdown_node:
            self._shutdown()
            return True

def main(args=None,detector_class=classification_wrapper,detector_args={'name': 'classification_wrapper'}):

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
        except Exception as exeption:
            traceback_logger_node = Node('node_class_traceback_logger')
            traceback_logger_node.get_logger().error(traceback.format_exc())
            raise exeption

    detector_obj.destroy_node()
    executor.shutdown()
    rclpy.shutdown()


#if __name__ == '__main__':
#    main()