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
from smap_interfaces.srv import AddClassifier

class classification_wrapper(Node):

    classifier_id = 0
    classes = []
    parameters = []

    def __init__(self,classifier_name='Cl1',classifier_type='object',classifier_architecture='no_defined'):
        super().__init__("smap_classification_{}".format(classifier_name))

        self.get_logger().info("Initializing smap wrapper!")

        self.classifier_name=classifier_name
        
        self.classifier_type=classifier_type

        self.classifier_architecture=classifier_architecture

        self.reentrant_cb_group=ReentrantCallbackGroup()

        self.__ret_valid=10

        self.__shutdown_node=False
        self.add_classifier()

        self.valid_timer=self.create_timer(1.0,self.__validate_classifier)

    def add_classifier(self):
        self.cli = self.create_client(
            AddClassifier,
            "add_classifier"
        )
        ret=10
        self.get_logger().info("smap wrapper wainting for service \'{}\'!".format(self.classifier_name,'add_classifier'))
        while not self.cli.wait_for_service(timeout_sec=1.0):
            if ret==10:
                self.get_logger().warning('Service not available, waiting again...')
            else:
                if ret==-1:
                    self.get_logger().error("Service not available!")
                    self.get_logger().fatal("Destroying node!")
                    self.__shutdown_node=True
                    return
                else:
                    self.get_logger().warning('Service not available [{}/10], waiting again...'.format(10-ret))
            ret=ret-1
        self.get_logger().info("Successful connection to service \'{}\'!".format('add_classifier'))

        req=AddClassifier.Request()
        req.name=self.classifier_name
        req.type=self.classifier_type
        req.architecture=self.classifier_architecture
        self.get_logger().info("Sending request...")
        self.fut_val=self.cli.call_async(req)


    def __validate_classifier(self):
        if self.__shutdown_node==True:
            return
        self.__ret_valid=10
        if self.fut_val.done():
            try:
                resp=self.fut_val.result()
            except Exception as e:
                self.get_logger().error("Add Classifier request faild!")
                self.get_logger().fatal("Destroying node!")
                self.__shutdown_node=True
                return
        else:
            if self.__ret_valid==10:
                self.get_logger().warning('Waiting for service response...')
                self.__ret_valid=self.__ret_valid-1
                return
            if self.__ret_valid==-1:
                self.get_logger().error('No service response!')
                self.get_logger().fatal("Destroying node!")
                self.__shutdown_node=True
                return
            self.get_logger().warning('Waiting for service response [{}/10]...'.format(self.__ret_valid))
            self.__ret_valid=self.__ret_valid-1
            
        self.get_logger().info("Processing service response!")

        if not resp.success:
            self.get_logger().error("Add Classifier request faild!")
            self.get_logger().fatal("Destroying node!")
            self.__shutdown_node=True
            return
        
        self.valid_timer.destroy()
        
        self.classifier_id=resp.classifier_id

        self.initialization()

        self.get_logger().info("smap classifier:")
        self.get_logger().info("\t name: {}".format(self.classifier_name))
        self.get_logger().info("\t id: {}".format(self.classifier_id))
        self.get_logger().info("\t type: {}".format(self.classifier_type))
        self.get_logger().info("\t architecture: {}".format(self.classifier_architecture))
        self.get_logger().info("smap wrapper launched!")
    
    def initialization(self):
        self.subscription=self.create_subscription(SmapData, '/smap/classifiers/data', self.predict, 10,callback_group= self.reentrant_cb_group)
        self.publisher=self.create_publisher(SmapPrediction, '/smap/classifiers/predictions', 10,callback_group= self.reentrant_cb_group)

    def train(self,data):
        pass

    def predict(self,data):
        #print(data)
        msg = SmapPrediction()
        msg.classifier_id=0
        self.publisher.publish(msg)

    def on_process(self): # Pooling
        if self.__shutdown_node:
            return True
        pass

def main(args=None,classifier_class=classification_wrapper,classifier_args={'name': 'classification_wrapper'}):

    rclpy.init(args=args)

    classifier_obj = classifier_class(classifier_name=classifier_args['name'])

    executor = MultiThreadedExecutor(4)
    executor.add_node(classifier_obj)

    while(rclpy.ok()):
        try:
            executor.spin_once()
            if classifier_obj.on_process():
                classifier_obj.destroy_node()
                executor.shutdown(0)
                rclpy.shutdown()
                return
        except Exception as exeption:
            traceback_logger_node = Node('node_class_traceback_logger')
            traceback_logger_node.get_logger().error(traceback.format_exc())
            raise exeption

    classifier_obj.destroy_node()
    executor.shutdown()
    rclpy.shutdown()


#if __name__ == '__main__':
#    main()