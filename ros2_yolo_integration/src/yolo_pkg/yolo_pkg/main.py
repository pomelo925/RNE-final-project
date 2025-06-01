import rclpy
from rclpy.executors import MultiThreadedExecutor
from yolo_pkg.ros_communicator import RosCommunicator
from yolo_pkg.image_processor import ImageProcessor
from yolo_pkg.yolo_depth_extractor import YoloDepthExtractor
from yolo_pkg.yolo_bounding_box import YoloBoundingBox
from yolo_pkg.boundingbox_visaulizer import BoundingBoxVisualizer
from yolo_pkg.camera_geometry import CameraGeometry
import threading
from std_msgs.msg import String  # Import String message type
from yolo_pkg.load_params import LoadParams


def menu():
    print("Select mode:")
    print("1: Draw 3D BBox with screenshot.")
    print("2: Draw 3D BBox without screenshot.")
    print("3: Draw 2D BBox with screenshot.")
    print("4: Draw 2D BBox without screenshot.")
    print("5: segmentation.")
    print("6: 5 fps screenshots.")
    print("Press Ctrl+C to exit.")

    user_input = input("Enter your choice (1/6): ")
    return user_input


def main():
    """
    Main function to initialize the node and run the bounding box visualizer.
    """
    load_params = LoadParams("yolo_pkg")
    ros_communicator, executor, ros_thread = _init_ros_node()
    image_processor = ImageProcessor(ros_communicator, load_params)
    yolo_boundingbox = YoloBoundingBox(image_processor, load_params)
    yolo_depth_extractor = YoloDepthExtractor(
        yolo_boundingbox, image_processor, ros_communicator
    )
    boundingbox_visualizer = BoundingBoxVisualizer(
        image_processor, yolo_boundingbox, ros_communicator
    )
    camera_geometry = CameraGeometry(yolo_depth_extractor)

    user_input = menu()

    try:
        while True:
            if user_input == "1":
                # 3D BBox with screenshot
                offsets_3d = camera_geometry.calculate_offset_from_crosshair_2d()
                boundingbox_visualizer.draw_bounding_boxes(
                    draw_crosshair=True,
                    screenshot=True,
                    segmentation_status=False,
                    bounding_status=True,
                    offsets_3d_json=offsets_3d,
                )
                offset_msg = String()
                offset_msg.data = offsets_3d
                ros_communicator.publish_data("object_offset", offset_msg)

            elif user_input == "2":
                # 3D BBox without screenshot
                offsets_3d = camera_geometry.calculate_offset_from_crosshair_2d()
                boundingbox_visualizer.draw_bounding_boxes(
                    draw_crosshair=True,
                    screenshot=False,
                    segmentation_status=False,
                    bounding_status=True,
                    offsets_3d_json=offsets_3d,
                )
                offset_msg = String()
                offset_msg.data = offsets_3d
                ros_communicator.publish_data("object_offset", offset_msg)

            elif user_input == "3":
                # 2D BBox with screenshot
                boundingbox_visualizer.draw_bounding_boxes(
                    draw_crosshair=True,
                    screenshot=True,
                    segmentation_status=False,
                    bounding_status=True,
                    log=False,
                )

            elif user_input == "4":
                # 2D BBox without screenshot
                boundingbox_visualizer.draw_bounding_boxes(
                    draw_crosshair=True,
                    screenshot=False,
                    segmentation_status=False,
                    bounding_status=True,
                    log=True,
                )

            elif user_input == "5":
                # segmentation
                boundingbox_visualizer.draw_bounding_boxes(
                    draw_crosshair=True,
                    screenshot=False,
                    segmentation_status=True,
                    bounding_status=False,
                    log=False,
                )

            elif user_input == "6":
                # 5 fps screenshots
                boundingbox_visualizer.draw_bounding_boxes(
                    draw_crosshair=True,
                    screenshot=False,
                    segmentation_status=False,
                    bounding_status=True,
                    log=False,
                )
                boundingbox_visualizer.save_fps_screenshot()

            else:
                print("Invalid input.")

            # Example action for yolo_depth_extractor (can be removed if not needed)
            # depth_data = yolo_depth_extractor.get_yolo_object_depth()
            # print(f"Object Depth: {depth_data}")

    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    finally:
        # Shut down the executor and ROS
        executor.shutdown()
        rclpy.shutdown()
        ros_thread.join()


def _init_ros_node():
    """
    Initialize the ROS 2 node with MultiThreadedExecutor for efficient handling of multiple subscribers.
    """
    rclpy.init()
    node = RosCommunicator()  # Initialize the ROS node
    executor = MultiThreadedExecutor()  # Use MultiThreadedExecutor
    executor.add_node(node)  # Add the node to the executor
    thread = threading.Thread(
        target=executor.spin
    )  # Start the executor in a separate thread
    thread.start()
    return node, executor, thread  # Return the node, executor, and thread


if __name__ == "__main__":
    main()
