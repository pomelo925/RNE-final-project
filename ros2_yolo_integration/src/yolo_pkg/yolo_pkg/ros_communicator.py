from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Imu, Image, CameraInfo
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PointStamped
import cv2
import numpy as np
from std_msgs.msg import Int32

from interfaces.msg import DetectedObject, DetectedObjectList
from interfaces.msg import Point, PointList


class RosCommunicator(Node):
    def __init__(self):
        super().__init__("RosCommunicator")

        # --- Subscriber and Publisher Initialization ---
        self.subscriber_dict = {
            "rgb_compress": {
                "topic": "/camera/image/compressed",
                "msg_type": CompressedImage,
                "callback": self._image_sub_callback,
            },
            "imu": {
                "topic": "/imu/data",
                "msg_type": Imu,
                "callback": self._imu_sub_callback,
            },
            "depth_image_compress": {
                "topic": "/camera/depth/compressed",
                "msg_type": CompressedImage,
                "callback": self._depth_image_compress_sub_callback,
            },
            "depth_image": {
                "topic": "/camera/depth/image_raw",
                "msg_type": Image,
                "callback": self._depth_image_sub_callback,
            },
            "target_label": {
                "topic": "/target_label",
                "msg_type": String,
                "callback": self._target_label_sub_callback,
            },
            # --- MaskedPublisher 相關 ---
            "masked_image": {
                "topic": "/camera/image/compressed",
                "msg_type": CompressedImage,
                "callback": self.image_callback,
            },
        }

        self.publisher_dict = {
            "yolo_image": {
                "topic": "/yolo/detection/compressed",
                "msg_type": CompressedImage,
            },
            "point": {
                "topic": "/yolo/detection/position",
                "msg_type": PointStamped,
            },
            "object_offset": {
                "topic": "/yolo/object/offset",
                "msg_type": String,
            },
            "detection_status": {
                "topic": "/yolo/detection/status",
                "msg_type": Bool,
            },
            "camera_info": {  
                "topic": "/camera/camera_info",
                "msg_type": CameraInfo,
            },
            "detection_list": {
                "topic": "/yolo/detection/list",
                "msg_type": DetectedObjectList,
            },
            # --- MaskedPublisher 相關 ---
            "masked": {
                "topic": "/opencv/image/masked",
                "msg_type": CompressedImage,
            },
            "polygon": {
                "topic": "/opencv/image/polygon",
                "msg_type": CompressedImage,
            },
            "edges": {
                "topic": "opencv/polygon/list",
                "msg_type": PointList,
            },
        }

        # Initialize Subscribers
        self.latest_data = {}
        for key, sub in self.subscriber_dict.items():
            self.latest_data[key] = None
            msg_type = sub["msg_type"]
            topic = sub["topic"]
            callback = sub["callback"]
            self.create_subscription(msg_type, topic, callback, 10)

        # Initialize Publishers
        self.publisher_instances = {}
        for key, pub in self.publisher_dict.items():
            self.publisher_instances[key] = self.create_publisher(
                pub["msg_type"], pub["topic"], 10
            )

        # --- MaskedPublisher 相關參數 ---
        self.rgb_lower = np.array([0, 0, 0])
        self.rgb_upper = np.array([125, 255, 255])
        self.hsv_lower = np.array([90, 0, 0])
        self.hsv_upper = np.array([180, 60, 145])
        self.MIN_AREA = 500
        self.latest_polygon_image = None

    # --- Callback Functions ---
    def _image_sub_callback(self, msg):
        self.latest_data["rgb_compress"] = msg
        # 同步發佈 CameraInfo
        camera_info_msg = self._generate_mock_camera_info(msg)
        self.publish_data("camera_info", camera_info_msg)

    def _imu_sub_callback(self, msg):
        self.latest_data["imu"] = msg

    def _depth_image_sub_callback(self, msg):
        self.latest_data["depth_image"] = msg

    def _depth_image_compress_sub_callback(self, msg):
        self.latest_data["depth_image_compress"] = msg

    def _target_label_sub_callback(self, msg):
        self.latest_data["target_label"] = msg

    # --- Getter Functions ---
    def get_latest_data(self, key):
        return self.latest_data.get(key)

    # --- Publisher Functions ---
    def publish_data(self, key, data):
        try:
            publisher = self.publisher_instances.get(key)
            if publisher:
                publisher.publish(data)
            else:
                self.get_logger().error(f"No publisher found for key: {key}")
        except Exception as e:
            self.get_logger().error(f"Could not publish data for {key}: {e}")

    def publish_detection_list(self, detected_objects):
        msg = DetectedObjectList()
        for obj in detected_objects:
            item = DetectedObject()
            item.label = obj["label"]
            x1, y1, x2, y2 = obj["box"]
            item.cx = int((x1 + x2) // 2)
            item.cy = int((y1 + y2) // 2)
            item.x1 = int(x1)
            item.y1 = int(y1)
            item.x2 = int(x2)
            item.y2 = int(y2)
            msg.objects.append(item)
        self.publish_data("detection_list", msg)

    # --- CameraInfo 模擬產生器 ---
    def _generate_mock_camera_info(self, image_msg):
        camera_info = CameraInfo()
        camera_info.header.stamp = image_msg.header.stamp
        camera_info.header.frame_id = image_msg.header.frame_id
        camera_info.height = 480
        camera_info.width = 640
        camera_info.distortion_model = "plumb_bob"
        camera_info.d = [0.001750, -0.003776, -0.000528, -0.000228, 0.000000]
        camera_info.k = [
            576.83946, 0.0, 319.59192,
            0.0, 577.82786, 238.89255,
            0.0, 0.0, 1.0
        ]
        camera_info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]
        camera_info.p = [
            576.83946, 0.0, 319.59192, 0.0,
            0.0, 577.82786, 238.89255, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        return camera_info

    # --- MaskedPublisher 相關 callback ---
    def rgb_range_callback(self, msg):
        arr = msg.data
        if len(arr) == 6:
            self.rgb_lower = np.array(arr[:3])
            self.rgb_upper = np.array(arr[3:])
            self.get_logger().info(f'Updated RGB: {self.rgb_lower}-{self.rgb_upper}')

    def hsv_range_callback(self, msg):
        arr = msg.data
        if len(arr) == 6:
            self.hsv_lower = np.array(arr[:3])
            self.hsv_upper = np.array(arr[3:])
            self.get_logger().info(f'Updated HSV: {self.hsv_lower}-{self.hsv_upper}')

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return

        # RGB 遮罩
        mask_rgb = cv2.inRange(img, self.rgb_lower, self.rgb_upper)
        # HSV 遮罩
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        # 兩者交集
        mask = cv2.bitwise_and(mask_rgb, mask_hsv)
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        # --- 多邊形處理 ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon_img = masked_img.copy()
        max_edges = 0
        max_approx = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_AREA:
                continue
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            edges = len(approx)
            # 如果邊數大於目前最大邊數，則更新
            if edges > max_edges:
                max_edges = edges
                max_approx = approx
            # 畫多邊形邊界（綠色）
            cv2.polylines(polygon_img, [approx], True, (0, 255, 0), 2)
            # 畫頂點（紅色）
            for pt in approx:
                cv2.circle(polygon_img, tuple(pt[0]), 6, (0, 0, 255), -1)

        # 發布多邊形圖像
        _, buffer_polygon = cv2.imencode('.jpg', polygon_img)
        polygon_msg = CompressedImage()
        polygon_msg.header = msg.header
        polygon_msg.format = "jpeg"
        polygon_msg.data = buffer_polygon.tobytes()
        self.publish_data("polygon", polygon_msg)
        self.latest_polygon_image = polygon_msg

        # 發布所有頂點的 x, y 以及 len
        point_list_msg = PointList()
        if max_approx is not None:
            for pt in max_approx:
                point_msg = Point()
                point_msg.x = int(pt[0][0])
                point_msg.y = int(pt[0][1])
                point_list_msg.points.append(point_msg)
            # 設定 len
            for p in point_list_msg.points:
                p.len = len(point_list_msg.points)
        
        self.publish_data("edges", point_list_msg)

        # 發布 masked image
        _, buffer = cv2.imencode('.jpg', masked_img)
        masked_msg = CompressedImage()
        masked_msg.header = msg.header
        masked_msg.format = "jpeg"
        masked_msg.data = buffer.tobytes()
        self.publish_data("masked", masked_msg)