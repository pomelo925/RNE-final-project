import subprocess
import os
import yaml
from interfaces.msg import DetectedObjectList

from pros_car_py.nav2_utils import (
    get_yaw_from_quaternion,
    get_direction_vector,
    get_angle_to_target,
    calculate_angle_point,
    cal_distance,
)
import time


class Nav2Processing:
    def __init__(self, ros_communicator, data_processor):
        self.ros_communicator = ros_communicator
        self.data_processor = data_processor
        self.finishFlag = False
        self.global_plan_msg = None
        self.index = 0
        self.index_length = 0
        self.recordFlag = 0
        self.goal_published_flag = False
        self.masked_publisher_started = False
        self._state_machine = self._load_state_machine()
        self._current_state = "START"

    def _load_state_machine(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "config", "states.yaml"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def reset_nav_process(self):
        self.finishFlag = False
        self.recordFlag = 0
        self.goal_published_flag = False

    def finish_nav_process(self):
        self.finishFlag = True
        self.recordFlag = 1

    def get_finish_flag(self):
        return self.finishFlag

    def get_action_from_nav2_plan(self, goal_coordinates=None):
        if goal_coordinates is not None and not self.goal_published_flag:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True
        orientation_points, coordinates = (
            self.data_processor.get_processed_received_global_plan()
        )
        action_key = "STOP"
        if not orientation_points or not coordinates:
            action_key = "STOP"
        else:
            try:
                z, w = orientation_points[0]
                plan_yaw = get_yaw_from_quaternion(z, w)
                car_position, car_orientation = (
                    self.data_processor.get_processed_amcl_pose()
                )
                car_orientation_z, car_orientation_w = (
                    car_orientation[2],
                    car_orientation[3],
                )
                goal_position = self.ros_communicator.get_latest_goal()
                target_distance = cal_distance(car_position, goal_position)
                if target_distance < 0.5:
                    action_key = "STOP"
                    self.finishFlag = True
                else:
                    car_yaw = get_yaw_from_quaternion(
                        car_orientation_z, car_orientation_w
                    )
                    diff_angle = (plan_yaw - car_yaw) % 360.0
                    if diff_angle < 30.0 or (diff_angle > 330 and diff_angle < 360):
                        action_key = "FORWARD"
                    elif diff_angle > 30.0 and diff_angle < 180.0:
                        action_key = "COUNTERCLOCKWISE_ROTATION"
                    elif diff_angle > 180.0 and diff_angle < 330.0:
                        action_key = "CLOCKWISE_ROTATION"
                    else:
                        action_key = "STOP"
            except:
                action_key = "STOP"
        return action_key

    def get_action_from_nav2_plan_no_dynamic_p_2_p(self, goal_coordinates=None):
        if goal_coordinates is not None and not self.goal_published_flag:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True

        # 只抓第一次路径
        if self.recordFlag == 0:
            if not self.check_data_availability():
                return "STOP"
            else:
                print("Get first path")
                self.index = 0
                self.global_plan_msg = (
                    self.data_processor.get_processed_received_global_plan_no_dynamic()
                )
                self.recordFlag = 1
                action_key = "STOP"

        car_position, car_orientation = self.data_processor.get_processed_amcl_pose()

        goal_position = self.ros_communicator.get_latest_goal()
        target_distance = cal_distance(car_position, goal_position)

        # 抓最近的物標(可調距離)
        target_x, target_y = self.get_next_target_point(car_position)

        if target_x is None or target_distance < 0.5:
            self.ros_communicator.reset_nav2()
            self.finish_nav_process()
            return "STOP"

        # 計算角度誤差
        diff_angle = self.calculate_diff_angle(
            car_position, car_orientation, target_x, target_y
        )
        if diff_angle < 20 and diff_angle > -20:
            action_key = "FORWARD"
        elif diff_angle < -20 and diff_angle > -180:
            action_key = "CLOCKWISE_ROTATION"
        elif diff_angle > 20 and diff_angle < 180:
            action_key = "COUNTERCLOCKWISE_ROTATION"
        return action_key

    def check_data_availability(self):
        return (
            self.data_processor.get_processed_received_global_plan_no_dynamic()
            and self.data_processor.get_processed_amcl_pose()
            and self.ros_communicator.get_latest_goal()
        )

    def get_next_target_point(self, car_position, min_required_distance=0.5):
        """
        選擇距離車輛 min_required_distance 以上最短路徑然後返回 target_x, target_y
        """
        if self.global_plan_msg is None or self.global_plan_msg.poses is None:
            print("Error: global_plan_msg is None or poses is missing!")
            return None, None
        while self.index < len(self.global_plan_msg.poses) - 1:
            target_x = self.global_plan_msg.poses[self.index].pose.position.x
            target_y = self.global_plan_msg.poses[self.index].pose.position.y
            distance_to_target = cal_distance(car_position, (target_x, target_y))

            if distance_to_target < min_required_distance:
                self.index += 1
            else:
                self.ros_communicator.publish_selected_target_marker(
                    x=target_x, y=target_y
                )
                return target_x, target_y

        return None, None

    def calculate_diff_angle(self, car_position, car_orientation, target_x, target_y):
        target_pos = [target_x, target_y]
        diff_angle = calculate_angle_point(
            car_orientation[2], car_orientation[3], car_position[:2], target_pos
        )
        return diff_angle

    def filter_negative_one(self, depth_list):
        return [depth for depth in depth_list if depth != -1.0]

    def camera_nav(self):
        """
        YOLO 目標資訊 (yolo_target_info) 說明：

        - 索引 0 (index 0)：
            - 表示是否成功偵測到目標
            - 0：未偵測到目標
            - 1：成功偵測到目標

        - 索引 1 (index 1)：
            - 目標的深度距離 (與相機的距離，單位為公尺)，如果沒偵測到目標就回傳 0
            - 與目標過近時(大約 40 公分以內)會回傳 -1

        - 索引 2 (index 2)：
            - 目標相對於畫面正中心的像素偏移量
            - 若目標位於畫面中心右側，數值為正
            - 若目標位於畫面中心左側，數值為負
            - 若沒有目標則回傳 0

        畫面 n 個等分點深度 (camera_multi_depth) 說明 :

        - 儲存相機畫面中央高度上 n 個等距水平點的深度值。
        - 若距離過遠、過近（小於 40 公分）或是實體相機有時候深度會出一些問題，則該點的深度值將設定為 -1。
        """
        yolo_target_info = self.data_processor.get_yolo_target_info()
        camera_multi_depth = self.data_processor.get_camera_x_multi_depth()
        if camera_multi_depth == None or yolo_target_info == None:
            return "STOP"

        camera_forward_depth = self.filter_negative_one(camera_multi_depth[7:13])
        camera_left_depth = self.filter_negative_one(camera_multi_depth[0:7])
        camera_right_depth = self.filter_negative_one(camera_multi_depth[13:20])

        action = "STOP"
        limit_distance = 0.7

        if all(depth > limit_distance for depth in camera_forward_depth):
            if yolo_target_info[0] == 1:
                if yolo_target_info[2] > 200.0:
                    action = "CLOCKWISE_ROTATION_SLOW"
                elif yolo_target_info[2] < -200.0:
                    action = "COUNTERCLOCKWISE_ROTATION_SLOW"
                else:
                    if yolo_target_info[1] < 0.8:
                        action = "STOP"
                    else:
                        action = "FORWARD_SLOW"
            else:
                action = "FORWARD"
        elif any(depth < limit_distance for depth in camera_left_depth):
            action = "CLOCKWISE_ROTATION"
        elif any(depth < limit_distance for depth in camera_right_depth):
            action = "COUNTERCLOCKWISE_ROTATION"
        return action

    def RGBDcamera_nav_unity(self):
        """
        YOLO 目標資訊 (yolo_target_info) 說明：

        - 索引 0 (index 0)：
            - 表示是否成功偵測到目標
            - 0：未偵測到目標
            - 1：成功偵測到目標

        - 索引 1 (index 1)：
            - 目標的深度距離 (與相機的距離，單位為公尺)，如果沒偵測到目標就回傳 0
            - 與目標過近時(大約 40 公分以內)會回傳 -1

        - 索引 2 (index 2)：
            - 目標相對於畫面正中心的像素偏移量
            - 若目標位於畫面中心右側，數值為正
            - 若目標位於畫面中心左側，數值為負
            - 若沒有目標則回傳 0

        畫面 n 個等分點深度 (camera_multi_depth) 說明 :

        - 儲存相機畫面中央高度上 n 個等距水平點的深度值。
        - 若距離過遠、過近（小於 40 公分）或是實體相機有時候深度會出一些問題，則該點的深度值將設定為 -1。
        """
        yolo_target_info = self.data_processor.get_yolo_target_info()
        camera_multi_depth = self.data_processor.get_camera_x_multi_depth()
        yolo_target_info[1] *= 100.0
        camera_multi_depth = list(
            map(lambda x: x * 100.0, self.data_processor.get_camera_x_multi_depth())
        )

        if camera_multi_depth == None or yolo_target_info == None:
            return "STOP"

        camera_forward_depth = self.filter_negative_one(camera_multi_depth[7:13])
        camera_left_depth = self.filter_negative_one(camera_multi_depth[0:7])
        camera_right_depth = self.filter_negative_one(camera_multi_depth[13:20])
        action = "STOP"
        limit_distance = 10.0
        print(yolo_target_info[1])
        if all(depth > limit_distance for depth in camera_forward_depth):
            if yolo_target_info[0] == 1:
                if yolo_target_info[2] > 200.0:
                    action = "CLOCKWISE_ROTATION_SLOW"
                elif yolo_target_info[2] < -200.0:
                    action = "COUNTERCLOCKWISE_ROTATION_SLOW"
                else:
                    if yolo_target_info[1] < 2.0:
                        action = "STOP"
                    else:
                        action = "FORWARD_SLOW"
            else:
                action = "FORWARD"
        elif any(depth < limit_distance for depth in camera_left_depth):
            action = "CLOCKWISE_ROTATION"
        elif any(depth < limit_distance for depth in camera_right_depth):
            action = "COUNTERCLOCKWISE_ROTATION"
        return action

    def print_status(self, mode: str, info: str):
        os.system("cls" if os.name == "nt" else "clear")  # 清版
        print("[MODE]\t\t|\t[INFO]")
        print(f"{mode.upper()}\t\t|\t{info}")

    def check_and_handle_pikachu_interrupt(self, pikachu_area_threshold=0.3):
        action, info = self.track_pikachu(pikachu_area_threshold=pikachu_area_threshold)
        if info["found_pikachu"]:
            self.print_status(action, f"bbox_area_ratio: {info['bbox_area_ratio']}")
            self.set_main_state("DONE")  # 或你想跳到的狀態
            self.ros_communicator.publish_car_control(action)
            return True
        return False

    def RGBcam_nav_unity_living_room_fixed(self):
        # 圖像設定
        image_width = 640
        image_height = 480
        image_area = image_width * image_height
        image_center_x = image_width // 2

        # 控制參數
        bbox_center_range_ratio = 0.06  # 目標邊界框中心範圍比例
        pikachu_area_threshold = 0.2    # 目標面積比例閾值
        obstacle_x_offset = 30          # 障礙物偏移量

        # 動作參數
        SEARCH_ROTATION_ACTION = "COUNTERCLOCKWISE_ROTATION"
        ALIGN_ROTATE_LEFT_ACTION = "COUNTERCLOCKWISE_ROTATION_SLOW"
        ALIGN_ROTATE_RIGHT_ACTION = "CLOCKWISE_ROTATION_SLOW"

        # 取得最新的 YOLO 偵測結果
        detected_list = self.ros_communicator.get_latest_yolo_detection_list()

        # 檢查偵測結果是否有效
        if not detected_list or not hasattr(detected_list, "objects"):
            self.print_status(SEARCH_ROTATION_ACTION, "no detections")
            return SEARCH_ROTATION_ACTION

        # 檢查是否有偵測到 "pikachu" 物件
        pikachu_objects = [obj for obj in detected_list.objects if obj.label == "pikachu"]
        if not pikachu_objects:
            self.print_status(SEARCH_ROTATION_ACTION, "pikachu not found")
            return SEARCH_ROTATION_ACTION

        # 找到最大的 "pikachu" 物件
        target = max(pikachu_objects, key=lambda obj: (obj.x2 - obj.x1) * (obj.y2 - obj.y1))
        bbox_x1 = target.x1
        bbox_x2 = target.x2
        center_range_px = int(image_width * bbox_center_range_ratio)

        # 檢查目標邊界框是否在畫面中心範圍內
        if not (bbox_x1 <= image_center_x + center_range_px and bbox_x2 >= image_center_x - center_range_px):
            # 根據目標中心位置微調方向
            if target.cx < image_center_x:
                self.print_status(ALIGN_ROTATE_LEFT_ACTION, "align left")
                return ALIGN_ROTATE_LEFT_ACTION
            else:
                self.print_status(ALIGN_ROTATE_RIGHT_ACTION, "align right")
                return ALIGN_ROTATE_RIGHT_ACTION

        # 計算目標面積比例
        bbox_area = (target.x2 - target.x1) * (target.y2 - target.y1)
        bbox_area_ratio = bbox_area / image_area
        self.print_status("FORWARD", f"bbox_area_ratio: {bbox_area_ratio:.2f}")

        # 若面積比例小於閾值，檢查障礙物
        if bbox_area_ratio < pikachu_area_threshold:
            obstacle_objects = [
                obj for obj in detected_list.objects
                if obj.label != "pikachu" and (
                    abs(obj.x1 - image_center_x) < obstacle_x_offset or
                    abs(obj.x2 - image_center_x) < obstacle_x_offset
                )
            ]

            # 若有障礙物，尋找最近的障礙物邊緣
            if obstacle_objects:
                def closest_edge_to_center(obj):
                    return min(abs(obj.x1 - image_center_x), abs(obj.x2 - image_center_x))

                nearest_obstacle = min(obstacle_objects, key=closest_edge_to_center) 

                # 根據最近障礙物的邊緣位置決定旋轉方向
                if abs(nearest_obstacle.x1 - image_center_x) < abs(nearest_obstacle.x2 - image_center_x):
                    self.print_status(ALIGN_ROTATE_RIGHT_ACTION, "avoid obstacle: left edge")
                    return ALIGN_ROTATE_RIGHT_ACTION
                else:
                    self.print_status(ALIGN_ROTATE_LEFT_ACTION, "avoid obstacle: right edge")
                    return ALIGN_ROTATE_LEFT_ACTION

            return "FORWARD"

        else:
            self.print_status("STOP", f"bbox_area_ratio: {bbox_area_ratio:.2f} (threshold reached)")
            return "STOP"

    def RGBcam_nav_unity_living_room_random(self):
        # 圖像設定
        image_width = 640
        image_height = 480
        image_area = image_width * image_height
        image_center_x = image_width // 2

        # 控制參數
        bbox_center_range_ratio = 0.06  # 目標邊界框中心範圍比例
        pikachu_area_threshold = 0.06    # 目標面積比例閾值
        obstacle_x_offset = 30          # 障礙物偏移量

        # 動作參數
        SEARCH_ROTATION_ACTION = "COUNTERCLOCKWISE_ROTATION"
        ALIGN_ROTATE_LEFT_ACTION = "COUNTERCLOCKWISE_ROTATION_SLOW"
        ALIGN_ROTATE_RIGHT_ACTION = "CLOCKWISE_ROTATION_SLOW"

        # 取得最新的 YOLO 偵測結果
        detected_list = self.ros_communicator.get_latest_yolo_detection_list()

        # 檢查偵測結果是否有效
        if not detected_list or not hasattr(detected_list, "objects"):
            self.print_status(SEARCH_ROTATION_ACTION, "no detections")
            return SEARCH_ROTATION_ACTION

        # 檢查是否有偵測到 "pikachu" 物件
        pikachu_objects = [obj for obj in detected_list.objects if obj.label == "pikachu"]
        if not pikachu_objects:
            self.print_status(SEARCH_ROTATION_ACTION, "pikachu not found")
            return SEARCH_ROTATION_ACTION

        # 找到最大的 "pikachu" 物件
        target = max(pikachu_objects, key=lambda obj: (obj.x2 - obj.x1) * (obj.y2 - obj.y1))
        bbox_x1 = target.x1
        bbox_x2 = target.x2
        center_range_px = int(image_width * bbox_center_range_ratio)

        # 檢查目標邊界框是否在畫面中心範圍內
        if not (bbox_x1 <= image_center_x + center_range_px and bbox_x2 >= image_center_x - center_range_px):
            # 根據目標中心位置微調方向
            if target.cx < image_center_x:
                self.print_status(ALIGN_ROTATE_LEFT_ACTION, "align left")
                return ALIGN_ROTATE_LEFT_ACTION
            else:
                self.print_status(ALIGN_ROTATE_RIGHT_ACTION, "align right")
                return ALIGN_ROTATE_RIGHT_ACTION

        # 計算目標面積比例
        bbox_area = (target.x2 - target.x1) * (target.y2 - target.y1)
        bbox_area_ratio = bbox_area / image_area
        self.print_status("FORWARD", f"bbox_area_ratio: {bbox_area_ratio:.2f}")

        # 若面積比例小於閾值，檢查障礙物
        if bbox_area_ratio < pikachu_area_threshold:
            obstacle_objects = [
                obj for obj in detected_list.objects
                if obj.label != "pikachu" and (
                    abs(obj.x1 - image_center_x) < obstacle_x_offset or
                    abs(obj.x2 - image_center_x) < obstacle_x_offset
                )
            ]

            # 若有障礙物，尋找最近的障礙物邊緣
            if obstacle_objects:
                def closest_edge_to_center(obj):
                    return min(abs(obj.x1 - image_center_x), abs(obj.x2 - image_center_x))

                nearest_obstacle = min(obstacle_objects, key=closest_edge_to_center) 

                # 根據最近障礙物的邊緣位置決定旋轉方向
                if abs(nearest_obstacle.x1 - image_center_x) < abs(nearest_obstacle.x2 - image_center_x):
                    self.print_status(ALIGN_ROTATE_RIGHT_ACTION, "avoid obstacle: left edge")
                    return ALIGN_ROTATE_RIGHT_ACTION
                else:
                    self.print_status(ALIGN_ROTATE_LEFT_ACTION, "avoid obstacle: right edge")
                    return ALIGN_ROTATE_LEFT_ACTION

            return "FORWARD"

        else:
            self.print_status("STOP", f"bbox_area_ratio: {bbox_area_ratio:.2f} (threshold reached)")
            return "STOP"


    def get_current_state(self):
        return self._current_state

    def set_current_state(self, state_name):
        self._current_state = state_name

    def reset_state_machine(self):
        if self._state_machine['states']:
            self._current_state = self._state_machine['states'][0]['name']

    ### --- 車輛控制 --- #####
    def move(self, action_or_shortcut: str, duration: float):
        """
        控制車輛執行指定動作或簡寫指令，持續 duration 秒。
        支援簡寫: F, R, L, B, S
        若偵測到皮卡丘則立即中斷。
        """
        MOVE_SHORTCUT_MAP = {
            "F": "FORWARD",
            "R": "CLOCKWISE_ROTATION_MEDIAN",
            "L": "COUNTERCLOCKWISE_ROTATION_MEDIAN",
            "B": "BACKWARD",
            "S": "STOP"
        }
        action = MOVE_SHORTCUT_MAP.get(action_or_shortcut.upper(), action_or_shortcut)
        self.ros_communicator.publish_car_control(action)
        start_time = time.time()
        while time.time() - start_time < duration:
            # 每 0.1 秒檢查一次
            if self.check_and_handle_pikachu_interrupt():
                self.ros_communicator.publish_car_control("STOP")
                return "PIKACHU_INTERRUPT"
            time.sleep(0.1)
        self.ros_communicator.publish_car_control("STOP")
        return "DONE"

    
    def track_pikachu(self, pikachu_area_threshold=0.06):
        """
        共用追蹤 pikachu 的邏輯，回傳 (action, info_dict)
        info_dict 內容包含：bbox_area_ratio, align_action, obstacle_action, found_pikachu
        """
        image_width = 640
        image_height = 480
        image_area = image_width * image_height
        image_center_x = image_width // 2
        bbox_center_range_ratio = 0.06
        obstacle_x_offset = 30

        SEARCH_ROTATION_ACTION = "COUNTERCLOCKWISE_ROTATION"
        ALIGN_ROTATE_LEFT_ACTION = "COUNTERCLOCKWISE_ROTATION_SLOW"
        ALIGN_ROTATE_RIGHT_ACTION = "CLOCKWISE_ROTATION_SLOW"

        detected_list = self.ros_communicator.get_latest_yolo_detection_list()
        info = {
            "bbox_area_ratio": None,
            "align_action": None,
            "obstacle_action": None,
            "found_pikachu": False,
        }

        if not detected_list or not hasattr(detected_list, "objects"):
            return SEARCH_ROTATION_ACTION, info

        pikachu_objects = [obj for obj in detected_list.objects if obj.label == "pikachu"]
        if not pikachu_objects:
            return SEARCH_ROTATION_ACTION, info

        info["found_pikachu"] = True
        target = max(pikachu_objects, key=lambda obj: (obj.x2 - obj.x1) * (obj.y2 - obj.y1))
        bbox_x1 = target.x1
        bbox_x2 = target.x2
        center_range_px = int(image_width * bbox_center_range_ratio)

        # 檢查目標邊界框是否在畫面中心範圍內
        if not (bbox_x1 <= image_center_x + center_range_px and bbox_x2 >= image_center_x - center_range_px):
            if target.cx < image_center_x:
                info["align_action"] = ALIGN_ROTATE_LEFT_ACTION
                return ALIGN_ROTATE_LEFT_ACTION, info
            else:
                info["align_action"] = ALIGN_ROTATE_RIGHT_ACTION
                return ALIGN_ROTATE_RIGHT_ACTION, info

        bbox_area = (target.x2 - target.x1) * (target.y2 - target.y1)
        bbox_area_ratio = bbox_area / image_area
        info["bbox_area_ratio"] = bbox_area_ratio

        if bbox_area_ratio < pikachu_area_threshold:
            obstacle_objects = [
                obj for obj in detected_list.objects
                if obj.label != "pikachu" and (
                    abs(obj.x1 - image_center_x) < obstacle_x_offset or
                    abs(obj.x2 - image_center_x) < obstacle_x_offset
                )
            ]
            if obstacle_objects:
                def closest_edge_to_center(obj):
                    return min(abs(obj.x1 - image_center_x), abs(obj.x2 - image_center_x))
                nearest_obstacle = min(obstacle_objects, key=closest_edge_to_center)
                if abs(nearest_obstacle.x1 - image_center_x) < abs(nearest_obstacle.x2 - image_center_x):
                    info["obstacle_action"] = ALIGN_ROTATE_RIGHT_ACTION
                    return ALIGN_ROTATE_RIGHT_ACTION, info
                else:
                    info["obstacle_action"] = ALIGN_ROTATE_LEFT_ACTION
                    return ALIGN_ROTATE_LEFT_ACTION, info
            return "FORWARD", info
        else:
            return "STOP", info

    def RGBcam_nav_unity_pikachu(self):
        action, info = self.track_pikachu(pikachu_area_threshold=0.06)
        # 可根據 info 印出 debug 訊息
        self.print_status(action, f"bbox_area_ratio: {info['bbox_area_ratio']}")
        while ( not info["found_pikachu"]):
            self.move("L", 0.1)
            self.move("F", 8)
            self.move("R", 0.4)
            self.move("F", 8)
            self.move("R", 0.1)
            self.move("F", 8)
            self.move("L", 0.1)
            self.move("F", 8)
            self.move("L", 0.1)
        return action

    def RGBcam_nav_unity_door_random(self):
        # 1. 若有 pikachu，直接 track_pikachu
        action, info = self.track_pikachu(pikachu_area_threshold=0.3)
        if info["found_pikachu"]:
            self.print_status(action, f"bbox_area_ratio: {info['bbox_area_ratio']}")
            return action

        # 2. State machine
        main_state = self.get_main_state()
        state_func = getattr(self, f"on_enter_{main_state}", None)
        if state_func is not None:
            action = state_func()
        else:
            action = "STOP"
        return action
    
    ### === 狀態機內部呼叫函數 === ###

    def get_next_main_state(self, current):
        order = ['START', 'MID1', 'MID2', 'END']
        idx = order.index(current)
        return order[idx + 1] if idx + 1 < len(order) else 'END'

    def get_main_state(self):
        return getattr(self, "_main_state", "START")

    def set_main_state(self, state):
        self._main_state = state

    def get_sub_state(self):
        return getattr(self, "_sub_state", 0)

    def set_sub_state(self, idx):
        self._sub_state = idx

    def step_state_machine(self):
        """
        執行目前 main_state 的 sub_state，並根據回傳值決定是否進入下一 sub_state 或 main_state
        """
        main_state = self.get_main_state()
        sub_state_idx = self.get_sub_state()
        sub_states = self.main_state_sequences.get(main_state, [])
        if sub_state_idx >= len(sub_states):
            # 進入下一個 main_state
            next_main = self.get_next_main_state(main_state)
            self.set_main_state(next_main)
            self.set_sub_state(0)
            return
        sub_state_name = sub_states[sub_state_idx]
        func = getattr(self, sub_state_name, None)
        if func:
            finished = func()
            if finished:
                self.set_sub_state(sub_state_idx + 1)

    ### === 大狀態與子狀態機架構 === ###

    def run_state_machine(self):
        """
        執行目前大狀態（on_enter_xxx），大狀態 function 內可直接呼叫 self.run_sub_state_machine([sub1, sub2, ...])
        """
        main_state = self.get_main_state()
        func = getattr(self, f"on_enter_{main_state}", None)
        if func:
            return func()
        else:
            return "STOP"

    def run_sub_state_machine(self, sub_state_list):
        """
        執行一組子狀態序列，依序呼叫 function，遇到未完成的子狀態即停止，回傳該子狀態的結果
        子狀態 function 必須回傳 True(完成) 或 False(未完成)
        """
        if not hasattr(self, "_sub_state_idx"):
            self._sub_state_idx = 0
        while self._sub_state_idx < len(sub_state_list):
            sub_func_name = sub_state_list[self._sub_state_idx]
            func = getattr(self, sub_func_name, None)
            if func:
                # print(f"Running sub-state: {sub_func_name}")
                finished = func()
                if finished:
                    self._sub_state_idx += 1
                    continue
                else:
                    return
            else:
                self._sub_state_idx += 1
        self._sub_state_idx = 0
        return True

    ### --- 狀態機條件判斷庫 --- ###
    def polygon_exist(self):
        """
        檢查是否有多邊形存在
        """
        polygon = self.ros_communicator.get_latest_polygon_list()
        return polygon is not None and len(polygon.points) > 0
    

    def get_polygon_type(self):
        """
        獲取目前 polygon 的類型：
        - "NONE"：沒有 polygon 存在
        - "MULTIPLE"：多個 polygon 存在
        - "CONVEX"：單個凸多邊形
        - "CONCAVE_14"：單個凹多邊形，類型 14
        - "CONCAVE_23"：單個凹多邊形，類型 23 
        """
        if not self.polygon_exist():
            return "NONE"
        
        polygon = self.ros_communicator.get_latest_polygon_list()
        if polygon.num > 1:
            return "MULTIPLE"
        

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        def is_convex(points):
            n = len(points)
            if n < 3:
                return False
            sign = None
            for i in range(n):
                o, a, b = points[i], points[(i + 1) % n], points[(i + 2) % n]
                cp = cross(o, a, b)
                if cp != 0:
                    if sign is None:
                        sign = cp > 0
                    elif (cp > 0) != sign:
                        return False
            return True
        
        points = polygon.points[:polygon.len[0]]
        coords = [(p.x, p.y) for p in points]

        if is_convex(coords):
            return "CONVEX"
        else:
            return "CONCAVE_14" if len(coords) <= 7 else "CONCAVE_23"
    

    ### --- 子狀態 function --- ###
    def F2N(self):
        """
        往前走，直至看不見任何 polygon。
        """
        if(self.polygon_exist()):
            self.ros_communicator.publish_car_control("FORWARD")
            return False
        
        self.ros_communicator.publish_car_control("STOP")
        return True

    def L2F(self):
        """
        左轉，直至 polygon 存在頂點佔據畫面最右側。
        """
        X_TOLERANCE = 1
        polygon = self.ros_communicator.get_latest_polygon_list()
        if not polygon or not polygon.points:
            self.ros_communicator.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            return False

        for point in polygon.points:
            if point.x >= 639 - X_TOLERANCE:
                self.ros_communicator.publish_car_control("STOP")
                return True

        self.ros_communicator.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
        return False

    def R2F(self):
        """
        右轉，直至 polygon 存在頂點佔據畫面最左側。
        """
        X_TOLERANCE = 1
        polygon = self.ros_communicator.get_latest_polygon_list()
        if not polygon or not polygon.points:
            self.ros_communicator.publish_car_control("CLOCKWISE_ROTATION_SLOW")
            return False

        for point in polygon.points:
            if point.x <= 1 + X_TOLERANCE:
                self.ros_communicator.publish_car_control("STOP")
                return True

        self.ros_communicator.publish_car_control("CLOCKWISE_ROTATION_SLOW")
        return False
    
    def L2N(self):
        """
        左轉，直至不存在任何 polygon。
        """
        if self.polygon_exist():
            self.ros_communicator.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            return False

        self.ros_communicator.publish_car_control("STOP")
        return True

    def R2N(self):
        """
        右轉，直至不存在任何 polygon。
        """
        if self.polygon_exist():
            self.ros_communicator.publish_car_control("CLOCKWISE_ROTATION_SLOW")
            return False

        self.ros_communicator.publish_car_control("STOP")
        return True
    
    def R2H(self):
        """
        右轉，直至 polygon 存在頂點佔據畫面左半部。
        """
        X_TOLERANCE = 0
        polygon = self.ros_communicator.get_latest_polygon_list()
        if not polygon or not polygon.points:
            self.ros_communicator.publish_car_control("CLOCKWISE_ROTATION_SLOW")
            return False

        for point in polygon.points:
            if point.x <= 319 + X_TOLERANCE:
                self.ros_communicator.publish_car_control("STOP")
                return True

        self.ros_communicator.publish_car_control("CLOCKWISE_ROTATION_SLOW")
        return False
    
    def L2H(self):
        """
        左轉，直至 polygon 存在頂點佔據畫面右半部。
        """
        X_TOLERANCE = 0
        polygon = self.ros_communicator.get_latest_polygon_list()
        if not polygon or not polygon.points:
            self.ros_communicator.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            return False

        for point in polygon.points:
            if point.x >= 320 - X_TOLERANCE:
                self.ros_communicator.publish_car_control("STOP")
                return True

        self.ros_communicator.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
        return False
    
    def F2M(self):
        """
        往前走，直至有多個 polygon 存在。
        """
        if not self.polygon_exist():
            self.ros_communicator.publish_car_control("FORWARD")
            return False
        
        if not self.get_polygon_type() == "MULTIPLE":
            self.ros_communicator.publish_car_control("FORWARD")
            return False
        
        self.ros_communicator.publish_car_control("STOP")
        return True
    
    def F2S(self):
        """
        往前走，直至 polygon 數量為 1 個。
        """
        if not self.polygon_exist():
            self.ros_communicator.publish_car_control("FORWARD")
            return False
        
        if self.ros_communicator.get_latest_polygon_list().num != 1:
            self.ros_communicator.publish_car_control("FORWARD")
            return False
        
        self.ros_communicator.publish_car_control("STOP")
        return True
    
    def L2S(self):
        """
        左轉，直至 polygon 數量為 1 個。
        """
        if not self.polygon_exist():
            self.ros_communicator.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            return False
        
        if self.get_polygon_type() != "CONVEX":
            self.ros_communicator.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            return False
        
        self.ros_communicator.publish_car_control("STOP")
        return True
    
    def R2S(self):
        """
        右轉，直至 polygon 數量為 1 個。
        """
        if not self.polygon_exist():
            self.ros_communicator.publish_car_control("CLOCKWISE_ROTATION_SLOW")
            return False
        
        if self.get_polygon_type() != "CONVEX":
            self.ros_communicator.publish_car_control("CLOCKWISE_ROTATION_SLOW")
            return False
        
        self.ros_communicator.publish_car_control("STOP")
        return True

    def F2FR(self):
        """
        往前走，直至 polygon 存在頂點佔據畫面最右側。
        """
        X_TOLERANCE = 1
        polygon = self.ros_communicator.get_latest_polygon_list()
        if not polygon or not polygon.points:
            self.ros_communicator.publish_car_control("FORWARD")
            return False

        for point in polygon.points:
            if point.x >= 639 - X_TOLERANCE:
                self.ros_communicator.publish_car_control("STOP")
                return True

        self.ros_communicator.publish_car_control("FORWARD")
        return False

    
    def F2FL(self):
        """
        往前走，直至 polygon 存在頂點佔據畫面最左側。
        """
        X_TOLERANCE = 1
        polygon = self.ros_communicator.get_latest_polygon_list()
        if not polygon or not polygon.points:
            self.ros_communicator.publish_car_control("FORWARD")
            return False

        for point in polygon.points:
            if point.x <= 1 + X_TOLERANCE:
                self.ros_communicator.publish_car_control("STOP")
                return True

        self.ros_communicator.publish_car_control("FORWARD")
        return False
    

    def R2A(self):
        """
        往右轉，直至 polygon 最上方兩個頂點的 y 距離差小於 tolerance。
        """
        Y_TOLERANCE = 5
        polygon = self.ros_communicator.get_latest_polygon_list()
        if not polygon or not polygon.points or len(polygon.points) < 2:
            self.ros_communicator.publish_car_control("CLOCKWISE_ROTATION_SLOW")
            return False

        # 取出所有點的 y 值，找出最小的兩個（最上方）
        sorted_points = sorted(polygon.points, key=lambda p: p.y)
        top_points = sorted_points[:2]
        y_diff = abs(top_points[0].y - top_points[1].y)

        if y_diff <= Y_TOLERANCE:
            self.ros_communicator.publish_car_control("STOP")
            return True

        self.ros_communicator.publish_car_control("CLOCKWISE_ROTATION_SLOW")
        return False
    
    def L2A(self):
        """
        往左轉，直至 polygon 最上方兩個頂點的 y 距離差小於 tolerance。
        """
        Y_TOLERANCE = 5
        polygon = self.ros_communicator.get_latest_polygon_list()
        if not polygon or not polygon.points or len(polygon.points) < 2:
            self.ros_communicator.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            return False

        # 取出所有點的 y 值，找出最小的兩個（最上方）
        sorted_points = sorted(polygon.points, key=lambda p: p.y)
        top_points = sorted_points[:2]
        y_diff = abs(top_points[0].y - top_points[1].y)

        if y_diff <= Y_TOLERANCE:
            self.ros_communicator.publish_car_control("STOP")
            return True

        self.ros_communicator.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
        return False
    
    def L2M(self):
        """
        往左轉，直至存在多個 polygon。
        """
        if not self.polygon_exist():
            self.ros_communicator.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            return False
        
        if self.get_polygon_type() != "MULTIPLE":
            self.ros_communicator.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            return False
        
        self.ros_communicator.publish_car_control("STOP")
        return True
    
    def R2M(self):
        """
        往右轉，直至存在多個 polygon。
        """
        if not self.polygon_exist():
            self.ros_communicator.publish_car_control("CLOCKWISE_ROTATION_SLOW")
            return False
        
        if self.get_polygon_type() != "MULTIPLE":
            self.ros_communicator.publish_car_control("CLOCKWISE_ROTATION_SLOW")
            return False
        
        self.ros_communicator.publish_car_control("STOP")
        return True
    
    def F2C14(self):
        """
        往前走，直至存在類型 14 的 Concave Polygon。
        """
        if not self.polygon_exist():
            self.ros_communicator.publish_car_control("FORWARD")
            return False
        
        if self.get_polygon_type() != "CONCAVE_14":
            self.ros_communicator.publish_car_control("FORWARD")
            return False
        
        self.ros_communicator.publish_car_control("STOP")
        return True
    
    def F2C23(self):
        """
        往前走，直至存在類型 23 的 Concave Polygon。
        """
        if not self.polygon_exist():
            self.ros_communicator.publish_car_control("FORWARD")
            return False
        
        if self.get_polygon_type() != "CONCAVE_23":
            self.ros_communicator.publish_car_control("FORWARD")
            return False
        
        self.ros_communicator.publish_car_control("STOP")
        return True
    
    def R2Cc(self):
        """
        往右轉，直至存在 Concave Polygon（類型 14 或 23）。
        """
        if not self.polygon_exist():
            self.ros_communicator.publish_car_control("CLOCKWISE_ROTATION_SLOW")
            return False
        
        polygon_type = self.get_polygon_type()
        if polygon_type not in ["CONCAVE_14", "CONCAVE_23"]:
            self.ros_communicator.publish_car_control("CLOCKWISE_ROTATION_SLOW")
            return False
        
        self.ros_communicator.publish_car_control("STOP")
        return True
    
    def L2Cc(self):
        """
        往左轉，直至存在 Concave Polygon（類型 14 或 23）。
        """
        if not self.polygon_exist():
            self.ros_communicator.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            return False
        
        polygon_type = self.get_polygon_type()
        if polygon_type not in ["CONCAVE_14", "CONCAVE_23"]:
            self.ros_communicator.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            return False
        
        self.ros_communicator.publish_car_control("STOP")
        return True

    def F2Cc(self):
        """
        往前走，直至存在 Concave Polygon（類型 14 或 23）。
        """
        if not self.polygon_exist():
            self.ros_communicator.publish_car_control("FORWARD")
            return False
        
        polygon_type = self.get_polygon_type()
        if polygon_type not in ["CONCAVE_14", "CONCAVE_23"]:
            self.ros_communicator.publish_car_control("FORWARD")
            return False
        
        self.ros_communicator.publish_car_control("STOP")
        return True
    
    def R2Cv(self):
        """
        往右轉，直至存在 Convex Polygon。
        """
        if not self.polygon_exist():
            self.ros_communicator.publish_car_control("CLOCKWISE_ROTATION_SLOW")
            return False
        
        if self.get_polygon_type() != "CONVEX":
            self.ros_communicator.publish_car_control("CLOCKWISE_ROTATION_SLOW")
            return False
        
        self.ros_communicator.publish_car_control("STOP")
        return True
    
    def L2Cv(self):
        """
        往左轉，直至存在 Convex Polygon。
        """
        if not self.polygon_exist():
            self.ros_communicator.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            return False
        
        if self.get_polygon_type() != "CONVEX":
            self.ros_communicator.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            return False
        
        self.ros_communicator.publish_car_control("STOP")
        return True
    
    def F2Cv(self):
        """
        往前走，直至存在 Convex Polygon。
        """
        if not self.polygon_exist():
            self.ros_communicator.publish_car_control("FORWARD")
            return False
        
        if self.get_polygon_type() != "CONVEX":
            self.ros_communicator.publish_car_control("FORWARD")
            return False
        
        self.ros_communicator.publish_car_control("STOP")
        return True
    
    def F2MvCv(self):
        """
        往前走，直至存在 Convex Polygon 或多個 Polygon。
        """
        if not self.polygon_exist():
            self.ros_communicator.publish_car_control("FORWARD")
            return False
        
        polygon_type = self.get_polygon_type()
        if polygon_type not in ["CONVEX", "MULTIPLE"]:
            self.ros_communicator.publish_car_control("FORWARD")
            return False
        
        self.ros_communicator.publish_car_control("STOP")
        return True


    ### --- 定義大狀態 function --- ###
    ## --- 主狀態 --- ##
    def on_enter_START(self):
        if self.polygon_exist():
            polygon = self.ros_communicator.get_latest_polygon_list()

            if all(point.x < 320 for point in polygon.points):
                print("\nS2")
                self.set_main_state("S2") # D2: 都在左側
                return "STOP"
            else:
                print("\nS3")
                self.set_main_state("S3") # D3: 都在右側
                return "STOP"
        
        else:
            while (not self.R2F()): # 轉向右側檢查
                pass

            if self.get_polygon_type() == "CONVEX": # D1：凹多邊形(另側)
                print("\nS1")
                self.set_main_state("S1")
                return "STOP"
            else:
                print("\nS4")
                self.set_main_state("S4") # D4: 凹多邊形
                return "STOP"


    def on_enter_MID11(self):
        print("<- MID11 ->")
        while (not self.R2F()):
            pass

        if self.get_polygon_type() == "CONCAVE_23":
            print("D11_2")
            self.set_main_state("D11_2")
            return "STOP"
        else:
            while(not self.R2A()):
                pass
            
            self.move("F", 2.3)  
            self.move("S", 0.3)
            if self.get_polygon_type() == "CONCAVE_23":
                print("D11_3")
                self.set_main_state("D11_3")
                return "STOP"
            else:
                print("D11_4")
                self.set_main_state("D11_4")
                return "STOP"

    def on_enter_MID12(self):
        print("<- MID12 ->")
        while (not self.L2F()):
            pass
        
        # Door 1
        if self.get_polygon_type() != "CONVEX":
            print("D12_1")
            self.set_main_state("D12_1")
            return "STOP"
        
        while(not self.R2N()):
            pass
        while(not self.R2F()):
            pass
        self.move("S", 0.3)

        if self.get_polygon_type() == "CONCAVE_23":
            print("D12_3")
            self.set_main_state("D12_3")
            return "STOP"
        else:
            print("D12_4")
            self.set_main_state("D12_4")
            return "STOP"

    def on_enter_MID13(self):
        print("<- MID13 ->")
        if self.get_polygon_type() != "CONVEX":
            print("D13_4")
            self.set_main_state("D13_4")
            return "D13_4"
        
        while(not self.L2A()):
            pass
        while(not self.F2N()):
            pass
        while(not self.L2F()):
            pass
        
        if self.get_polygon_type() == "CONCAVE_23":
            print("D13_2")
            self.set_main_state("D13_2")
            return "STOP"
        else:
            print("D13_1")
            self.set_main_state("D13_1")
            return "STOP"
        

    def on_enter_MID14(self):
        print("<- MID14 ->")
        while (not self.L2F()):
            pass
        # Door 3
        if self.get_polygon_type() == "CONCAVE_23":
            print("D14_3")
            self.set_main_state("D14_3")
            return "STOP"
        
        else:
            while(not self.L2A()):
                pass
            
            self.move("F", 2.5)
            self.move("S", 0.3)

            while(not self.L2F()):
                pass

            if self.get_polygon_type() == "CONCAVE_23":
                print("D14_2")
                self.set_main_state("D14_2")
                return "STOP"
            else:
                print("D14_1")
                self.set_main_state("D14_1")
                return "STOP"


    def on_enter_MID21(self):
        print("<- MID21 ->")
        while (not self.R2F()):
            pass
        
        # Door 2
        if self.get_polygon_type() == "CONCAVE_23":
            self.move("F", 1.5)

            while (not self.L2Cv()):
                pass

            self.set_main_state("DONE")
            return "STOP"

        # Doot 3
        self.move("F", 1.5)

        while (not self.R2F()):
            pass
        
        if self.get_polygon_type() == "CONCAVE_23":
            while (not self.F2Cc()):
                pass
            while (not self.L2Cv()):
                pass
            while (not self.F2N()):
                pass
            while (not self.L2F()):
                pass
            self.set_main_state("DONE")
            return "STOP"

        # Door 4
        else:
            while (not self.F2N()):
                pass
            while (not self.L2H()):
                pass
            while (not self.F2N()):
                pass
            while (not self.L2F()):
                pass
            while (not self.F2N()): # 垂直出去
                pass
            while (not self.L2F()):
                pass
        self.set_main_state("DONE")
        return "STOP"
    
    def on_enter_MID24(self):
        print("<- MID24 ->")
        while (not self.L2F()):
            pass
        
        # Door 3
        if self.get_polygon_type() == "CONCAVE_23":
            print("D24_3")
            self.move("F", 1.5)
            while (not self.R2Cv()):
                pass
            self.set_main_state("DONE")
            return "STOP"
        
        # Door 2
        self.move("F", 1.5)

        while (not self.L2F()):
            pass
        
        if self.get_polygon_type() == "CONCAVE_23":
            print("D24_2")
            while (not self.F2Cc()):
                pass
            while (not self.R2Cv()):
                pass
            while (not self.F2N()):
                pass
            while (not self.R2F()):
                pass
            self.set_main_state("DONE")
            return "STOP"
        
        # Door 1
        else:
            print("D24_1")
            while (not self.L2F()):
                pass
            while (not self.F2N()):
                pass
            while (not self.R2H()):
                pass
            while (not self.F2N()):
                pass
            while (not self.R2F()):
                pass
            while (not self.F2N()): # 垂直出去
                pass
            while (not self.R2F()):
                pass

        self.set_main_state("DONE")
        return "STOP"

    def on_enter_MID22(self):
        print("<- MID22 ->")
        # Door 1
        while (not self.L2F()):
            pass

        if self.get_polygon_type() != "CONVEX":
            print("D22_1")
            while (not self.F2N()):
                pass
            while (not self.R2H()):
                pass
            while (not self.F2N()):
                pass
            while (not self.R2F()):
                pass
            while (not self.F2N()):
                pass # 垂直出去
            while (not self.R2F()):
                pass
            self.set_main_state("DONE")
            return "STOP"

        
        # Door 3
        while (not self.R2Cc()):
            pass
        while (not self.R2F()):
            pass

        if self.get_polygon_type() == "CONCAVE_23":
            print("D22_3")
            while (not self.F2Cv()):
                pass
            while (not self.L2Cc()):
                pass
            while (not self.L2Cv()):
                pass
            while (not self.F2N()):
                pass
            while (not self.L2F()):
                pass
            self.set_main_state("DONE")
            return "STOP"
        
        # Door 4
        print("D22_4")
        while (not self.R2F()):
            pass
        while (not self.R2A()):
            pass

        while( not self.F2N()):
            pass
        while (not self.L2H()):
            pass
        while (not self.F2N()):
            pass
        while (not self.L2F()):
            pass
        while (not self.F2N()):
            pass
        while (not self.L2F()):
            pass
        self.set_main_state("DONE")
        return "STOP"
    
    def on_enter_MID23(self):
        print("<- MID23 ->")
       # Door 4
        while (not self.R2F()):
            pass

        if self.get_polygon_type() != "CONVEX":
            print("D23_4")
            while (not self.L2H()):
                pass
            while (not self.L2F()):
                pass
            while (not self.F2N()):
                pass # 垂直出去
            while (not self.L2F()):
                pass
            self.set_main_state("DONE")
            return "STOP"

        
        # Door 2
        while (not self.L2Cc()):
            pass
        while (not self.L2F()):
            pass

        if self.get_polygon_type() == "CONCAVE_23":
            print("D23_3")
            while (not self.F2Cv()):
                pass
            while (not self.R2Cc()):
                pass
            while (not self.R2Cv()):
                pass
            while (not self.F2N()):
                pass
            while (not self.R2F()):
                pass
            self.set_main_state("DONE")
            return "STOP"
        
        # Door 1
        print("D23_1")
        while( not self.F2N()):
            pass
        while (not self.R2H()):
            pass
        while (not self.F2N()):
            pass
        while (not self.R2F()):
            pass
        while (not self.F2N()):
            pass
        while (not self.R2F()):
            pass
        self.set_main_state("DONE")
        return "STOP"
    
    def on_enter_MID21_open(self):
        print("<- MID21 ->")
        while (not self.R2F()):
            if self.check_and_handle_pikachu_interrupt():
                return "STOP"
            pass

        # 1 -> 2 -> 3 -> 4
        while True:
            if self.check_and_handle_pikachu_interrupt():
                return "STOP"
            
            # Door 2
            self.move("F", 2)
            self.move("L", 1)
            self.move("R", 1)

            # Door 3
            self.move("F", 2)
            self.move("L", 1)
            self.move("R", 1)

            # Door 4
            self.move("F", 2)
            self.move("L", 0.3)
            self.move("F", 2)
            self.move("L", 1)
            self.move("R", 1)

    
    def on_enter_MID22_open(self):
        print("<- MID22 ->")
        while (not self.L2F()):
            if self.check_and_handle_pikachu_interrupt():
                return "STOP"
            pass
        
        while True:
            if self.check_and_handle_pikachu_interrupt():
                return "STOP"
            
            # Door 1
            self.move("F", 2)
            self.move("R", 0.3)
            self.move("F", 2)
            self.move("R", 1)
            self.move("L", 1)

            self.move("B", 2)
            self.move("L", 0.3)

            # Door 3 
            self.move("B", 5)
            self.move("R", 1)
            self.move("L", 1)

            # Door 4
            self.move("B", 5)
            self.move("R", 0.3)
            self.move("F", 3)
            self.move("L", 1)
            self.move("R", 1)

    
    def on_enter_MID23_open(self):
        print("<- MID23 ->")
        while (not self.R2F()):
            if self.check_and_handle_pikachu_interrupt():
                return "STOP"
            pass

        while True:
            if self.check_and_handle_pikachu_interrupt():
                return "STOP"
            
            # Door 4
            self.move("F", 2)
            self.move("L", 0.3)
            self.move("F", 3)
            self.move("L", 1)
            self.move("R", 1)

            self.move("B", 3)
            self.move("L", 0.3)

            # Door 2 
            self.move("F", 5)
            self.move("R", 1)
            self.move("L", 1)

            # Door 1
            self.move("F", 5)
            self.move("R", 0.3)
            self.move("F", 3)
            self.move("R", 1)
            self.move("L", 1)

    
    def on_enter_MID24_open(self):
        print("<- MID24 ->")
        while (not self.L2F()):
            if self.check_and_handle_pikachu_interrupt():
                return "STOP"
            pass

        # 4 -> 3 -> 2 -> 1
        while True:
            if self.check_and_handle_pikachu_interrupt():
                return "STOP"
            
            # Door 3
            self.move("F", 2)
            self.move("R", 1)
            self.move("L", 1)

            # Door 2
            self.move("F", 2)
            self.move("R", 1)
            self.move("L", 1)

            # Door 1
            self.move("F", 2)
            self.move("R", 0.3)
            self.move("F", 3)
            self.move("R", 1)
            self.move("L", 1)



    def on_enter_DONE(self):
        return "STOP"
    

    ## --- 子情況 --- ##
    # --- START --- #
    def on_enter_S1(self):
        if self.run_sub_state_machine(['L2N', 'L2F', 'F2N', 'R2F', 'F2N']):
            self.set_main_state("MID11")
        return "STOP"

    def on_enter_S2(self):
        if self.run_sub_state_machine(['L2F', 'F2M', 'R2S', 'R2F', 'F2N']):
            self.set_main_state("MID12")
        return "STOP"
    
    def on_enter_S3(self):
        if self.run_sub_state_machine(['R2F', 'F2M', 'F2S', 'L2F']):
            self.set_main_state("MID13")
        return "STOP"
    
    def on_enter_S4(self):
        if self.run_sub_state_machine(['F2N', 'L2H', 'F2N', 'L2H', 'F2N']):
            self.set_main_state("MID14")
        return "STOP"
    
    
    # --- MID --- #
    # D11
    def on_enter_D11_2(self): #OK
        if self.run_sub_state_machine(['L2F', 'F2M', 'L2S', 'L2F', 'F2N']):
            self.set_main_state("MID22")
            # self.set_main_state("MID22_open")
        return "STOP"
    
    def on_enter_D11_3(self): #OK
        if self.run_sub_state_machine(["L2F", "F2M", "L2S", "F2N"]):
            self.set_main_state("MID23")
            # self.set_main_state("MID23_open")
        return "STOP"

    def on_enter_D11_4(self): #OK
        if self.run_sub_state_machine(['R2F', 'L2A', 'F2N', 'L2F', 'F2N']):
            self.set_main_state("MID24")
            # self.set_main_state("MID24_open")
        return "STOP"
    

    # D12
    def on_enter_D12_1(self): #OK
        if self.run_sub_state_machine(['F2N', 'R2H', 'F2N', 'R2H', 'F2N']):
            self.set_main_state("MID21")
            # self.set_main_state("MID21_open")
        return "STOP"
    
    def on_enter_D12_3(self): #OK
        if self.run_sub_state_machine(['F2MvCv', 'L2S', 'L2F', 'F2N']):
            self.set_main_state("MID23")
            # self.set_main_state("MID23_open")
        return "STOP"

    def on_enter_D12_4(self): #OK
        if self.run_sub_state_machine(['F2N', 'L2F', 'F2N', 'L2H', 'F2N', 'L2H', 'F2N']):
            self.set_main_state("MID24")
            # self.set_main_state("MID24_open")
        return "STOP"
    
    
    # D13
    def on_enter_D13_1(self): #OK
        self.move("F", 0.2)
        self.move("L", 0.2)
        if self.run_sub_state_machine(["L2F", "L2A", "F2N", "R2F", "F2N", "R2H"]):
            self.set_main_state("MID21")
            # self.set_main_state("MID21_open")
        return "STOP"
    
    def on_enter_D13_2(self): #OK
        if self.run_sub_state_machine(['F2MvCv', 'R2S', 'R2A', 'F2N']):
            self.set_main_state("MID22")
            # self.set_main_state("MID22_open")
        return "STOP"

    def on_enter_D13_4(self): #OK
        if self.run_sub_state_machine(['F2Cv', 'R2F', 'F2N', 'L2F', 'F2N']):
            self.set_main_state("MID24")
            # self.set_main_state("MID24_open")
        return "STOP"


    # D14
    def on_enter_D14_1(self): #OK
        if self.run_sub_state_machine(['L2F', 'F2N', 'R2H', 'F2N', 'R2H', 'F2N', 'R2H', 'F2N', 'R2H', 'F2N']):
            self.move("F", 0.2)
            self.set_main_state("MID21")
            # self.set_main_state("MID21_open")
        return "STOP"
        
    def on_enter_D14_2(self): #OK
        self.move("S", 121)
        if self.run_sub_state_machine(["F2MvCv", "R2M", "R2S", "R2F", "R2A", "F2N"]):
            self.set_main_state("MID22")
            # self.set_main_state("MID22_open")
        return "STOP"
    
    def on_enter_D14_3(self): #OK
        if self.run_sub_state_machine(["L2F", "F2Cv", "R2F", "F2Cv", "R2M", "R2S", "R2A", "F2N"]):
            self.set_main_state("MID23")
            # self.set_main_state("MID23_open")
        return "STOP"