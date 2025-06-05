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

    def RGBcam_nav_unity_door_random(self):
        # 1. 先判斷是否有辨識到 pikachu
        detected_list = self.ros_communicator.get_latest_yolo_detection_list()
        image_width = 640
        image_height = 480
        image_area = image_width * image_height
        image_center_x = image_width // 2

        bbox_center_range_ratio = 0.06
        pikachu_area_threshold = 0.06
        obstacle_x_offset = 30

        SEARCH_ROTATION_ACTION = "COUNTERCLOCKWISE_ROTATION"
        ALIGN_ROTATE_LEFT_ACTION = "COUNTERCLOCKWISE_ROTATION_SLOW"
        ALIGN_ROTATE_RIGHT_ACTION = "CLOCKWISE_ROTATION_SLOW"

        # 若有 pikachu，直接用 living_room_random 邏輯
        if detected_list and hasattr(detected_list, "objects"):
            pikachu_objects = [obj for obj in detected_list.objects if obj.label == "pikachu"]
            if pikachu_objects:
                target = max(pikachu_objects, key=lambda obj: (obj.x2 - obj.x1) * (obj.y2 - obj.y1))
                bbox_x1 = target.x1
                bbox_x2 = target.x2
                center_range_px = int(image_width * bbox_center_range_ratio)
                if not (bbox_x1 <= image_center_x + center_range_px and bbox_x2 >= image_center_x - center_range_px):
                    if target.cx < image_center_x:
                        self.print_status(ALIGN_ROTATE_LEFT_ACTION, "align left")
                        return ALIGN_ROTATE_LEFT_ACTION
                    else:
                        self.print_status(ALIGN_ROTATE_RIGHT_ACTION, "align right")
                        return ALIGN_ROTATE_RIGHT_ACTION
                bbox_area = (target.x2 - target.x1) * (target.y2 - target.y1)
                bbox_area_ratio = bbox_area / image_area
                self.print_status("FORWARD", f"bbox_area_ratio: {bbox_area_ratio:.2f}")
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
                            self.print_status(ALIGN_ROTATE_RIGHT_ACTION, "avoid obstacle: left edge")
                            return ALIGN_ROTATE_RIGHT_ACTION
                        else:
                            self.print_status(ALIGN_ROTATE_LEFT_ACTION, "avoid obstacle: right edge")
                            return ALIGN_ROTATE_LEFT_ACTION
                    return "FORWARD"
                else:
                    self.print_status("STOP", f"bbox_area_ratio: {bbox_area_ratio:.2f} (threshold reached)")
                    return "STOP"

        # 2. State machine
        state_name = self.get_current_state() # 取得目前 state 名稱
        state_func = getattr(self, f"on_enter_{state_name}", None) # 動態取得對應的 state function

        polygon_list = self.ros_communicator.get_latest_polygon_list()
        
        if state_func is not None:
            # 呼叫 state function，傳入 polygon_list，並取得 action
            action = state_func(polygon_list)
        else:
            action = "STOP"
        return action

    def RGBcam_nav_unity_pikachu(self):
        action = "STOP"
        return action


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
        time.sleep(duration)
        self.ros_communicator.publish_car_control("STOP")

    ##### --- state machine --- #####

    ### --- 檢測狀態 --- ###
    # 這個階段的狀態主要用於尋找與檢測目標點。
    # 這裡狀態將依序進行：START -> MID1 -> MID2 -> FINAL。
    
    def on_enter_START(self, polygon_list):
        """
        只判斷一次，有無 polygon list：
        - 沒有，則 BACKWARD 1s，並 STOP
        - 有，若頂點皆在畫面右側，則順時針轉 1s 並 STOP；左側則逆時針轉 1s 並 stop。(只看 x 座標)
        """
        
        # 沒有 polygon list
        if not polygon_list or not hasattr(polygon_list, "points") or not polygon_list.points:
            self.move("R", 0.2)
            new_polygon_list = self.ros_communicator.get_latest_polygon_list()

            # 檢測目標：D4
            if new_polygon_list.len>=5:
                self.move("R", 0.3)
                self.move("F", 2.2)
                self.move("S", 0.2)
                self.move("L", 0.5)
                self.move("F", 0.6)
                self.set_current_state("MID1")
                return "STOP"

            # 檢測目標：D1
            else:
                self.move("L", 0.7)
                self.move("F", 2.2)
                self.move("S", 0.2)
                self.move("R", 0.5)
                self.move("F", 0.6)
                self.set_current_state("MID1")
                return "STOP"
        
        # 起初就有 polygon list
        else:
            ## 判斷所有頂點的 x 座標是否都在畫面右側或左側
            x_list = [p.x for p in polygon_list.points]
            
            # 檢測目標：D3
            if all(x > 640//2 for x in x_list):
                self.move("R", 0.5)
                self.move("F", 0.6)
                self.move("S", 0.5)
                self.move("L", 0.5)
                self.move("F", 0.6)
                self.set_current_state("MID1")
                return "STOP"
            
            # 檢測目標：D2
            else:
                self.move("L", 0.5)
                self.move("F", 0.6)
                self.move("S", 0.5)
                self.move("R", 0.5)
                self.move("F", 0.6)
                self.set_current_state("MID1")
                return "STOP"

    def on_enter_MID1(self, polygon_list):
        # 只做決策
        return "STOP"

    def on_enter_MID2(self, polygon_list):
        pass

    def on_enter_FINAL(self, polygon_list):
        pass

    
    ### --- 移動狀態 --- ###
    # 這個階段的狀態僅做移動，並行走固定的路線。
    # 所有行走路線為固定腳本，移動時間到就停止移動，並回到停留與檢測狀態。

    def on_enter_D11(self):
        pass

    def on_enter_D12(self):
        pass

    def on_enter_D13(self):
        pass

    def on_enter_D14(self):
        pass

    def on_enter_D21(self):
        pass

    def on_enter_D22(self):
        pass

    def on_enter_D23(self):
        pass

    def on_enter_D24(self):
        pass

    def on_enter_D31(self):
        pass

    def on_enter_D32(self):
        pass

    def on_enter_D33(self):
        pass

    def on_enter_D34(self):
        pass

    def on_enter_D41(self):
        pass

    def on_enter_D42(self):
        pass

    def on_enter_D43(self):
        pass

    def on_enter_D44(self):
        pass

    def on_enter_D1F(self):
        pass

    def on_enter_D2F(self):
        pass

    def on_enter_D3F(self):
        pass

    def on_enter_D4F(self):
        pass