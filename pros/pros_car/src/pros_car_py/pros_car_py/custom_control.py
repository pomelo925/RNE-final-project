import time

class CustomControl:
    def __init__(self, car_controller, arm_controller):
        self.car_controller = car_controller
        self.arm_controller = arm_controller
        
    def manual_control(self, key):
        self.car_controller.manual_control(key)
        self.arm_controller.manual_control(key)

    def auto_rotate_clockwise(self, duration=5):
        """讓車子自動順時針旋轉指定秒數"""
        start_time = time.time()
        while time.time() - start_time < duration:
            self.car_controller.manual_control('r')  # 'r' 代表順時針旋轉
            time.sleep(0.1)
        self.car_controller.manual_control('z')  # 停止
