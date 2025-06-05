#!/bin/bash
set -e


echo "[0/4] 清除可能的舊 ROS 軟體源..."
rm -f /etc/apt/sources.list.d/ros2-latest.list

echo "[1/4] 建立 /etc/apt/keyrings 資料夾..."
mkdir -p /etc/apt/keyrings

echo "[2/4] 下載並轉換 ROS GPG 金鑰..."
curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | \
  gpg --dearmor -o /etc/apt/keyrings/ros.gpg

echo "[3/4] 設定 ROS 2 軟體源（覆蓋 ros.list）..."
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/ros.gpg] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros.list

echo "[4/4] 更新 APT 清單..."
apt update

apt install -y nano \
              x11-apps

apt install -y ros-humble-rtabmap-ros \
              ros-humble-foxglove-bridge \