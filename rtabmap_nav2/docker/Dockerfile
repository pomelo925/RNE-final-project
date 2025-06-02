################################################################################################
# - Base stage
#   - This stage serves as the foundational stage for all other stages.
#   - Base image: OSRF ROS Humble Desktop Full
#       - https://hub.docker.com/r/osrf/ros/tags?page=1&name=humble-desktop-full
################################################################################################

FROM osrf/ros:humble-desktop-full AS base

LABEL org.opencontainers.image.title="Harbor Tracking App (CUDA 12.8 + Python 3.10)"
LABEL org.opencontainers.image.authors="yoseph.huang@gmail.com"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]


################################################################################################
# user-setup stage
# - Creates a non-root user for improved security.
################################################################################################
FROM base AS user-setup

ARG USER_UID=1000
ARG USER_GID=1000
ENV USERNAME=appuser

## Create a non-root user
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash && \
    apt-get update && \
    apt-get install -y sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    rm -rf /var/lib/apt/lists/*


################################################################################################
# - RTABMap stage
#   - In this stage, I will install required packages for RTAB-Map.
#   - Reference:
#       - https://github.com/introlab/rtabmap_ros/tree/humble-devel?tab=readme-ov-file#installation
#       - https://github.com/introlab/rtabmap_ros.git
################################################################################################

FROM user-setup AS rtabmap

RUN apt-get update && \
    apt-get install -y \
        ros-humble-rtabmap-ros \
        ros-humble-foxglove-bridge && \
    rm -rf /var/lib/apt/lists/*


################################################################################################
# Nav2 stage
# - In this stage, I will install required packages for ROS2 Nav2 and exploration module.
################################################################################################
FROM rtabmap AS nav2

# Install Nav2 dependencies
RUN apt-get update && \
    apt-get install -y \
        ros-humble-nav2-* \
    rm -rf /var/lib/apt/lists/*



################################################################################################
# Final stage
# - In this stage, I will install some debugging tools and set up the environment.
################################################################################################
FROM nav2 AS final


WORKDIR /home/appuser
USER appuser