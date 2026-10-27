FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=humble \
    WS=/ws

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget curl \
        python3-pip python3-colcon-common-extensions python3-rosdep python3-vcstool \
        iproute2 iputils-ping net-tools ethtool tcpdump \
        ros-humble-rmw-cyclonedds-cpp \
        ros-humble-rtabmap-ros \
        ros-humble-rqt-image-view ros-humble-rqt-reconfigure ros-humble-rviz2 \
        ros-humble-tf2-tools ros-humble-image-transport-plugins

RUN mkdir -p ${WS}
WORKDIR ${WS}

RUN git clone --branch humble --recurse-submodules \
        https://github.com/carnegierobotics/multisense_ros2.git ${WS}/src

RUN apt-get update \
    && . /opt/ros/${ROS_DISTRO}/setup.sh \
    && rosdep update \
    && rosdep install --from-paths src -y --ignore-src

RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
    && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "source /ws/install/setup.bash" >> /root/.bashrc \
    && echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc

CMD ["bash"]
