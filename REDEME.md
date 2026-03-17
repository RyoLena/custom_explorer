 0. 每个终端都要先执行

  source /opt/ros/foxy/setup.bash
  source ~/SLAM_ws/install/setup.bash
  export TURTLEBOT3_MODEL=burger    # 或 waffle / waffle_pi

  Terminal 1 — Gazebo 仿真环境

  ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

  Terminal 2 — SLAM Toolbox（建图 + 提供 /map 和 TF）

  ros2 launch slam_toolbox online_async_launch.py use_sim_time:=True

  Terminal 3 — Nav2 导航栈（提供 navigate_to_pose action server）

  ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True

  Terminal 4 — RViz 可视化

  ros2 launch nav2_bringup rviz_launch.py

  RViz 打开后，手动添加：
  - Add → By topic → /exploration_markers → MarkerArray

  Terminal 5 — 启动 Explorer 节点

  # 默认参数
  ros2 run custom_explorer_cpp explorer_node

  # 或自定义策略权重
  ros2 run custom_explorer_cpp explorer_node --ros-args -p alpha:=0.5 -p beta:=2.0