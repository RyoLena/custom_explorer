#include "rclcpp/rclcpp.hpp"
#include <cmath>
#include <cstddef>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <queue>
#include <set>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_srvs/srv/empty.hpp>

#include <rclcpp_action/client.hpp>
#include <rclcpp_action/client_goal_handle.hpp>
#include <rclcpp_action/create_client.hpp>
#include <rclcpp_action/types.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>

using namespace std::chrono_literals;
using NavigateToPose = nav2_msgs::action::NavigateToPose;
using GoalHandleNav = rclcpp_action::ClientGoalHandle<NavigateToPose>;

class ExplorerNode : public rclcpp::Node {
public:
  ExplorerNode() : Node("explorer_node") {
    // Subscriptions
    subscription_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/map", 10,
        std::bind(&ExplorerNode::map_callback, this, std::placeholders::_1));

    amcl_pose_sub_ =
        this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/amcl_pose", 10,
            std::bind(&ExplorerNode::amcl_pose_callback, this,
                      std::placeholders::_1));

    // Action client
    action_client_ =
        rclcpp_action::create_client<NavigateToPose>(this, "navigate_to_pose");

    // Service client for relocalization
    relocalization_client_ =
        this->create_client<std_srvs::srv::Empty>(
            "/reinitialize_global_localization");

    // TF2
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Publisher
    marker_pub_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/exploration_markers", 10);

    // Parameters
    this->declare_parameter("alpha", 1.0);
    this->declare_parameter("beta", 0.5);
    alpha_ = this->get_parameter("alpha").as_double();
    beta_ = this->get_parameter("beta").as_double();

    // Timer
    timer_ =
        this->create_wall_timer(5.0s, std::bind(&ExplorerNode::explore, this));

    // Init state
    robot_world_x_ = 0.0;
    robot_world_y_ = 0.0;
    is_navigating_ = false;
    consecutive_failures_ = 0;
    exploration_complete_ = false;
    is_localization_lost_ = false;
    current_target_grid_ = {-1, -1};

    RCLCPP_INFO(logger_, "Explorer node initialized (alpha=%.2f, beta=%.2f)",
                alpha_, beta_);
  }

  void navigate_to(double x, double y);
  void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  void goal_response_callback(
      std::shared_future<GoalHandleNav::SharedPtr> future);
  void result_callback(const GoalHandleNav::WrappedResult &result);
  void amcl_pose_callback(
      const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);

  void update_robot_pose();
  std::vector<std::vector<std::pair<int, int>>> find_frontier_clusters();
  std::pair<double, double>
  cluster_centroid(const std::vector<std::pair<int, int>> &cluster);
  int find_best_frontier_cluster();
  void publish_markers();
  void explore();

private:
  // ROS interfaces
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr subscription_;
  rclcpp::Subscription<
      geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_pose_sub_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr action_client_;
  rclcpp::Client<std_srvs::srv::Empty>::SharedPtr relocalization_client_;
  nav_msgs::msg::OccupancyGrid::SharedPtr map_data_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Logger logger_ = this->get_logger();

  // TF2
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Publisher
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_pub_;

  // Robot state
  double robot_world_x_;
  double robot_world_y_;
  bool is_navigating_;
  bool exploration_complete_;
  bool is_localization_lost_;

  // Navigation state
  std::set<std::pair<int, int>> failed_frontiers_;
  int consecutive_failures_;
  std::pair<int, int> current_target_grid_;

  // Frontier clusters (cached from last find)
  std::vector<std::vector<std::pair<int, int>>> frontier_clusters_;

  // Parameters
  double alpha_;
  double beta_;

  // Visualization
  std::vector<std::pair<double, double>> path_history_;
};

// ============================================================
// Step 1: TF2 robot pose
// ============================================================

void ExplorerNode::update_robot_pose() {
  try {
    auto transform = tf_buffer_->lookupTransform(
        "map", "base_link", tf2::TimePointZero);
    robot_world_x_ = transform.transform.translation.x;
    robot_world_y_ = transform.transform.translation.y;
    RCLCPP_DEBUG(logger_, "Robot pose: x=%.2f, y=%.2f",
                 robot_world_x_, robot_world_y_);
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN(logger_, "Could not get robot pose: %s", ex.what());
  }
}

// ============================================================
// Step 6: AMCL pose callback — kidnapped robot recovery
// ============================================================

void ExplorerNode::amcl_pose_callback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
  double cov_x = msg->pose.covariance[0];
  double cov_y = msg->pose.covariance[7];
  double cov_theta = msg->pose.covariance[35];
  const double cov_threshold = 0.5;

  if (cov_x > cov_threshold || cov_y > cov_threshold ||
      cov_theta > cov_threshold) {
    if (!is_localization_lost_) {
      is_localization_lost_ = true;
      RCLCPP_WARN(logger_,
                  "Localization lost! cov_x=%.3f, cov_y=%.3f, cov_theta=%.3f",
                  cov_x, cov_y, cov_theta);
      // Trigger global relocalization
      if (relocalization_client_->wait_for_service(2s)) {
        auto request = std::make_shared<std_srvs::srv::Empty::Request>();
        relocalization_client_->async_send_request(request);
        RCLCPP_INFO(logger_, "Triggered global relocalization");
      } else {
        RCLCPP_WARN(logger_, "Relocalization service not available");
      }
    }
  } else {
    if (is_localization_lost_) {
      is_localization_lost_ = false;
      RCLCPP_INFO(logger_, "Localization recovered.");
    }
  }
}

// ============================================================
// Map callback
// ============================================================

void ExplorerNode::map_callback(
    const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
  map_data_ = msg;
  RCLCPP_DEBUG(logger_, "Map received (%ux%u)",
               msg->info.width, msg->info.height);
}

// ============================================================
// Navigation
// ============================================================

void ExplorerNode::navigate_to(double x, double y) {
  geometry_msgs::msg::PoseStamped goal_pose;
  goal_pose.header.frame_id = "map";
  goal_pose.header.stamp = this->get_clock()->now();
  goal_pose.pose.position.x = x;
  goal_pose.pose.position.y = y;
  goal_pose.pose.orientation.w = 1.0;

  NavigateToPose::Goal goal_msg;
  goal_msg.pose = goal_pose;

  RCLCPP_INFO(logger_, "Navigate to goal: x=%.2f, y=%.2f", x, y);

  if (!action_client_->wait_for_action_server(5s)) {
    RCLCPP_ERROR(logger_, "Action server not available!");
    is_navigating_ = false;
    return;
  }

  auto send_goal_options =
      rclcpp_action::Client<NavigateToPose>::SendGoalOptions();

  send_goal_options.goal_response_callback = std::bind(
      &ExplorerNode::goal_response_callback, this, std::placeholders::_1);

  send_goal_options.result_callback =
      std::bind(&ExplorerNode::result_callback, this, std::placeholders::_1);

  action_client_->async_send_goal(goal_msg, send_goal_options);
}

// ============================================================
// Step 3: Goal response — reject resets navigating flag
// ============================================================

void ExplorerNode::goal_response_callback(
    std::shared_future<GoalHandleNav::SharedPtr> future) {
  auto goal_handle = future.get();
  if (goal_handle == nullptr) {
    RCLCPP_WARN(logger_, "Goal rejected");
    is_navigating_ = false;
    return;
  }
  RCLCPP_INFO(logger_, "Goal accepted");
}

// ============================================================
// Step 3: Result callback — blacklist + failure counting
// ============================================================

void ExplorerNode::result_callback(
    const GoalHandleNav::WrappedResult &result) {
  is_navigating_ = false;

  if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
    RCLCPP_INFO(logger_, "Goal succeeded");
    consecutive_failures_ = 0;
  } else {
    RCLCPP_WARN(logger_, "Goal failed");
    if (current_target_grid_.first != -1) {
      failed_frontiers_.insert(current_target_grid_);
      RCLCPP_INFO(logger_, "Added grid (%d, %d) to blacklist (total: %zu)",
                  current_target_grid_.first, current_target_grid_.second,
                  failed_frontiers_.size());
    }
    consecutive_failures_++;
    if (consecutive_failures_ >= 5) {
      exploration_complete_ = true;
      RCLCPP_WARN(logger_,
                  "5 consecutive failures. Marking exploration complete.");
    }
  }
}

// ============================================================
// Step 2: BFS frontier clustering (8-connected)
// ============================================================

std::vector<std::vector<std::pair<int, int>>>
ExplorerNode::find_frontier_clusters() {
  std::vector<std::vector<std::pair<int, int>>> clusters;
  int rows = static_cast<int>(map_data_->info.height);
  int cols = static_cast<int>(map_data_->info.width);
  std::vector<bool> visited(rows * cols, false);

  // 8-connected neighbor offsets
  const int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
  const int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};

  for (int i = 1; i < rows - 1; i++) {
    for (int j = 1; j < cols - 1; j++) {
      int index = i * cols + j;
      if (visited[index] || map_data_->data[index] != 0)
        continue;

      // Check if this free cell borders unknown
      bool is_frontier = false;
      for (int d = 0; d < 8; d++) {
        int ni = i + dx[d];
        int nj = j + dy[d];
        if (map_data_->data[ni * cols + nj] == -1) {
          is_frontier = true;
          break;
        }
      }
      if (!is_frontier)
        continue;

      // BFS to find connected frontier cells
      std::vector<std::pair<int, int>> cluster;
      std::queue<std::pair<int, int>> bfs_queue;
      bfs_queue.push({i, j});
      visited[index] = true;

      while (!bfs_queue.empty()) {
        std::pair<int, int> cell = bfs_queue.front();
        bfs_queue.pop();
        cluster.push_back(cell);

        for (int d = 0; d < 8; d++) {
          int ni = cell.first + dx[d];
          int nj = cell.second + dy[d];
          if (ni < 1 || ni >= rows - 1 || nj < 1 || nj >= cols - 1)
            continue;
          int nidx = ni * cols + nj;
          if (visited[nidx] || map_data_->data[nidx] != 0)
            continue;

          // Check if neighbor is also a frontier cell
          bool neighbor_frontier = false;
          for (int dd = 0; dd < 8; dd++) {
            int nni = ni + dx[dd];
            int nnj = nj + dy[dd];
            if (map_data_->data[nni * cols + nnj] == -1) {
              neighbor_frontier = true;
              break;
            }
          }
          if (neighbor_frontier) {
            visited[nidx] = true;
            bfs_queue.push({ni, nj});
          }
        }
      }

      // Filter small clusters (noise)
      if (cluster.size() >= 3) {
        clusters.push_back(cluster);
      }
    }
  }

  RCLCPP_INFO(logger_, "Found %zu frontier clusters", clusters.size());
  for (size_t c = 0; c < clusters.size(); c++) {
    RCLCPP_DEBUG(logger_, "  Cluster %zu: %zu cells", c, clusters[c].size());
  }

  return clusters;
}

// ============================================================
// Step 2: Cluster centroid (grid -> world coordinates)
// ============================================================

std::pair<double, double> ExplorerNode::cluster_centroid(
    const std::vector<std::pair<int, int>> &cluster) {
  double sum_x = 0.0, sum_y = 0.0;
  for (const auto &cell : cluster) {
    // row=i -> y, col=j -> x
    sum_x += cell.second * map_data_->info.resolution +
             map_data_->info.origin.position.x;
    sum_y += cell.first * map_data_->info.resolution +
             map_data_->info.origin.position.y;
  }
  double n = static_cast<double>(cluster.size());
  return {sum_x / n, sum_y / n};
}

// ============================================================
// Step 3+4: Best frontier cluster (score = -alpha*dist + beta*size)
// ============================================================

int ExplorerNode::find_best_frontier_cluster() {
  frontier_clusters_ = find_frontier_clusters();

  int best_idx = -1;
  double best_score = -std::numeric_limits<double>::max();

  for (size_t i = 0; i < frontier_clusters_.size(); i++) {
    // Convert centroid to grid coords for blacklist check
    std::pair<double, double> centroid_world =
        cluster_centroid(frontier_clusters_[i]);
    int centroid_row = static_cast<int>(
        (centroid_world.second - map_data_->info.origin.position.y) /
        map_data_->info.resolution);
    int centroid_col = static_cast<int>(
        (centroid_world.first - map_data_->info.origin.position.x) /
        map_data_->info.resolution);

    if (failed_frontiers_.count({centroid_row, centroid_col}))
      continue;

    double dist =
        std::sqrt(std::pow(robot_world_x_ - centroid_world.first, 2) +
                  std::pow(robot_world_y_ - centroid_world.second, 2));
    double cluster_size = static_cast<double>(frontier_clusters_[i].size());

    double score = -alpha_ * dist + beta_ * cluster_size;
    if (score > best_score) {
      best_score = score;
      best_idx = static_cast<int>(i);
    }
  }

  return best_idx;
}

// ============================================================
// Step 5: RViz MarkerArray visualization
// ============================================================

void ExplorerNode::publish_markers() {
  visualization_msgs::msg::MarkerArray marker_array;
  auto stamp = this->get_clock()->now();
  int id = 0;

  // Delete all previous markers first
  visualization_msgs::msg::Marker delete_marker;
  delete_marker.header.frame_id = "map";
  delete_marker.header.stamp = stamp;
  delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
  marker_array.markers.push_back(delete_marker);

  // 1. Frontier clusters (POINTS, different color per cluster)
  for (size_t c = 0; c < frontier_clusters_.size(); c++) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = stamp;
    marker.ns = "frontier_clusters";
    marker.id = id++;
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;

    // Hue rotation for distinct cluster colors
    float hue = static_cast<float>(c) /
                static_cast<float>(
                    frontier_clusters_.size() > 0 ? frontier_clusters_.size()
                                                  : 1);
    marker.color.r =
        std::max(0.0f, std::min(1.0f, std::abs(hue * 6.0f - 3.0f) - 1.0f));
    marker.color.g =
        std::max(0.0f, std::min(1.0f, 2.0f - std::abs(hue * 6.0f - 2.0f)));
    marker.color.b =
        std::max(0.0f, std::min(1.0f, 2.0f - std::abs(hue * 6.0f - 4.0f)));
    marker.color.a = 0.8f;

    for (const auto &cell : frontier_clusters_[c]) {
      geometry_msgs::msg::Point p;
      p.x = cell.second * map_data_->info.resolution +
            map_data_->info.origin.position.x;
      p.y = cell.first * map_data_->info.resolution +
            map_data_->info.origin.position.y;
      p.z = 0.05;
      marker.points.push_back(p);
    }
    marker_array.markers.push_back(marker);
  }

  // 2. Current navigation target (SPHERE, green)
  if (current_target_grid_.first != -1 && map_data_) {
    visualization_msgs::msg::Marker target_marker;
    target_marker.header.frame_id = "map";
    target_marker.header.stamp = stamp;
    target_marker.ns = "current_target";
    target_marker.id = id++;
    target_marker.type = visualization_msgs::msg::Marker::SPHERE;
    target_marker.action = visualization_msgs::msg::Marker::ADD;
    target_marker.scale.x = 0.3;
    target_marker.scale.y = 0.3;
    target_marker.scale.z = 0.3;
    target_marker.color.r = 0.0f;
    target_marker.color.g = 1.0f;
    target_marker.color.b = 0.0f;
    target_marker.color.a = 1.0f;
    target_marker.pose.position.x =
        current_target_grid_.second * map_data_->info.resolution +
        map_data_->info.origin.position.x;
    target_marker.pose.position.y =
        current_target_grid_.first * map_data_->info.resolution +
        map_data_->info.origin.position.y;
    target_marker.pose.position.z = 0.1;
    target_marker.pose.orientation.w = 1.0;
    marker_array.markers.push_back(target_marker);
  }

  // 3. Blacklisted frontiers (SPHERE_LIST, red)
  if (!failed_frontiers_.empty() && map_data_) {
    visualization_msgs::msg::Marker blacklist_marker;
    blacklist_marker.header.frame_id = "map";
    blacklist_marker.header.stamp = stamp;
    blacklist_marker.ns = "blacklisted";
    blacklist_marker.id = id++;
    blacklist_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    blacklist_marker.action = visualization_msgs::msg::Marker::ADD;
    blacklist_marker.scale.x = 0.15;
    blacklist_marker.scale.y = 0.15;
    blacklist_marker.scale.z = 0.15;
    blacklist_marker.color.r = 1.0f;
    blacklist_marker.color.g = 0.0f;
    blacklist_marker.color.b = 0.0f;
    blacklist_marker.color.a = 0.8f;

    for (const auto &cell : failed_frontiers_) {
      geometry_msgs::msg::Point p;
      p.x = cell.second * map_data_->info.resolution +
            map_data_->info.origin.position.x;
      p.y = cell.first * map_data_->info.resolution +
            map_data_->info.origin.position.y;
      p.z = 0.1;
      blacklist_marker.points.push_back(p);
    }
    marker_array.markers.push_back(blacklist_marker);
  }

  // 4. Robot path history (LINE_STRIP, blue)
  if (path_history_.size() >= 2) {
    visualization_msgs::msg::Marker path_marker;
    path_marker.header.frame_id = "map";
    path_marker.header.stamp = stamp;
    path_marker.ns = "path_history";
    path_marker.id = id++;
    path_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    path_marker.action = visualization_msgs::msg::Marker::ADD;
    path_marker.scale.x = 0.05;
    path_marker.color.r = 0.0f;
    path_marker.color.g = 0.0f;
    path_marker.color.b = 1.0f;
    path_marker.color.a = 0.8f;

    for (const auto &pt : path_history_) {
      geometry_msgs::msg::Point p;
      p.x = pt.first;
      p.y = pt.second;
      p.z = 0.02;
      path_marker.points.push_back(p);
    }
    marker_array.markers.push_back(path_marker);
  }

  marker_pub_->publish(marker_array);
}

// ============================================================
// Main exploration loop
// ============================================================

void ExplorerNode::explore() {
  if (exploration_complete_) {
    return;
  }

  if (is_navigating_) {
    RCLCPP_DEBUG(logger_, "Navigation in progress, skipping");
    return;
  }

  if (is_localization_lost_) {
    RCLCPP_WARN(logger_, "Localization lost, pausing exploration");
    return;
  }

  if (!map_data_) {
    RCLCPP_WARN(logger_, "No Map Data Available");
    return;
  }

  // Update robot pose from TF
  update_robot_pose();

  // Record path history
  path_history_.push_back({robot_world_x_, robot_world_y_});

  // Find best frontier cluster
  int best_idx = find_best_frontier_cluster();
  if (best_idx < 0) {
    RCLCPP_INFO(logger_, "No frontiers found. Exploration complete!");
    exploration_complete_ = true;
    return;
  }

  // Get centroid of best cluster
  std::pair<double, double> centroid =
      cluster_centroid(frontier_clusters_[best_idx]);

  // Record target grid coords for blacklist tracking
  int target_row = static_cast<int>(
      (centroid.second - map_data_->info.origin.position.y) /
      map_data_->info.resolution);
  int target_col = static_cast<int>(
      (centroid.first - map_data_->info.origin.position.x) /
      map_data_->info.resolution);
  current_target_grid_ = {target_row, target_col};

  RCLCPP_INFO(logger_,
              "Selected cluster %d (%zu cells), target: (%.2f, %.2f)",
              best_idx, frontier_clusters_[best_idx].size(),
              centroid.first, centroid.second);

  is_navigating_ = true;
  publish_markers();
  navigate_to(centroid.first, centroid.second);
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ExplorerNode>());
  rclcpp::shutdown();
  return 0;
}
