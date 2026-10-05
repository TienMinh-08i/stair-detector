#ifndef EIGEN_NO_DEBUG
#define EIGEN_NO_DEBUG
#endif

#include <libobsensor/ObSensor.hpp>

#include <opencv2/opencv.hpp>

#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using NormalImage = std::vector<Eigen::Vector3f>;

struct Roi {
    int min_u = 0;
    int max_u = 0;
    int min_v = 0;
    int max_v = 0;
};

struct FloorPlane {
    Eigen::Vector3f normal = Eigen::Vector3f::UnitY();
    float d = 0.0f;
    int inliers = 0;
    bool valid = false;
};

struct FrameStats {
    int valid_points = 0;
    int normal_points = 0;
    int horizontal_points = 0;
    int component_count = 0;
};

struct Timings {
    double pointcloud_ms = 0.0;
    double cloud_ms = 0.0;
    double normals_ms = 0.0;
    double floor_ms = 0.0;
    double components_ms = 0.0;
    double vis_ms = 0.0;
};

struct Component {
    int id = -1;
    int num_points = 0;
    int min_u = 0;
    int max_u = 0;
    int min_v = 0;
    int max_v = 0;
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
};

struct PlanePatch {
    int component_id = -1;
    int num_points = 0;
    bool is_floor_reference = false;

    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f normal = Eigen::Vector3f::UnitY();

    float d = 0.0f;
    float height_from_floor = 0.0f;
    float width = 0.0f;
    float depth = 0.0f;
    float rms_error = 0.0f;
    int floor_roi_overlap = 0;

    int min_u = 0;
    int max_u = 0;
    int min_v = 0;
    int max_v = 0;
};

struct StairAnalysis {
    bool stairs_detected = false;
    bool depression_detected = false;
    bool region_valid = false;
    std::string stair_type = "NONE";
    int step_levels = 0;
    float riser_m = 0.0f;
    float confidence = 0.0f;
    float depression_depth_m = 0.0f;
    int stair_points = 0;
    int depression_points = 0;
    int min_u = 0;
    int min_v = 0;
    int max_u = 0;
    int max_v = 0;
};

struct TrackedComponent {
    int color_index = 0;
    int age = 0;
    int min_u = 0;
    int max_u = 0;
    int min_v = 0;
    int max_v = 0;
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
};

static constexpr float kMinDepthM = 0.30f;
static constexpr float kMaxDepthM = 4.00f;
static constexpr float kHorizontalAngleDeg = 20.0f;
static constexpr int kNormalPixelRadius = 2;
static constexpr float kNormalMaxNeighborDistanceM = 0.12f;
static constexpr int kNormalStride = 2;
static constexpr int kOverlayStride = 2;
static constexpr int kFloorNormalStride = 2;
static constexpr int kFloorMinNormals = 120;
static constexpr float kFloorClusterAngleDeg = 18.0f;
static constexpr float kFloorMaxUpdateAngleDeg = 25.0f;
static constexpr int kFloorPointStride = 3;
static constexpr int kFloorMinPoints = 600;
static constexpr int kMinComponentPoints = 1800;
static constexpr float kComponentMaxNeighborDistanceM = 0.22f;

static float degToRad(float deg) {
    return deg * static_cast<float>(M_PI) / 180.0f;
}

static float nanValue() {
    return std::numeric_limits<float>::quiet_NaN();
}

static std::string timestampBase() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::tm tm{};
    localtime_r(&t, &tm);

    std::ostringstream oss;
    oss << "capture_" << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

static bool validObPoint(const OBPoint &p) {
    return std::isfinite(p.x) &&
           std::isfinite(p.y) &&
           std::isfinite(p.z) &&
           p.z > 0.0f;
}

static Eigen::Vector3f pointMeters(const OBPoint &p) {
    return Eigen::Vector3f(
        p.x * 0.001f,
        p.y * 0.001f,
        p.z * 0.001f
    );
}

static bool validPclPoint(const pcl::PointXYZ &p) {
    return std::isfinite(p.x) &&
           std::isfinite(p.y) &&
           std::isfinite(p.z) &&
           p.z > 0.0f;
}

static bool validNormal(const Eigen::Vector3f &n) {
    return std::isfinite(n.x()) &&
           std::isfinite(n.y()) &&
           std::isfinite(n.z()) &&
           n.squaredNorm() > 0.25f;
}

static Roi usefulRoi(int w, int h) {
    Roi roi;
    roi.min_u = static_cast<int>(0.05f * w);
    roi.max_u = static_cast<int>(0.95f * w);
    roi.min_v = static_cast<int>(0.20f * h);
    roi.max_v = h;
    return roi;
}

static Roi floorRoi(int w, int h) {
    return usefulRoi(w, h);
}

static bool inRoi(int u, int v, const Roi &roi) {
    return u >= roi.min_u &&
           u < roi.max_u &&
           v >= roi.min_v &&
           v < roi.max_v;
}

static void makeOrganizedCloud(
    const std::shared_ptr<ob::Frame> &pointFrame,
    int w,
    int h,
    const Roi &roi,
    Cloud &cloud,
    FrameStats &stats
) {
    stats.valid_points = 0;
    stats.normal_points = 0;
    stats.horizontal_points = 0;
    stats.component_count = 0;

    cloud.width = static_cast<std::uint32_t>(w);
    cloud.height = static_cast<std::uint32_t>(h);
    cloud.is_dense = false;
    cloud.points.resize(static_cast<std::size_t>(w) * h);

    const float nan = nanValue();
    pcl::PointXYZ invalid;
    invalid.x = nan;
    invalid.y = nan;
    invalid.z = nan;
    std::fill(cloud.points.begin(), cloud.points.end(), invalid);

    if(!pointFrame ||
       !pointFrame->data() ||
       pointFrame->dataSize() == 0) {
        return;
    }

    const std::size_t count =
        pointFrame->dataSize() / sizeof(OBPoint);
    const std::size_t expected =
        static_cast<std::size_t>(w) * h;

    if(count < expected) {
        return;
    }

    const auto *points =
        reinterpret_cast<const OBPoint *>(pointFrame->data());

    for(int v = 0; v < h; ++v) {
        for(int u = 0; u < w; ++u) {
            const std::size_t idx =
                static_cast<std::size_t>(v) * w + u;

            if(!inRoi(u, v, roi)) {
                continue;
            }

            const OBPoint &src = points[idx];
            if(!validObPoint(src)) {
                continue;
            }

            const float x_m = src.x * 0.001f;
            const float y_m = src.y * 0.001f;
            const float z_m = src.z * 0.001f;

            if(z_m < kMinDepthM || z_m > kMaxDepthM) {
                continue;
            }

            auto &dst = cloud.points[idx];
            dst.x = x_m;
            dst.y = y_m;
            dst.z = z_m;
            stats.valid_points++;
        }
    }
}

static cv::Mat makeFastDepthGrayBgr(
    const std::shared_ptr<ob::DepthFrame> &depthFrame,
    float minDepthMm = 300.0f,
    float maxDepthMm = 4000.0f
) {
    const int w = depthFrame->width();
    const int h = depthFrame->height();
    const auto *raw =
        reinterpret_cast<const uint16_t *>(depthFrame->data());
    const float scale = depthFrame->getValueScale();

    cv::Mat bgr(h, w, CV_8UC3, cv::Scalar(0, 0, 0));

    for(int y = 0; y < h; ++y) {
        auto *dst = bgr.ptr<cv::Vec3b>(y);

        for(int x = 0; x < w; ++x) {
            const uint16_t dRaw = raw[y * w + x];
            if(dRaw == 0) {
                continue;
            }

            const float dMm = static_cast<float>(dRaw) * scale;
            if(dMm < minDepthMm || dMm > maxDepthMm) {
                continue;
            }

            const float t =
                (dMm - minDepthMm) / (maxDepthMm - minDepthMm);
            const float inv = 1.0f - std::clamp(t, 0.0f, 1.0f);
            const uint8_t g = static_cast<uint8_t>(255.0f * inv);
            dst[x] = cv::Vec3b(g, g, g);
        }
    }

    return bgr;
}

static cv::Mat makeDepthMillimeterImage(
    const std::shared_ptr<ob::DepthFrame> &depthFrame
) {
    const int w = depthFrame->width();
    const int h = depthFrame->height();
    const auto *raw =
        reinterpret_cast<const uint16_t *>(depthFrame->data());
    const float scale = depthFrame->getValueScale();

    cv::Mat depthMm(h, w, CV_16UC1, cv::Scalar(0));

    for(int y = 0; y < h; ++y) {
        auto *dst = depthMm.ptr<uint16_t>(y);
        for(int x = 0; x < w; ++x) {
            const uint16_t dRaw = raw[y * w + x];
            if(dRaw == 0) {
                continue;
            }

            const float dMm = static_cast<float>(dRaw) * scale;
            if(dMm > 0.0f && dMm <= 65535.0f) {
                dst[x] =
                    static_cast<uint16_t>(std::lround(dMm));
            }
        }
    }

    return depthMm;
}

static void computeOrganizedNormals(
    const Cloud &cloud,
    NormalImage &normals,
    FrameStats &stats
) {
    const int w = static_cast<int>(cloud.width);
    const int h = static_cast<int>(cloud.height);
    const float nan = nanValue();

    if(normals.size() != cloud.points.size()) {
        normals.resize(cloud.points.size());
    }

    const Eigen::Vector3f invalid(nan, nan, nan);
    std::fill(normals.begin(), normals.end(), invalid);

    if(w < 3 || h < 3 || cloud.points.size() != static_cast<std::size_t>(w) * h) {
        return;
    }

    stats.normal_points = 0;

    const float max_neighbor_dist2 =
        kNormalMaxNeighborDistanceM * kNormalMaxNeighborDistanceM;

    for(int v = kNormalPixelRadius;
        v < h - kNormalPixelRadius;
        v += kNormalStride) {

        for(int u = kNormalPixelRadius;
            u < w - kNormalPixelRadius;
            u += kNormalStride) {

            const std::size_t idx =
                static_cast<std::size_t>(v) * w + u;
            const auto &center = cloud.points[idx];

            if(!validPclPoint(center)) {
                continue;
            }

            const auto &left =
                cloud.points[
                    static_cast<std::size_t>(v) * w +
                    (u - kNormalPixelRadius)
                ];
            const auto &right =
                cloud.points[
                    static_cast<std::size_t>(v) * w +
                    (u + kNormalPixelRadius)
                ];
            const auto &up =
                cloud.points[
                    static_cast<std::size_t>(v - kNormalPixelRadius) * w +
                    u
                ];
            const auto &down =
                cloud.points[
                    static_cast<std::size_t>(v + kNormalPixelRadius) * w +
                    u
                ];

            if(!validPclPoint(left) ||
               !validPclPoint(right) ||
               !validPclPoint(up) ||
               !validPclPoint(down)) {
                continue;
            }

            auto dist2 = [](const pcl::PointXYZ &a,
                            const pcl::PointXYZ &b) {
                const float dx = a.x - b.x;
                const float dy = a.y - b.y;
                const float dz = a.z - b.z;
                return dx * dx + dy * dy + dz * dz;
            };

            if(dist2(left, center) > max_neighbor_dist2 ||
               dist2(right, center) > max_neighbor_dist2 ||
               dist2(up, center) > max_neighbor_dist2 ||
               dist2(down, center) > max_neighbor_dist2) {
                continue;
            }

            const float gx_x = right.x - left.x;
            const float gx_y = right.y - left.y;
            const float gx_z = right.z - left.z;

            const float gy_x = down.x - up.x;
            const float gy_y = down.y - up.y;
            const float gy_z = down.z - up.z;

            float nx = gx_y * gy_z - gx_z * gy_y;
            float ny = gx_z * gy_x - gx_x * gy_z;
            float nz = gx_x * gy_y - gx_y * gy_x;

            const float len2 = nx * nx + ny * ny + nz * nz;
            if(len2 < 1e-12f || !std::isfinite(len2)) {
                continue;
            }

            const float inv_len = 1.0f / std::sqrt(len2);
            nx *= inv_len;
            ny *= inv_len;
            nz *= inv_len;

            const Eigen::Vector3f normal(nx, ny, nz);

            for(int yy = v; yy < std::min(v + kNormalStride, h); ++yy) {
                for(int xx = u; xx < std::min(u + kNormalStride, w); ++xx) {
                    const std::size_t block_idx =
                        static_cast<std::size_t>(yy) * w + xx;

                    if(!validPclPoint(cloud.points[block_idx])) {
                        continue;
                    }

                    normals[block_idx] = normal;
                    stats.normal_points++;
                }
            }
        }
    }
}

static FloorPlane estimateFloorNormalFromNormals(
    const NormalImage &normals,
    int w,
    int h,
    const Roi &roi,
    const FloorPlane &previous
) {
    struct NormalCluster {
        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        int count = 0;
    };

    std::vector<NormalCluster> clusters;
    clusters.reserve(16);

    const float cluster_dot_threshold =
        std::cos(degToRad(kFloorClusterAngleDeg));

    for(int v = roi.min_v; v < roi.max_v; v += kFloorNormalStride) {
        for(int u = roi.min_u; u < roi.max_u; u += kFloorNormalStride) {
            if(u < 0 || u >= w || v < 0 || v >= h) {
                continue;
            }

            Eigen::Vector3f n =
                normals[static_cast<std::size_t>(v) * w + u];

            if(!validNormal(n)) {
                continue;
            }

            bool added = false;
            for(auto &cluster : clusters) {
                float dot = n.dot(cluster.mean);

                if(std::abs(dot) < cluster_dot_threshold) {
                    continue;
                }

                if(dot < 0.0f) {
                    n = -n;
                }

                cluster.sum += n;
                cluster.count++;
                const float len = cluster.sum.norm();
                if(len > 1e-6f && std::isfinite(len)) {
                    cluster.mean = cluster.sum / len;
                }

                added = true;
                break;
            }

            if(!added) {
                NormalCluster cluster;
                cluster.sum = n;
                cluster.mean = n;
                cluster.count = 1;
                clusters.push_back(cluster);
            }
        }
    }

    FloorPlane floor = previous;

    if(clusters.empty()) {
        return floor;
    }

    auto best_it =
        std::max_element(
            clusters.begin(),
            clusters.end(),
            [](const NormalCluster &a, const NormalCluster &b) {
                return a.count < b.count;
            }
        );

    if(best_it == clusters.end() ||
       best_it->count < kFloorMinNormals) {
        return floor;
    }

    Eigen::Vector3f n = best_it->mean;
    const float len = n.norm();
    if(len < 1e-6f || !std::isfinite(len)) {
        return floor;
    }

    n /= len;

    if(previous.valid) {
        float dot = n.dot(previous.normal);

        if(dot < 0.0f) {
            n = -n;
            dot = -dot;
        }

        const float update_dot_threshold =
            std::cos(degToRad(kFloorMaxUpdateAngleDeg));

        if(dot < update_dot_threshold) {
            return floor;
        }

        n = (0.92f * previous.normal + 0.08f * n).normalized();
    }

    floor.normal = n;
    floor.d = 0.0f;
    floor.inliers = best_it->count;
    floor.valid = true;
    return floor;
}

static FloorPlane estimateFloorPlaneFromPoints(
    const Cloud &cloud,
    const Roi &roi,
    const FloorPlane &previous
) {
    const int w = static_cast<int>(cloud.width);
    const int h = static_cast<int>(cloud.height);

    std::vector<Eigen::Vector3f> points;
    points.reserve(6000);

    for(int v = roi.min_v; v < roi.max_v; v += kFloorPointStride) {
        for(int u = roi.min_u; u < roi.max_u; u += kFloorPointStride) {
            if(u < 0 || u >= w || v < 0 || v >= h) {
                continue;
            }

            const auto &p =
                cloud.points[static_cast<std::size_t>(v) * w + u];

            if(!validPclPoint(p)) {
                continue;
            }

            points.emplace_back(p.x, p.y, p.z);
        }
    }

    FloorPlane floor = previous;

    if(static_cast<int>(points.size()) < kFloorMinPoints) {
        return floor;
    }

    constexpr int kRansacIterations = 120;
    constexpr float kPlaneDistanceThreshold = 0.025f;
    const float max_angle_dot = std::cos(degToRad(30.0f));
    const float threshold2 =
        kPlaneDistanceThreshold * kPlaneDistanceThreshold;
    uint32_t rng = 0x9e3779b9u + static_cast<uint32_t>(points.size());
    auto nextIndex = [&]() {
        rng = rng * 1664525u + 1013904223u;
        return static_cast<std::size_t>(rng) % points.size();
    };

    Eigen::Vector3f best_normal = Eigen::Vector3f::Zero();
    float best_d = 0.0f;
    int best_inliers = 0;
    for(int iteration = 0; iteration < kRansacIterations; ++iteration) {
        const std::size_t ia = nextIndex();
        const std::size_t ib = nextIndex();
        const std::size_t ic = nextIndex();
        if(ia == ib || ia == ic || ib == ic) {
            continue;
        }

        Eigen::Vector3f normal =
            (points[ib] - points[ia]).cross(points[ic] - points[ia]);
        const float norm = normal.norm();
        if(norm < 1e-5f) {
            continue;
        }
        normal /= norm;

        if(previous.valid) {
            float alignment = normal.dot(previous.normal);
            if(alignment < 0.0f) {
                normal = -normal;
                alignment = -alignment;
            }
            if(alignment < max_angle_dot) {
                continue;
            }
        }
        else if(normal.y() < 0.0f) {
            normal = -normal;
        }

        const float d = -normal.dot(points[ia]);
        int inliers = 0;
        for(const Eigen::Vector3f &point : points) {
            const float distance = normal.dot(point) + d;
            if(distance * distance <= threshold2) {
                ++inliers;
            }
        }

        if(inliers > best_inliers) {
            best_inliers = inliers;
            best_normal = normal;
            best_d = d;
        }
    }

    if(best_inliers < kFloorMinPoints) {
        return floor;
    }

    const Eigen::Vector3f model_normal = best_normal;
    const float model_d = best_d;

    if(previous.valid) {
        best_normal =
            (0.85f * previous.normal + 0.15f * best_normal).normalized();
    }

    double offset_sum = 0.0;
    int offset_count = 0;
    for(const Eigen::Vector3f &point : points) {
        const float distance = model_normal.dot(point) + model_d;
        if(distance * distance <= threshold2) {
            offset_sum -= static_cast<double>(best_normal.dot(point));
            offset_count++;
        }
    }

    floor.normal = best_normal;
    floor.d = offset_count > 0
        ? static_cast<float>(offset_sum / offset_count)
        : best_d;
    floor.inliers = best_inliers;
    floor.valid = true;
    return floor;
}

static cv::Mat makeHorizontalMaskOverlay(
    const cv::Mat &depthVis,
    const Cloud &cloud,
    const NormalImage &normals,
    const FloorPlane &floor,
    float cosThreshold,
    std::vector<uint8_t> &horizontalMask,
    FrameStats &stats
) {
    cv::Mat overlay = depthVis.clone();

    stats.horizontal_points = 0;
    horizontalMask.assign(cloud.points.size(), 0);

    if(!floor.valid ||
       normals.size() != cloud.points.size()) {
        return overlay;
    }

    const int w = static_cast<int>(cloud.width);
    const int h = static_cast<int>(cloud.height);

    for(int v = 0; v < h; v += kOverlayStride) {
        for(int u = 0; u < w; u += kOverlayStride) {
            const std::size_t idx =
                static_cast<std::size_t>(v) * w + u;

            if(!validPclPoint(cloud.points[idx])) {
                continue;
            }

            Eigen::Vector3f n = normals[idx];
            if(!validNormal(n)) {
                continue;
            }

            const float score =
                std::abs(n.dot(floor.normal));

            if(score <= cosThreshold) {
                continue;
            }

            for(int yy = v; yy < std::min(v + kOverlayStride, h); ++yy) {
                auto *dst = overlay.ptr<cv::Vec3b>(yy);

                for(int xx = u; xx < std::min(u + kOverlayStride, w); ++xx) {
                    const std::size_t block_idx =
                        static_cast<std::size_t>(yy) * w + xx;

                    if(!validPclPoint(cloud.points[block_idx])) {
                        continue;
                    }

                    horizontalMask[block_idx] = 1;
                    stats.horizontal_points++;
                    dst[xx] = cv::Vec3b(0, 0, 255);
                }
            }
        }
    }

    return overlay;
}

static void smoothHorizontalMask(
    const Cloud &cloud,
    std::vector<uint8_t> &horizontalMask,
    FrameStats &stats
) {
    const int w = static_cast<int>(cloud.width);
    const int h = static_cast<int>(cloud.height);

    if(w <= 0 ||
       h <= 0 ||
       horizontalMask.size() != cloud.points.size()) {
        return;
    }

    cv::Mat mask(h, w, CV_8UC1, cv::Scalar(0));

    for(int y = 0; y < h; ++y) {
        auto *dst = mask.ptr<uint8_t>(y);

        for(int x = 0; x < w; ++x) {
            const std::size_t idx =
                static_cast<std::size_t>(y) * w + x;

            if(horizontalMask[idx] != 0 &&
               validPclPoint(cloud.points[idx])) {
                dst[x] = 255;
            }
        }
    }

    cv::Mat closed;
    cv::morphologyEx(
        mask,
        closed,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(9, 7)
        )
    );

    cv::Mat opened;
    cv::morphologyEx(
        closed,
        opened,
        cv::MORPH_OPEN,
        cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(3, 3)
        )
    );

    stats.horizontal_points = 0;

    for(int y = 0; y < h; ++y) {
        const auto *src = opened.ptr<uint8_t>(y);

        for(int x = 0; x < w; ++x) {
            const std::size_t idx =
                static_cast<std::size_t>(y) * w + x;

            const bool keep =
                src[x] != 0 &&
                validPclPoint(cloud.points[idx]);

            horizontalMask[idx] = keep ? 1 : 0;

            if(keep) {
                stats.horizontal_points++;
            }
        }
    }
}

static float pointDistance2(
    const pcl::PointXYZ &a,
    const pcl::PointXYZ &b
) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

static std::vector<Component> extractHorizontalComponents(
    const Cloud &cloud,
    const NormalImage &normals,
    const std::vector<uint8_t> &horizontalMask,
    std::vector<int> &labels
) {
    std::vector<Component> components;

    const int w = static_cast<int>(cloud.width);
    const int h = static_cast<int>(cloud.height);
    const std::size_t total = cloud.points.size();

    labels.assign(total, -1);

    if(total == 0 ||
       normals.size() != total ||
       horizontalMask.size() != total ||
       w <= 0 ||
       h <= 0) {
        return components;
    }

    const float max_dist2 =
        kComponentMaxNeighborDistanceM *
        kComponentMaxNeighborDistanceM;

    std::vector<int> queue;
    queue.reserve(4096);

    auto canConnect = [&](int a_idx, int b_idx) {
        if(horizontalMask[static_cast<std::size_t>(b_idx)] == 0 ||
           labels[static_cast<std::size_t>(b_idx)] != -1) {
            return false;
        }

        const auto &pa = cloud.points[static_cast<std::size_t>(a_idx)];
        const auto &pb = cloud.points[static_cast<std::size_t>(b_idx)];

        if(!validPclPoint(pa) || !validPclPoint(pb)) {
            return false;
        }

        if(pointDistance2(pa, pb) > max_dist2) {
            return false;
        }

        return true;
    };

    int next_label = 0;

    for(int v = 0; v < h; ++v) {
        for(int u = 0; u < w; ++u) {
            const int seed_idx = v * w + u;
            const std::size_t seed_sidx =
                static_cast<std::size_t>(seed_idx);

            if(horizontalMask[seed_sidx] == 0 ||
               labels[seed_sidx] != -1 ||
               !validPclPoint(cloud.points[seed_sidx]) ||
               !validNormal(normals[seed_sidx])) {
                continue;
            }

            Component comp;
            comp.id = next_label;
            comp.min_u = comp.max_u = u;
            comp.min_v = comp.max_v = v;

            Eigen::Vector3f sum(0.0f, 0.0f, 0.0f);

            queue.clear();
            queue.push_back(seed_idx);
            labels[seed_sidx] = next_label;

            std::size_t head = 0;
            while(head < queue.size()) {
                const int idx = queue[head++];
                const int cu = idx % w;
                const int cv = idx / w;

                const auto &p = cloud.points[static_cast<std::size_t>(idx)];
                sum += Eigen::Vector3f(p.x, p.y, p.z);
                comp.num_points++;
                comp.min_u = std::min(comp.min_u, cu);
                comp.max_u = std::max(comp.max_u, cu);
                comp.min_v = std::min(comp.min_v, cv);
                comp.max_v = std::max(comp.max_v, cv);

                const int neighbors[8][2] = {
                    {cu - 1, cv},
                    {cu + 1, cv},
                    {cu, cv - 1},
                    {cu, cv + 1},
                    {cu - 1, cv - 1},
                    {cu + 1, cv - 1},
                    {cu - 1, cv + 1},
                    {cu + 1, cv + 1}
                };

                for(const auto &nb : neighbors) {
                    const int nu = nb[0];
                    const int nv = nb[1];

                    if(nu < 0 || nu >= w || nv < 0 || nv >= h) {
                        continue;
                    }

                    const int nidx = nv * w + nu;
                    if(!canConnect(idx, nidx)) {
                        continue;
                    }

                    labels[static_cast<std::size_t>(nidx)] = next_label;
                    queue.push_back(nidx);
                }
            }

            if(comp.num_points >= kMinComponentPoints) {
                comp.centroid = sum / static_cast<float>(comp.num_points);
                components.push_back(comp);
                next_label++;
            }
            else {
                for(int idx : queue) {
                    labels[static_cast<std::size_t>(idx)] = -2;
                }
            }
        }
    }

    return components;
}

static cv::Mat makeComponentOverlay(
    const cv::Mat &depthVis,
    const std::vector<int> &labels,
    const std::vector<Component> &components,
    const std::vector<int> &componentColorIndex
) {
    cv::Mat overlay = depthVis.clone();

    if(labels.empty() || components.empty()) {
        return overlay;
    }

    static const cv::Vec3b colors[] = {
        cv::Vec3b(40, 40, 255),
        cv::Vec3b(40, 220, 80),
        cv::Vec3b(255, 120, 40),
        cv::Vec3b(40, 220, 255),
        cv::Vec3b(220, 70, 220),
        cv::Vec3b(220, 220, 50),
        cv::Vec3b(90, 140, 255),
        cv::Vec3b(120, 255, 180),
        cv::Vec3b(255, 180, 120),
        cv::Vec3b(180, 120, 255)
    };

    const int color_count =
        static_cast<int>(sizeof(colors) / sizeof(colors[0]));

    for(int y = 0; y < overlay.rows; ++y) {
        auto *dst = overlay.ptr<cv::Vec3b>(y);
        for(int x = 0; x < overlay.cols; ++x) {
            const int label =
                labels[static_cast<std::size_t>(y) * overlay.cols + x];

            if(label < 0) {
                continue;
            }

            int color_index = label % color_count;
            if(label < static_cast<int>(componentColorIndex.size()) &&
               componentColorIndex[static_cast<std::size_t>(label)] >= 0) {
                color_index =
                    componentColorIndex[static_cast<std::size_t>(label)] %
                    color_count;
            }

            dst[x] = colors[color_index];
        }
    }

    for(const auto &comp : components) {
        cv::rectangle(
            overlay,
            cv::Rect(
                comp.min_u,
                comp.min_v,
                std::max(1, comp.max_u - comp.min_u + 1),
                std::max(1, comp.max_v - comp.min_v + 1)
            ),
            cv::Scalar(255, 255, 255),
            1,
            cv::LINE_AA
        );
    }

    return overlay;
}

static float bboxIoU(
    int a_min_u,
    int a_min_v,
    int a_max_u,
    int a_max_v,
    int b_min_u,
    int b_min_v,
    int b_max_u,
    int b_max_v
) {
    const int ix0 = std::max(a_min_u, b_min_u);
    const int iy0 = std::max(a_min_v, b_min_v);
    const int ix1 = std::min(a_max_u, b_max_u);
    const int iy1 = std::min(a_max_v, b_max_v);

    const int iw = std::max(0, ix1 - ix0 + 1);
    const int ih = std::max(0, iy1 - iy0 + 1);
    const int intersection = iw * ih;

    const int area_a =
        std::max(1, a_max_u - a_min_u + 1) *
        std::max(1, a_max_v - a_min_v + 1);
    const int area_b =
        std::max(1, b_max_u - b_min_u + 1) *
        std::max(1, b_max_v - b_min_v + 1);

    const int uni = area_a + area_b - intersection;
    if(uni <= 0) {
        return 0.0f;
    }

    return static_cast<float>(intersection) / static_cast<float>(uni);
}

static std::vector<int> updateComponentColorTracking(
    const std::vector<Component> &components,
    std::vector<TrackedComponent> &trackedComponents,
    int &nextColorIndex
) {
    std::vector<int> color_by_id;
    int max_id = -1;
    for(const auto &component : components) {
        max_id = std::max(max_id, component.id);
    }

    if(max_id >= 0) {
        color_by_id.assign(static_cast<std::size_t>(max_id + 1), -1);
    }

    std::vector<uint8_t> used(
        trackedComponents.size(),
        static_cast<uint8_t>(0)
    );
    std::vector<TrackedComponent> next_tracked;
    next_tracked.reserve(components.size());

    for(const auto &component : components) {
        int best_index = -1;
        float best_score = -1.0f;

        for(std::size_t i = 0; i < trackedComponents.size(); ++i) {
            if(used[i] != 0) {
                continue;
            }

            const auto &tracked = trackedComponents[i];
            const float centroid_dist =
                (component.centroid - tracked.centroid).norm();
            const float iou =
                bboxIoU(
                    component.min_u,
                    component.min_v,
                    component.max_u,
                    component.max_v,
                    tracked.min_u,
                    tracked.min_v,
                    tracked.max_u,
                    tracked.max_v
                );

            if(centroid_dist > 0.35f && iou < 0.08f) {
                continue;
            }

            const float score =
                iou * 2.0f + std::max(0.0f, 1.0f - centroid_dist / 0.35f);

            if(score > best_score) {
                best_score = score;
                best_index = static_cast<int>(i);
            }
        }

        int color_index = 0;
        if(best_index >= 0) {
            used[static_cast<std::size_t>(best_index)] = 1;
            color_index =
                trackedComponents[static_cast<std::size_t>(best_index)]
                    .color_index;
        }
        else {
            color_index = nextColorIndex++;
        }

        if(component.id >= 0 &&
           component.id < static_cast<int>(color_by_id.size())) {
            color_by_id[static_cast<std::size_t>(component.id)] = color_index;
        }

        TrackedComponent tracked;
        tracked.color_index = color_index;
        tracked.age =
            best_index >= 0
                ? trackedComponents[static_cast<std::size_t>(best_index)].age + 1
                : 1;
        tracked.min_u = component.min_u;
        tracked.max_u = component.max_u;
        tracked.min_v = component.min_v;
        tracked.max_v = component.max_v;
        tracked.centroid = component.centroid;
        next_tracked.push_back(tracked);
    }

    trackedComponents = next_tracked;
    return color_by_id;
}

static std::vector<PlanePatch> fitPlanePatches(
    const Cloud &cloud,
    const std::vector<int> &labels,
    const std::vector<Component> &components,
    const FloorPlane &floor,
    const Roi &floor_roi
) {
    std::vector<PlanePatch> patches;

    if(components.empty() ||
       labels.size() != cloud.points.size()) {
        return patches;
    }

    patches.reserve(components.size());

    for(const auto &component : components) {
        Eigen::Vector3f centroid(0.0f, 0.0f, 0.0f);
        int count = 0;

        for(std::size_t i = 0; i < labels.size(); ++i) {
            if(labels[i] != component.id) {
                continue;
            }

            const auto &p = cloud.points[i];
            if(!validPclPoint(p)) {
                continue;
            }

            centroid += Eigen::Vector3f(p.x, p.y, p.z);
            count++;
        }

        if(count < kMinComponentPoints) {
            continue;
        }

        centroid /= static_cast<float>(count);

        Eigen::Matrix3f covariance =
            Eigen::Matrix3f::Zero();

        for(std::size_t i = 0; i < labels.size(); ++i) {
            if(labels[i] != component.id) {
                continue;
            }

            const auto &p = cloud.points[i];
            if(!validPclPoint(p)) {
                continue;
            }

            const Eigen::Vector3f q =
                Eigen::Vector3f(p.x, p.y, p.z) - centroid;

            covariance += q * q.transpose();
        }

        covariance /= static_cast<float>(count);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        if(solver.info() != Eigen::Success) {
            continue;
        }

        Eigen::Vector3f normal =
            solver.eigenvectors().col(0).normalized();

        if(floor.valid && normal.dot(floor.normal) < 0.0f) {
            normal = -normal;
        }

        const Eigen::Vector3f axis_depth =
            solver.eigenvectors().col(2).normalized();
        const Eigen::Vector3f axis_width =
            solver.eigenvectors().col(1).normalized();

        float min_depth = std::numeric_limits<float>::max();
        float max_depth = -std::numeric_limits<float>::max();
        float min_width = std::numeric_limits<float>::max();
        float max_width = -std::numeric_limits<float>::max();
        double squared_error_sum = 0.0;
        int floor_roi_overlap = 0;

        for(std::size_t i = 0; i < labels.size(); ++i) {
            if(labels[i] != component.id) {
                continue;
            }

            const auto &p = cloud.points[i];
            if(!validPclPoint(p)) {
                continue;
            }

            const Eigen::Vector3f q =
                Eigen::Vector3f(p.x, p.y, p.z) - centroid;

            const float depth_proj = q.dot(axis_depth);
            const float width_proj = q.dot(axis_width);
            const float plane_dist = q.dot(normal);

            min_depth = std::min(min_depth, depth_proj);
            max_depth = std::max(max_depth, depth_proj);
            min_width = std::min(min_width, width_proj);
            max_width = std::max(max_width, width_proj);

            squared_error_sum +=
                static_cast<double>(plane_dist) *
                static_cast<double>(plane_dist);

            const int pixel = static_cast<int>(i);
            const int u = pixel % static_cast<int>(cloud.width);
            const int v = pixel / static_cast<int>(cloud.width);

            if(inRoi(u, v, floor_roi)) {
                floor_roi_overlap++;
            }
        }

        PlanePatch patch;
        patch.component_id = component.id;
        patch.num_points = count;
        patch.centroid = centroid;
        patch.normal = normal;
        patch.d = -normal.dot(centroid);
        patch.width = max_width - min_width;
        patch.depth = max_depth - min_depth;
        patch.rms_error =
            static_cast<float>(
                std::sqrt(squared_error_sum / count)
            );
        patch.min_u = component.min_u;
        patch.max_u = component.max_u;
        patch.min_v = component.min_v;
        patch.max_v = component.max_v;
        patch.floor_roi_overlap = floor_roi_overlap;

        patches.push_back(patch);
    }

    if(patches.empty()) {
        return patches;
    }

    auto largest_patch_it =
        std::max_element(
            patches.begin(),
            patches.end(),
            [](const PlanePatch &a, const PlanePatch &b) {
                return a.num_points < b.num_points;
            }
        );

    Eigen::Vector3f floor_normal = floor.valid
        ? floor.normal.normalized()
        : largest_patch_it->normal.normalized();
    const float floor_d = floor.valid
        ? floor.d
        : -floor_normal.dot(largest_patch_it->centroid);

    for(auto &patch : patches) {
        patch.height_from_floor =
            floor_normal.dot(patch.centroid) + floor_d;

        if(std::abs(patch.height_from_floor) < 0.005f) {
            patch.height_from_floor = 0.0f;
        }

        patch.is_floor_reference = false;
    }

    if(floor.valid) {
        const float parallel_threshold =
            std::cos(degToRad(25.0f));
        auto floor_patch_it = patches.end();
        float best_abs_height = std::numeric_limits<float>::max();

        for(auto it = patches.begin(); it != patches.end(); ++it) {
            const float parallel_score =
                std::abs(it->normal.dot(floor_normal));
            const float abs_height =
                std::abs(it->height_from_floor);

            if(parallel_score < parallel_threshold || abs_height > 0.08f) {
                continue;
            }

            if(floor_patch_it == patches.end() ||
               abs_height < best_abs_height ||
               (std::abs(abs_height - best_abs_height) < 0.01f &&
                it->num_points > floor_patch_it->num_points)) {
                floor_patch_it = it;
                best_abs_height = abs_height;
            }
        }

        if(floor_patch_it != patches.end()) {
            floor_patch_it->is_floor_reference = true;
        }
    }
    else {
        largest_patch_it->is_floor_reference = true;
    }

    std::sort(
        patches.begin(),
        patches.end(),
        [](const PlanePatch &a, const PlanePatch &b) {
            if(std::abs(a.height_from_floor - b.height_from_floor) > 0.015f) {
                return a.height_from_floor < b.height_from_floor;
            }

            return a.num_points > b.num_points;
        }
    );

    return patches;
}

static void growBox(StairAnalysis &analysis, const PlanePatch &patch) {
    if(!analysis.region_valid) {
        analysis.min_u = patch.min_u;
        analysis.min_v = patch.min_v;
        analysis.max_u = patch.max_u;
        analysis.max_v = patch.max_v;
        analysis.region_valid = true;
        return;
    }

    analysis.min_u = std::min(analysis.min_u, patch.min_u);
    analysis.min_v = std::min(analysis.min_v, patch.min_v);
    analysis.max_u = std::max(analysis.max_u, patch.max_u);
    analysis.max_v = std::max(analysis.max_v, patch.max_v);
}

static int countHeightLevels(std::vector<float> heights, float tolerance) {
    if(heights.empty()) {
        return 0;
    }

    std::sort(heights.begin(), heights.end());

    int levels = 1;
    float current = heights.front();

    for(float h : heights) {
        if(std::abs(h - current) > tolerance) {
            levels++;
            current = h;
        }
    }

    return levels;
}

static float medianStepHeight(std::vector<float> heights) {
    if(heights.empty()) {
        return 0.0f;
    }

    std::sort(heights.begin(), heights.end());
    std::vector<float> diffs;
    diffs.reserve(heights.size());

    float previous = 0.0f;
    for(float h : heights) {
        const float diff = h - previous;
        if(diff >= 0.08f && diff <= 0.28f) {
            diffs.push_back(diff);
            previous = h;
        }
    }

    if(diffs.empty()) {
        return 0.0f;
    }

    std::sort(diffs.begin(), diffs.end());
    return diffs[diffs.size() / 2];
}

static StairAnalysis analyzeStairsAndDepressions(
    const std::vector<PlanePatch> &patches,
    const FloorPlane &floor
) {
    StairAnalysis analysis;

    if(!floor.valid || patches.empty()) {
        return analysis;
    }

    const Eigen::Vector3f floor_normal = floor.normal.normalized();
    const float parallel_threshold = std::cos(degToRad(30.0f));

    std::vector<float> positive_heights;
    std::vector<float> negative_heights;
    std::vector<const PlanePatch *> positive_patches;
    std::vector<const PlanePatch *> negative_patches;
    int positive_points = 0;
    int negative_points = 0;
    int positive_planes = 0;
    int negative_planes = 0;

    for(const auto &patch : patches) {
        if(patch.is_floor_reference) {
            continue;
        }

        const float parallel_score =
            std::abs(patch.normal.normalized().dot(floor_normal));
        if(parallel_score < parallel_threshold) {
            continue;
        }

        if(patch.rms_error > 0.035f ||
           patch.num_points < 1800 ||
           std::max(patch.width, patch.depth) < 0.28f) {
            continue;
        }

        const float height = patch.height_from_floor;
        const float abs_height = std::abs(height);
        if(abs_height < 0.08f || abs_height > 0.80f) {
            continue;
        }

        if(height > 0.0f) {
            positive_heights.push_back(abs_height);
            positive_patches.push_back(&patch);
            positive_points += patch.num_points;
            positive_planes++;
        }
        else {
            negative_heights.push_back(abs_height);
            negative_patches.push_back(&patch);
            negative_points += patch.num_points;
            negative_planes++;

            const float smaller_extent =
                std::min(patch.width, patch.depth);
            const float larger_extent =
                std::max(patch.width, patch.depth);
            const int bbox_width = patch.max_u - patch.min_u + 1;
            const int bbox_height = patch.max_v - patch.min_v + 1;
            const int bbox_area = bbox_width * bbox_height;

            if(abs_height >= 0.12f &&
               patch.num_points >= 7000 &&
               smaller_extent >= 0.25f &&
               larger_extent >= 0.75f &&
               bbox_area >= 18000) {
                analysis.depression_detected = true;
                analysis.depression_depth_m =
                    std::max(analysis.depression_depth_m, abs_height);
                analysis.depression_points += patch.num_points;
                growBox(analysis, patch);
            }
        }
    }

    const int positive_levels = countHeightLevels(positive_heights, 0.06f);
    const int negative_levels = countHeightLevels(negative_heights, 0.06f);
    const float positive_riser = medianStepHeight(positive_heights);
    const float negative_riser = medianStepHeight(negative_heights);

    const bool positive_stair =
        positive_levels >= 2 ||
        (positive_levels == 1 &&
         positive_planes >= 1 &&
         positive_points >= 6000 &&
         positive_riser >= 0.08f &&
         positive_riser <= 0.28f);

    const bool negative_stair =
        negative_levels >= 2 ||
        (negative_levels == 1 &&
         negative_planes >= 1 &&
         negative_points >= 6000 &&
         negative_riser >= 0.08f &&
         negative_riser <= 0.28f);

    if(positive_stair || negative_stair) {
        const bool choose_positive =
            positive_levels > negative_levels ||
            (positive_levels == negative_levels &&
             positive_points >= negative_points);

        analysis.stairs_detected = true;
        analysis.stair_type = choose_positive ? "UP_OR_RAISED" : "DOWN_OR_LOWER";
        analysis.step_levels = choose_positive ? positive_levels : negative_levels;
        analysis.riser_m = choose_positive ? positive_riser : negative_riser;
        analysis.stair_points = choose_positive ? positive_points : negative_points;

        const auto &selected_patches =
            choose_positive ? positive_patches : negative_patches;
        for(const auto *patch : selected_patches) {
            growBox(analysis, *patch);
        }

        float confidence = 0.45f;
        confidence += 0.18f * static_cast<float>(analysis.step_levels);
        confidence += std::min(0.20f, analysis.stair_points / 60000.0f);
        if(analysis.riser_m >= 0.12f && analysis.riser_m <= 0.22f) {
            confidence += 0.15f;
        }
        analysis.confidence = std::min(0.95f, confidence);
    }

    return analysis;
}

static void drawRoi(cv::Mat &image, const Roi &roi, const cv::Scalar &color) {
    cv::rectangle(
        image,
        cv::Rect(
            roi.min_u,
            roi.min_v,
            std::max(1, roi.max_u - roi.min_u),
            std::max(1, roi.max_v - roi.min_v)
        ),
        color,
        1,
        cv::LINE_AA
    );
}

static void drawDetectionOverlay(
    cv::Mat &image,
    const StairAnalysis &analysis,
    double fps
) {
    const bool alert =
        analysis.stairs_detected || analysis.depression_detected;

    const cv::Scalar color =
        analysis.depression_detected
            ? cv::Scalar(0, 0, 255)
            : (analysis.stairs_detected
                ? cv::Scalar(0, 220, 255)
                : cv::Scalar(0, 220, 0));

    std::ostringstream stair_label;
    if(analysis.stairs_detected) {
        stair_label
            << "STAIRS: YES " << analysis.stair_type
            << " levels=" << analysis.step_levels
            << " riser=" << std::fixed << std::setprecision(2)
            << analysis.riser_m << "m"
            << " conf=" << analysis.confidence;
    }
    else {
        stair_label << "STAIRS: NO";
    }

    std::ostringstream depression_label;
    if(analysis.depression_detected) {
        depression_label
            << "DEPRESSION: YES depth="
            << std::fixed << std::setprecision(2)
            << analysis.depression_depth_m << "m";
    }
    else {
        depression_label << "DEPRESSION: NO";
    }

    cv::rectangle(
        image,
        cv::Rect(0, 0, image.cols, 62),
        alert ? cv::Scalar(20, 20, 20) : cv::Scalar(10, 50, 10),
        cv::FILLED
    );

    cv::putText(
        image,
        stair_label.str(),
        cv::Point(10, 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv::LINE_AA
    );

    cv::putText(
        image,
        depression_label.str(),
        cv::Point(10, 50),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv::LINE_AA
    );

    std::ostringstream fps_label;
    fps_label << "FPS: " << std::fixed << std::setprecision(1) << fps;

    cv::putText(
        image,
        fps_label.str(),
        cv::Point(std::max(10, image.cols - 125), 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA
    );

    if(analysis.region_valid) {
        const int x0 = std::clamp(analysis.min_u, 0, image.cols - 1);
        const int y0 = std::clamp(analysis.min_v, 0, image.rows - 1);
        const int x1 = std::clamp(analysis.max_u, 0, image.cols - 1);
        const int y1 = std::clamp(analysis.max_v, 0, image.rows - 1);

        cv::rectangle(
            image,
            cv::Rect(
                x0,
                y0,
                std::max(1, x1 - x0 + 1),
                std::max(1, y1 - y0 + 1)
            ),
            color,
            3,
            cv::LINE_AA
        );
    }
}

static bool saveOrganizedCloudPly(const Cloud &cloud, const std::string &path) {
    Cloud valid;
    valid.is_dense = false;
    valid.points.reserve(cloud.points.size());

    for(const auto &p : cloud.points) {
        if(validPclPoint(p)) {
            valid.points.push_back(p);
        }
    }

    valid.width = static_cast<std::uint32_t>(valid.points.size());
    valid.height = 1;

    return pcl::io::savePLYFileASCII(path, valid) == 0;
}

int main(int argc, char **argv) {
    const std::string outputDir =
        argc >= 2 ? argv[1] : "captures";

    std::filesystem::create_directories(outputDir);

    try {
        ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_ERROR);

        ob::Pipeline pipeline;
        auto config = std::make_shared<ob::Config>();

        auto depthProfiles =
            pipeline.getStreamProfileList(OB_SENSOR_DEPTH);

        if(!depthProfiles || depthProfiles->count() == 0) {
            std::cerr << "No depth profile.\n";
            return 1;
        }

        auto depthProfile =
            depthProfiles->getProfile(OB_PROFILE_DEFAULT);

        config->enableStream(depthProfile);
        pipeline.start(config);

        ob::PointCloudFilter pcFilter;
        pcFilter.setCameraParam(pipeline.getCameraParam());
        pcFilter.setCreatePointFormat(OB_FORMAT_POINT);

        Cloud::Ptr organized(new Cloud);
        NormalImage normals;
        std::vector<uint8_t> horizontalMask;
        std::vector<int> componentLabels;
        std::vector<Component> components;
        std::vector<int> componentColorIndex;
        std::vector<TrackedComponent> trackedComponents;
        std::vector<PlanePatch> planePatches;
        StairAnalysis stairAnalysis;
        FloorPlane floor;
        FrameStats stats;
        Timings timings;
        bool floorLocked = false;

        cv::namedWindow("Horizontal Components", cv::WINDOW_NORMAL);
        cv::resizeWindow("Horizontal Components", 1280, 600);

        int frameCounter = 0;
        int nextColorIndex = 0;
        int fpsFrames = 0;
        double fps = 0.0;
        auto fpsStart = std::chrono::steady_clock::now();
        auto printStart = fpsStart;

        const float cosThreshold =
            std::cos(degToRad(kHorizontalAngleDeg));

        std::cout
            << "\n===== REALTIME STAIR + DEPRESSION DETECTOR =====\n"
            << "Pipeline: Depth -> Organized PointXYZ -> organized-grid normals "
            << "-> locked floor plane -> horizontal planes -> stair/depression decision\n"
            << "Fast settings: normal stride=" << kNormalStride
            << ", overlay stride=" << kOverlayStride << "\n"
            << "F : lock/unlock current floor plane (normal + offset)\n"
            << "S : save current PLY + depth + overlay\n"
            << "Q / ESC : quit\n\n";

        while(true) {
            auto frameset = pipeline.waitForFrames(100);
            if(!frameset || !frameset->depthFrame()) {
                continue;
            }

            auto depthFrame = frameset->depthFrame();
            const int w = depthFrame->width();
            const int h = depthFrame->height();

            const auto t0 = std::chrono::steady_clock::now();
            const float depthScale = depthFrame->getValueScale();
            pcFilter.setPositionDataScaled(depthScale);
            auto pointFrame = pcFilter.process(frameset);
            const auto t1 = std::chrono::steady_clock::now();

            const Roi roi = usefulRoi(w, h);
            const Roi floor_roi = floorRoi(w, h);

            makeOrganizedCloud(
                pointFrame,
                w,
                h,
                roi,
                *organized,
                stats
            );
            const auto t2 = std::chrono::steady_clock::now();

            computeOrganizedNormals(*organized, normals, stats);
            const auto t3 = std::chrono::steady_clock::now();

            if(!floorLocked) {
                floor =
                    estimateFloorPlaneFromPoints(
                        *organized,
                        floor_roi,
                        floor
                    );
            }
            const auto t4 = std::chrono::steady_clock::now();

            cv::Mat depthVis = makeFastDepthGrayBgr(depthFrame);
            cv::Mat horizontalOverlay =
                makeHorizontalMaskOverlay(
                    depthVis,
                    *organized,
                    normals,
                    floor,
                    cosThreshold,
                    horizontalMask,
                    stats
                );

            smoothHorizontalMask(*organized, horizontalMask, stats);

            const auto t5 = std::chrono::steady_clock::now();

            components =
                extractHorizontalComponents(
                    *organized,
                    normals,
                    horizontalMask,
                    componentLabels
                );

            stats.component_count =
                static_cast<int>(components.size());

            componentColorIndex =
                updateComponentColorTracking(
                    components,
                    trackedComponents,
                    nextColorIndex
                );

            planePatches =
                fitPlanePatches(
                    *organized,
                    componentLabels,
                    components,
                    floor,
                    floor_roi
                );

            stairAnalysis =
                analyzeStairsAndDepressions(
                    planePatches,
                    floor
                );

            const auto t6 = std::chrono::steady_clock::now();

            cv::Mat overlay =
                makeComponentOverlay(
                    depthVis,
                    componentLabels,
                    components,
                    componentColorIndex
                );

            if(components.empty()) {
                overlay = horizontalOverlay;
            }

            fpsFrames++;
            frameCounter++;
            const auto now = std::chrono::steady_clock::now();
            const double fpsElapsed =
                std::chrono::duration<double>(now - fpsStart).count();

            if(fpsElapsed >= 1.0) {
                fps = static_cast<double>(fpsFrames) / fpsElapsed;
                fpsFrames = 0;
                fpsStart = now;
            }

            drawRoi(overlay, roi, cv::Scalar(180, 180, 180));
            drawDetectionOverlay(overlay, stairAnalysis, fps);
            const auto t7 = std::chrono::steady_clock::now();

            timings.pointcloud_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            timings.cloud_ms =
                std::chrono::duration<double, std::milli>(t2 - t1).count();
            timings.normals_ms =
                std::chrono::duration<double, std::milli>(t3 - t2).count();
            timings.floor_ms =
                std::chrono::duration<double, std::milli>(t4 - t3).count();
            timings.components_ms =
                std::chrono::duration<double, std::milli>(t6 - t5).count();
            timings.vis_ms =
                std::chrono::duration<double, std::milli>(
                    (t5 - t4) + (t7 - t6)
                ).count();

            const double printElapsed =
                std::chrono::duration<double>(now - printStart).count();

            if(printElapsed >= 1.0) {
                printStart = now;

                std::cout
                    << std::fixed
                    << std::setprecision(2)
                    << "FPS=" << fps
                    << " | organized=" << organized->width
                    << "x" << organized->height
                    << " | valid=" << stats.valid_points
                    << " | normals=" << stats.normal_points
                    << " | horizontal=" << stats.horizontal_points
                    << " | components=" << stats.component_count
                    << " | planes=" << planePatches.size()
                    << " | floor_valid=" << (floor.valid ? "yes" : "no")
                    << " | floor_lock=" << (floorLocked ? "yes" : "no")
                    << " | floor_inliers=" << floor.inliers
                    << " | floor_normal=("
                    << floor.normal.x() << ", "
                    << floor.normal.y() << ", "
                    << floor.normal.z() << ")"
                    << " | floor_d=" << floor.d << "m"
                    << " | stairs="
                    << (stairAnalysis.stairs_detected ? "yes" : "no")
                    << " type=" << stairAnalysis.stair_type
                    << " levels=" << stairAnalysis.step_levels
                    << " riser=" << stairAnalysis.riser_m << "m"
                    << " conf=" << stairAnalysis.confidence
                    << " | depression="
                    << (stairAnalysis.depression_detected ? "yes" : "no")
                    << " depth=" << stairAnalysis.depression_depth_m << "m"
                    << " | ms pc=" << timings.pointcloud_ms
                    << " cloud=" << timings.cloud_ms
                    << " normal=" << timings.normals_ms
                    << " floor=" << timings.floor_ms
                    << " comp=" << timings.components_ms
                    << " vis=" << timings.vis_ms
                    << "\n";

                const int maxPrint =
                    std::min<int>(5, planePatches.size());

                for(int i = 0; i < maxPrint; ++i) {
                    const auto &p =
                        planePatches[static_cast<std::size_t>(i)];

                    std::cout
                        << "  P" << i
                        << " C" << p.component_id
                        << (p.is_floor_reference ? " FLOOR" : "")
                        << " points=" << p.num_points
                        << " h=" << p.height_from_floor << "m"
                        << " rms=" << p.rms_error << "m"
                        << " size=(" << p.width << " x " << p.depth << ")"
                        << " roi=" << p.floor_roi_overlap
                        << " centroid=("
                        << p.centroid.x() << ", "
                        << p.centroid.y() << ", "
                        << p.centroid.z() << ")"
                        << " normal=("
                        << p.normal.x() << ", "
                        << p.normal.y() << ", "
                        << p.normal.z() << ")"
                        << " bbox=["
                        << p.min_u << "," << p.min_v
                        << " -> "
                        << p.max_u << "," << p.max_v
                        << "]\n";
                }
            }

            cv::Mat combined(h, w * 2, CV_8UC3);
            depthVis.copyTo(combined(cv::Rect(0, 0, w, h)));
            overlay.copyTo(combined(cv::Rect(w, 0, w, h)));

            cv::putText(
                combined,
                "DEPTH",
                cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(255, 255, 255),
                2,
                cv::LINE_AA
            );

            cv::putText(
                combined,
                "HORIZONTAL COMPONENTS",
                cv::Point(w + 10, 88),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(255, 255, 255),
                2,
                cv::LINE_AA
            );

            cv::imshow("Horizontal Components", combined);

            const int key = cv::waitKey(1) & 0xFF;
            if(key == 'q' || key == 'Q' || key == 27) {
                break;
            }

            if(key == 'f' || key == 'F') {
                if(floor.valid) {
                    floorLocked = !floorLocked;

                    std::cout
                        << "Floor plane (normal + offset) "
                        << (floorLocked ? "LOCKED" : "UNLOCKED")
                        << " n=("
                        << floor.normal.x() << ", "
                        << floor.normal.y() << ", "
                        << floor.normal.z() << ") d="
                        << floor.d << "m\n";
                }
                else {
                    std::cout
                        << "Cannot lock floor normal yet: floor is not valid.\n";
                }
            }

            if(key == 's' || key == 'S') {
                const std::string base = timestampBase();
                const std::string plyPath =
                    outputDir + "/" + base + "_organized_valid.ply";
                const std::string depthPath =
                    outputDir + "/" + base + "_depth_mm.png";
                const std::string overlayPath =
                    outputDir + "/" + base + "_components.png";

                const bool plyOk =
                    saveOrganizedCloudPly(*organized, plyPath);
                const bool depthOk =
                    cv::imwrite(
                        depthPath,
                        makeDepthMillimeterImage(depthFrame)
                    );
                const bool overlayOk =
                    cv::imwrite(overlayPath, overlay);

                std::cout
                    << "\n===== SAVED =====\n"
                    << "PLY: " << plyPath
                    << (plyOk ? " [OK]" : " [FAIL]") << "\n"
                    << "Depth: " << depthPath
                    << (depthOk ? " [OK]" : " [FAIL]") << "\n"
                    << "Horizontal overlay: " << overlayPath
                    << (overlayOk ? " [OK]" : " [FAIL]") << "\n\n";
            }
        }

        pipeline.stop();
        cv::destroyAllWindows();
    }
    catch(const std::exception &e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
