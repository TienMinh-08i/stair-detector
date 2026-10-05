#include <opencv2/opencv.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using Cloud = pcl::PointCloud<pcl::PointXYZ>;

struct PlaneModel {
    Eigen::Vector3f normal;
    float d;
    int downsample_inliers;
    Eigen::Vector3f centroid;
};

static float medianAbsZ(const Cloud::Ptr &cloud) {
    std::vector<float> values;
    values.reserve(cloud->size());

    for(const auto &p : cloud->points) {
        if(std::isfinite(p.z) && std::abs(p.z) > 1e-9f) {
            values.push_back(std::abs(p.z));
        }
    }

    if(values.empty()) {
        return 0.0f;
    }

    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    return values[mid];
}

static void scaleCloud(Cloud::Ptr &cloud, float scale) {
    for(auto &p : cloud->points) {
        p.x *= scale;
        p.y *= scale;
        p.z *= scale;
    }
}

static cv::Mat makeDepthPreview(const cv::Mat &depthMm,
                                int minDepthMm = 300,
                                int maxDepthMm = 4000) {
    cv::Mat gray(depthMm.size(), CV_8UC1, cv::Scalar(0));

    for(int y = 0; y < depthMm.rows; ++y) {
        const auto *src = depthMm.ptr<uint16_t>(y);
        auto *dst = gray.ptr<uint8_t>(y);

        for(int x = 0; x < depthMm.cols; ++x) {
            const uint16_t d = src[x];

            if(d == 0 || d < minDepthMm || d > maxDepthMm) {
                dst[x] = 0;
                continue;
            }

            const float t =
                static_cast<float>(d - minDepthMm) /
                static_cast<float>(maxDepthMm - minDepthMm);

            dst[x] = static_cast<uint8_t>(
                (1.0f - std::clamp(t, 0.0f, 1.0f)) * 255.0f
            );
        }
    }

    cv::Mat color;
    cv::applyColorMap(gray, color, cv::COLORMAP_TURBO);

    cv::Mat invalidMask = (depthMm == 0);
    color.setTo(cv::Scalar(0, 0, 0), invalidMask);

    return color;
}

int main(int argc, char **argv) {
    if(argc < 3) {
        std::cout
            << "Usage:\n"
            << "  ./detect_planes_opencv cloud.ply depth_mm.png\n\n"
            << "Example:\n"
            << "  ./detect_planes_opencv "
            << "../captures/capture_20260922_165434.ply "
            << "../captures/capture_20260922_165434_depth_mm.png\n";
        return 0;
    }

    const std::string plyPath = argv[1];
    const std::string depthPath = argv[2];

    // Initial values for your current Orbbec data.
    const int maxPlanes = 10;
    const float voxelSize = 0.015f;          // 1.5 cm
    const float ransacThreshold = 0.015f;    // 1.5 cm
    const float classifyThreshold = 0.018f;  // classify full-res points
    const int minPlanePoints = 250;

    Cloud::Ptr full(new Cloud);

    if(pcl::io::loadPLYFile<pcl::PointXYZ>(plyPath, *full) != 0) {
        std::cerr << "Could not load PLY: " << plyPath << "\n";
        return 1;
    }

    std::vector<int> finiteIndices;
    Cloud::Ptr finite(new Cloud);
    pcl::removeNaNFromPointCloud(*full, *finite, finiteIndices);

    // Your capture program only writes valid points, so this should normally
    // remain the same size.
    full = finite;

    if(full->empty()) {
        std::cerr << "PLY contains no valid points.\n";
        return 1;
    }

    cv::Mat depthMm = cv::imread(depthPath, cv::IMREAD_UNCHANGED);

    if(depthMm.empty()) {
        std::cerr << "Could not load depth image: " << depthPath << "\n";
        return 1;
    }

    if(depthMm.type() != CV_16UC1) {
        std::cerr
            << "Depth image must be 16-bit single-channel PNG (CV_16UC1).\n";
        return 1;
    }

    const int nonZeroDepth = cv::countNonZero(depthMm);

    std::cout << "\n===== INPUT =====\n";
    std::cout << "PLY points:        " << full->size() << "\n";
    std::cout << "Depth resolution:  "
              << depthMm.cols << " x " << depthMm.rows << "\n";
    std::cout << "Non-zero depth px: " << nonZeroDepth << "\n";

    if(static_cast<std::size_t>(nonZeroDepth) != full->size()) {
        std::cerr
            << "\nERROR: PLY point count != non-zero depth pixel count.\n"
            << "These two files must come from the SAME capture.\n";
        return 1;
    }

    const float medianZ = medianAbsZ(full);

    if(medianZ > 20.0f) {
        std::cout
            << "Median |Z| = " << medianZ
            << " -> converting PLY mm to meters\n";
        scaleCloud(full, 0.001f);
    }
    else {
        std::cout
            << "Median |Z| = " << medianZ
            << " -> treating PLY as meters\n";
    }

    // Downsample only for finding plane models.
    // The original full-resolution cloud is preserved for pixel overlay.
    Cloud::Ptr down(new Cloud);

    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(full);
    voxel.setLeafSize(voxelSize, voxelSize, voxelSize);
    voxel.filter(*down);

    std::cout << "After voxel:       " << down->size() << "\n";

    Cloud::Ptr remaining(new Cloud(*down));
    std::vector<PlaneModel> planes;

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1500);
    seg.setDistanceThreshold(ransacThreshold);

    pcl::ExtractIndices<pcl::PointXYZ> extract;

    std::cout << "\n===== RANSAC PLANES =====\n";

    for(int planeId = 0; planeId < maxPlanes; ++planeId) {
        if(static_cast<int>(remaining->size()) < minPlanePoints) {
            break;
        }

        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);

        seg.setInputCloud(remaining);
        seg.segment(*inliers, *coeff);

        if(static_cast<int>(inliers->indices.size()) < minPlanePoints) {
            break;
        }

        Cloud::Ptr planeCloud(new Cloud);
        extract.setInputCloud(remaining);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*planeCloud);

        Cloud::Ptr rest(new Cloud);
        extract.setNegative(true);
        extract.filter(*rest);

        if(coeff->values.size() < 4) {
            break;
        }

        Eigen::Vector3f n(
            coeff->values[0],
            coeff->values[1],
            coeff->values[2]
        );

        const float norm = n.norm();

        if(norm < 1e-8f) {
            remaining = rest;
            continue;
        }

        n /= norm;
        const float d = coeff->values[3] / norm;

        Eigen::Vector4f centroid4;
        pcl::compute3DCentroid(*planeCloud, centroid4);

        PlaneModel model;
        model.normal = n;
        model.d = d;
        model.downsample_inliers =
            static_cast<int>(planeCloud->size());
        model.centroid =
            Eigen::Vector3f(
                centroid4[0],
                centroid4[1],
                centroid4[2]
            );

        planes.push_back(model);

        std::cout
            << "Plane " << planeId
            << " | downsample points=" << model.downsample_inliers
            << " | normal=("
            << n.x() << ", "
            << n.y() << ", "
            << n.z() << ")"
            << " | d=" << d
            << " | centroid=("
            << model.centroid.x() << ", "
            << model.centroid.y() << ", "
            << model.centroid.z() << ") m\n";

        remaining = rest;
    }

    if(planes.empty()) {
        std::cerr << "No plane found.\n";
        return 1;
    }

    // Classify each ORIGINAL point to the nearest detected plane.
    std::vector<int> pointLabels(full->size(), -1);
    std::vector<int> fullPlaneCounts(planes.size(), 0);

    for(std::size_t i = 0; i < full->size(); ++i) {
        const auto &p = full->points[i];
        const Eigen::Vector3f v(p.x, p.y, p.z);

        float bestDistance =
            std::numeric_limits<float>::max();

        int bestPlane = -1;

        for(std::size_t k = 0; k < planes.size(); ++k) {
            const float dist =
                std::abs(planes[k].normal.dot(v) + planes[k].d);

            if(dist < bestDistance) {
                bestDistance = dist;
                bestPlane = static_cast<int>(k);
            }
        }

        if(bestDistance <= classifyThreshold) {
            pointLabels[i] = bestPlane;
            fullPlaneCounts[bestPlane]++;
        }
    }

    std::cout << "\n===== FULL-RES CLASSIFICATION =====\n";

    for(std::size_t i = 0; i < planes.size(); ++i) {
        std::cout
            << "Plane " << i
            << ": " << fullPlaneCounts[i]
            << " pixels/points\n";
    }

    // OpenCV BGR colors.
    const std::array<cv::Vec3b, 10> colors = {{
        cv::Vec3b(60,  60, 255),   // red
        cv::Vec3b(80, 220,  80),   // green
        cv::Vec3b(255,120,  70),   // blue-ish
        cv::Vec3b(50, 210, 255),   // yellow
        cv::Vec3b(230, 80, 190),   // purple
        cv::Vec3b(220,220,  50),   // cyan
        cv::Vec3b(180,100, 245),
        cv::Vec3b(70, 190, 190),
        cv::Vec3b(245,190,  90),
        cv::Vec3b(80, 150, 245)
    }};

    cv::Mat depthPreview = makeDepthPreview(depthMm);

    // Dark background so planes stand out.
    cv::Mat overlay;
    depthPreview.convertTo(overlay, CV_8UC3, 0.28);

    std::size_t pointIndex = 0;

    for(int y = 0; y < depthMm.rows; ++y) {
        const auto *depthRow = depthMm.ptr<uint16_t>(y);
        auto *overlayRow = overlay.ptr<cv::Vec3b>(y);

        for(int x = 0; x < depthMm.cols; ++x) {
            if(depthRow[x] == 0) {
                overlayRow[x] = cv::Vec3b(0, 0, 0);
                continue;
            }

            if(pointIndex >= pointLabels.size()) {
                std::cerr << "Internal point/pixel mapping error.\n";
                return 1;
            }

            const int label = pointLabels[pointIndex];

            if(label >= 0) {
                const cv::Vec3b c =
                    colors[static_cast<std::size_t>(label) % colors.size()];

                const cv::Vec3b b = overlayRow[x];

                overlayRow[x] = cv::Vec3b(
                    static_cast<uint8_t>(0.15f * b[0] + 0.85f * c[0]),
                    static_cast<uint8_t>(0.15f * b[1] + 0.85f * c[1]),
                    static_cast<uint8_t>(0.15f * b[2] + 0.85f * c[2])
                );
            }

            ++pointIndex;
        }
    }

    // Add compact legend.
    int legendY = 25;

    for(std::size_t i = 0; i < planes.size(); ++i) {
        const cv::Vec3b c = colors[i % colors.size()];

        cv::rectangle(
            overlay,
            cv::Rect(10, legendY - 13, 18, 18),
            cv::Scalar(c[0], c[1], c[2]),
            cv::FILLED
        );

        cv::putText(
            overlay,
            "P" + std::to_string(i)
                + ": " + std::to_string(fullPlaneCounts[i]),
            cv::Point(35, legendY),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(255, 255, 255),
            1,
            cv::LINE_AA
        );

        legendY += 22;
    }

    const std::string outputPath = "planes_overlay.png";

    cv::imwrite(outputPath, overlay);

    std::cout
        << "\n[SAVED] " << outputPath << "\n"
        << "Bright colors = RANSAC planes\n"
        << "Dark depth pixels = not assigned to a detected plane\n\n"
        << "Press Q or ESC in an OpenCV window to quit.\n";

    cv::namedWindow("Depth", cv::WINDOW_NORMAL);
    cv::namedWindow("RANSAC Plane Overlay", cv::WINDOW_NORMAL);

    cv::resizeWindow("Depth", 960, 600);
    cv::resizeWindow("RANSAC Plane Overlay", 960, 600);

    cv::imshow("Depth", depthPreview);
    cv::imshow("RANSAC Plane Overlay", overlay);

    while(true) {
        const int key = cv::waitKey(30) & 0xFF;

        if(key == 'q' || key == 'Q' || key == 27) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
