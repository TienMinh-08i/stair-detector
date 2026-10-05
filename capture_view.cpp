#include <libobsensor/ObSensor.hpp>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::string timestampBase() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::tm tm{};
    localtime_r(&t, &tm);

    std::ostringstream oss;
    oss << "capture_" << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

static cv::Mat makeDepthVisualization(const std::shared_ptr<ob::DepthFrame> &depthFrame,
                                      float minDepthMm = 300.0f,
                                      float maxDepthMm = 4000.0f) {
    const int w = depthFrame->width();
    const int h = depthFrame->height();

    const auto *raw = reinterpret_cast<const uint16_t *>(depthFrame->data());
    const float scale = depthFrame->getValueScale();

    cv::Mat gray(h, w, CV_8UC1, cv::Scalar(0));
    cv::Mat validMask(h, w, CV_8UC1, cv::Scalar(0));

    for(int y = 0; y < h; ++y) {
        auto *g = gray.ptr<uint8_t>(y);
        auto *m = validMask.ptr<uint8_t>(y);

        for(int x = 0; x < w; ++x) {
            const uint16_t dRaw = raw[y * w + x];

            if(dRaw == 0) {
                g[x] = 0;
                m[x] = 0;
                continue;
            }

            const float dMm = static_cast<float>(dRaw) * scale;

            if(dMm < minDepthMm || dMm > maxDepthMm) {
                g[x] = 0;
                m[x] = 0;
                continue;
            }

            // Nearer points -> larger intensity.
            const float t = (dMm - minDepthMm) / (maxDepthMm - minDepthMm);
            const float inv = 1.0f - std::clamp(t, 0.0f, 1.0f);

            g[x] = static_cast<uint8_t>(inv * 255.0f);
            m[x] = 255;
        }
    }

    cv::Mat colored;
    cv::applyColorMap(gray, colored, cv::COLORMAP_TURBO);

    // Invalid depth -> black.
    cv::Mat invalidMask;
    cv::bitwise_not(validMask, invalidMask);
    colored.setTo(cv::Scalar(0, 0, 0), invalidMask);

    return colored;
}

static cv::Mat makeDepthMillimeterImage(const std::shared_ptr<ob::DepthFrame> &depthFrame) {
    const int w = depthFrame->width();
    const int h = depthFrame->height();

    const auto *raw = reinterpret_cast<const uint16_t *>(depthFrame->data());
    const float scale = depthFrame->getValueScale();

    cv::Mat depthMm(h, w, CV_16UC1, cv::Scalar(0));

    for(int y = 0; y < h; ++y) {
        auto *dst = depthMm.ptr<uint16_t>(y);

        for(int x = 0; x < w; ++x) {
            const uint16_t dRaw = raw[y * w + x];

            if(dRaw == 0) {
                dst[x] = 0;
                continue;
            }

            const float dMm = static_cast<float>(dRaw) * scale;

            if(dMm <= 0.0f || dMm > 65535.0f) {
                dst[x] = 0;
            }
            else {
                dst[x] = static_cast<uint16_t>(std::lround(dMm));
            }
        }
    }

    return depthMm;
}

static bool savePointCloudPly(const std::shared_ptr<ob::Frame> &pointFrame,
                              const std::string &path) {
    if(!pointFrame || !pointFrame->data() || pointFrame->dataSize() == 0) {
        return false;
    }

    const int count = static_cast<int>(pointFrame->dataSize() / sizeof(OBPoint));
    const auto *points = reinterpret_cast<const OBPoint *>(pointFrame->data());

    std::vector<OBPoint> valid;
    valid.reserve(count);

    for(int i = 0; i < count; ++i) {
        const auto &p = points[i];

        if(!std::isfinite(p.x) ||
           !std::isfinite(p.y) ||
           !std::isfinite(p.z)) {
            continue;
        }

        if(p.z <= 0.0f) {
            continue;
        }

        valid.push_back(p);
    }

    FILE *fp = std::fopen(path.c_str(), "wb");
    if(!fp) {
        return false;
    }

    std::fprintf(fp, "ply\n");
    std::fprintf(fp, "format ascii 1.0\n");
    std::fprintf(fp, "element vertex %zu\n", valid.size());
    std::fprintf(fp, "property float x\n");
    std::fprintf(fp, "property float y\n");
    std::fprintf(fp, "property float z\n");
    std::fprintf(fp, "end_header\n");

    // Orbbec PointCloudFilter + setPositionDataScaled(depth scale)
    // yields point coordinates in millimeters, matching the official sample.
    for(const auto &p : valid) {
        std::fprintf(fp, "%.6f %.6f %.6f\n", p.x, p.y, p.z);
    }

    std::fclose(fp);
    return true;
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
            std::cerr << "No depth stream profile found.\n";
            return 1;
        }

        auto depthProfile =
            depthProfiles->getProfile(OB_PROFILE_DEFAULT);

        config->enableStream(depthProfile);

        pipeline.start(config);

        ob::PointCloudFilter pointCloudFilter;
        pointCloudFilter.setCameraParam(pipeline.getCameraParam());
        pointCloudFilter.setCreatePointFormat(OB_FORMAT_POINT);

        cv::namedWindow("Orbbec Depth Preview", cv::WINDOW_NORMAL);
        cv::resizeWindow("Orbbec Depth Preview", 960, 600);

        std::cout
            << "\n===== ORBBEC OPENCV CAPTURE =====\n"
            << "Look at the depth preview and position the camera.\n\n"
            << "Keys inside OpenCV window:\n"
            << "  S : save current point cloud + depth image\n"
            << "  Q : quit\n"
            << "  ESC : quit\n\n";

        bool printedInfo = false;

        while(true) {
            auto frameset = pipeline.waitForFrames(100);

            if(!frameset || !frameset->depthFrame()) {
                continue;
            }

            auto depthFrame = frameset->depthFrame();
            const float depthScale = depthFrame->getValueScale();

            pointCloudFilter.setPositionDataScaled(depthScale);

            auto pointFrame = pointCloudFilter.process(frameset);

            cv::Mat depthVis = makeDepthVisualization(depthFrame);
            cv::Mat depthMm = makeDepthMillimeterImage(depthFrame);

            const int rawPointCount =
                pointFrame
                    ? static_cast<int>(pointFrame->dataSize() / sizeof(OBPoint))
                    : 0;

            int validPointCount = 0;

            if(pointFrame && pointFrame->data()) {
                const auto *points =
                    reinterpret_cast<const OBPoint *>(pointFrame->data());

                for(int i = 0; i < rawPointCount; ++i) {
                    if(std::isfinite(points[i].x) &&
                       std::isfinite(points[i].y) &&
                       std::isfinite(points[i].z) &&
                       points[i].z > 0.0f) {
                        ++validPointCount;
                    }
                }
            }

            cv::putText(
                depthVis,
                "Valid 3D points: " + std::to_string(validPointCount),
                cv::Point(15, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(255, 255, 255),
                2
            );

            cv::putText(
                depthVis,
                "S: save   Q/ESC: quit",
                cv::Point(15, 60),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(255, 255, 255),
                2
            );

            cv::imshow("Orbbec Depth Preview", depthVis);

            if(!printedInfo) {
                std::cout
                    << "Depth resolution: "
                    << depthFrame->width()
                    << " x "
                    << depthFrame->height()
                    << "\n";

                std::cout
                    << "Depth value scale: "
                    << depthScale
                    << "\n";

                std::cout
                    << "PointCloud raw points: "
                    << rawPointCount
                    << "\n";

                std::cout
                    << "Valid point count: "
                    << validPointCount
                    << "\n\n";

                printedInfo = true;
            }

            const int key = cv::waitKey(1) & 0xFF;

            if(key == 'q' || key == 'Q' || key == 27) {
                break;
            }

            if(key == 's' || key == 'S') {
                const std::string base = timestampBase();

                const std::string plyPath =
                    outputDir + "/" + base + ".ply";

                const std::string depthPath =
                    outputDir + "/" + base + "_depth_mm.png";

                const std::string visPath =
                    outputDir + "/" + base + "_depth_vis.png";

                bool plyOk =
                    savePointCloudPly(pointFrame, plyPath);

                bool depthOk =
                    cv::imwrite(depthPath, depthMm);

                bool visOk =
                    cv::imwrite(visPath, depthVis);

                std::cout
                    << "\n===== SAVED =====\n"
                    << "PLY:       " << plyPath
                    << (plyOk ? "  [OK]" : "  [FAILED]")
                    << "\n"
                    << "Depth PNG: " << depthPath
                    << (depthOk ? "  [OK]" : "  [FAILED]")
                    << "\n"
                    << "Preview:   " << visPath
                    << (visOk ? "  [OK]" : "  [FAILED]")
                    << "\n"
                    << "Valid 3D points: "
                    << validPointCount
                    << "\n\n";
            }
        }

        pipeline.stop();
        cv::destroyAllWindows();
    }
    catch(const std::exception &e) {
        std::cerr
            << "Fatal error: "
            << e.what()
            << "\n";

        return 1;
    }

    return 0;
}
