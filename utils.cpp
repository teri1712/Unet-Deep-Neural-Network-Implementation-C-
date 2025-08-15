#include "header.h"
#include <filesystem>
#include <iostream>

namespace utils {

    extern std::tuple<int, int, int> color_map[11];

    float calc_mIoU(const torch::Tensor& pred, const torch::Tensor& target,
                    int class_index) {
        auto pred_mask = (pred == class_index);
        auto label_mask = (target == class_index);

        auto inter = torch::sum((pred_mask & label_mask).to(torch::kFloat), {1, 2});
        auto uni = torch::sum((pred_mask | label_mask).to(torch::kFloat), {1, 2});

        auto iou = inter / uni;
        int batch_size = iou.size(0);
        return iou.sum().item<float>() / batch_size;
    }

    void create_pred_image(torch::Tensor pred, const std::string& path) {
        const int64_t* tensor_data = pred.data_ptr<int64_t>();

        cv::Mat segment_img(256, 480, CV_8UC3);
        for (int i = 0; i < 256; ++i) {
            for (int j = 0; j < 480; ++j) {
                std::tuple<int, int, int> color = color_map[tensor_data[i * 480 + j]];
                segment_img.at<cv::Vec3b>(i, j)[0] = std::get<2>(color);
                segment_img.at<cv::Vec3b>(i, j)[1] = std::get<1>(color);
                segment_img.at<cv::Vec3b>(i, j)[2] = std::get<0>(color);
            }
        }
        cv::imwrite(path + "__pred.jpg", segment_img);
    }

    std::vector<float> evaluate_IoU(std::shared_ptr<models::ForwardNet> net,
                               const std::string& milestone) {
        std::vector<float> mIoU(11);

        for (const auto& entry :
             std::filesystem::directory_iterator("../dataset/validate")) {
            if (!entry.is_directory()) continue;
            std::string path = entry.path().string();
            std::string filename = entry.path().filename().string();

            int class_index = 0;
            if (filename == "Team B") {
                class_index = 0;
            } else if (filename == "Team A") {
                class_index = 1;
            } else if (filename == "GK B") {
                class_index = 2;
            } else if (filename == "GK A") {
                class_index = 3;
            } else if (filename == "Ground") {
                class_index = 4;
            } else if (filename == "Advertisement") {
                class_index = 5;
            } else if (filename == "Audience") {
                class_index = 6;
            } else if (filename == "Goal Bar") {
                class_index = 7;
            } else if (filename == "Ball") {
                class_index = 8;
            } else if (filename == "Referee") {
                class_index = 9;
            } else {
                class_index = 10;
            }
            data::CustomDataset dataset(path);
            auto loader =
                torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                    dataset.map(torch::data::transforms::Stack<>()),
                    dataset.size().value());

            for (auto& batch : *loader) {
                torch::Tensor pred_prob = net->forward(batch.data);
                torch::Tensor pred = torch::argmax(pred_prob, 1).to(torch::kLong);
                float get = calc_mIoU(pred, batch.target, class_index);
                mIoU[class_index] = get;

                for (int i = 0; i < (int)dataset.size().value(); i++) {
                    auto pred_tensor_of_image = pred.select(0, i);
                    std::filesystem::path p(dataset.list_images[i]);

                    create_pred_image(
                        pred_tensor_of_image,
                        path + "/pred/" + p.filename().string() + milestone + ".jpg");
                }
            }
        }
        return mIoU;
    }

    std::string to_string_float(float value) {
        std::string s = std::to_string(value);
        if ((int)s.length() == 1) {
            return s;
        }
        while (s.back() == '0') {
            s.pop_back();
        }

        return s;
    }

}
