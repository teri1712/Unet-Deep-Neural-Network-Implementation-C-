#include "header.h"
#include <filesystem>

namespace data {

    extern int color_index[256][256][256];

    torch::Tensor read_data(const std::string& filename) {
        cv::Mat img = cv::imread(filename);
        torch::Tensor img_tensor =
            torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
        img_tensor = img_tensor.permute({2, 0, 1});
        return img_tensor.to(torch::kFloat).div_(255);
    }

    torch::Tensor read_label(const std::string& filename) {
        cv::Mat img = cv::imread(filename);
        std::vector<int8_t> temp(img.rows * img.cols);
        int sum_row = 0;
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                uchar b = img.at<cv::Vec3b>(i, j)[0];
                uchar g = img.at<cv::Vec3b>(i, j)[1];
                uchar r = img.at<cv::Vec3b>(i, j)[2];
                temp[sum_row + j] = color_index[r][g][b];
            }
            sum_row += img.cols;
        }
        torch::Tensor img_tensor =
            torch::from_blob(temp.data(), {img.rows, img.cols}, torch::kByte);
        return img_tensor.to(torch::kLong);
    }

    std::vector<torch::Tensor> process_images(const std::vector<std::string>& list_images) {
        std::vector<torch::Tensor> states;
        for (const std::string& filename : list_images) {
            torch::Tensor img = read_data(filename);
            states.push_back(img);
        }
        return states;
    }

    std::vector<torch::Tensor> process_labels(const std::vector<std::string>& list_labels) {
        std::vector<torch::Tensor> labels;
        for (const std::string& label_filename : list_labels) {
            torch::Tensor label = read_label(label_filename);
            labels.push_back(label);
        }
        return labels;
    }

    CustomDataset::CustomDataset(const std::string& dataset_location) {
        for (const auto& entry : std::filesystem::directory_iterator(dataset_location)) {
            if (entry.is_directory()) continue;
            std::string path = entry.path().string();
            std::string filename = entry.path().filename().string();
            if (filename.find("fuse") == std::string::npos &&
                filename.find("save") == std::string::npos) {
                list_images.push_back(path);
                list_labels.push_back(path + "___fuse.png");
            }
        }
        images = process_images(list_images);
        labels = process_labels(list_labels);
    }

    torch::data::Example<> CustomDataset::get(size_t index) {
        torch::Tensor sample_img = images.at(index);
        torch::Tensor sample_label = labels.at(index);
        return {sample_img.clone(), sample_label.clone()};
    }

    torch::optional<size_t> CustomDataset::size() const { return labels.size(); }

}
