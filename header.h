#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <tuple>

namespace data {

    torch::Tensor read_data(const std::string& filename);
    torch::Tensor read_label(const std::string& filename);
    std::vector<torch::Tensor> process_images(const std::vector<std::string>& list_images);
    std::vector<torch::Tensor> process_labels(const std::vector<std::string>& list_labels);

    struct CustomDataset : public torch::data::Dataset<CustomDataset> {
    public:
        std::vector<torch::Tensor> images, labels;
        std::vector<std::string> list_images, list_labels;

        CustomDataset(const std::string& dataset_location);
        torch::data::Example<> get(size_t index) override;
        torch::optional<size_t> size() const override;
    };

}

namespace models {

    struct ForwardNet : public torch::nn::Module {
    public:
        ForwardNet() = default;
        virtual torch::Tensor forward(torch::Tensor x) = 0;
    };

    struct ConvBlockImpl : public ForwardNet {
        ConvBlockImpl(int64_t in_channels, int64_t out_channels);
        torch::Tensor forward(torch::Tensor x) override;

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    };
    TORCH_MODULE(ConvBlock);

    struct UNet : public ForwardNet {
        UNet(int channels);
        torch::Tensor forward(torch::Tensor x) override;

        ConvBlock econv1{nullptr}, econv2{nullptr}, econv3{nullptr}, econv4{nullptr},
                  cconv{nullptr}, dconv4{nullptr}, dconv3{nullptr}, dconv2{nullptr},
                  dconv1{nullptr};

        torch::nn::ConvTranspose2d upconv1{nullptr}, upconv2{nullptr},
                                   upconv3{nullptr}, upconv4{nullptr};
        torch::nn::Conv2d final_conv{nullptr};
    };

    struct SEBlockImpl : torch::nn::Module {
    private:
        torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    public:
        SEBlockImpl(int in_channels);
        torch::Tensor forward(torch::Tensor x);
    };
    TORCH_MODULE(SEBlock);

    struct MBConvImpl : torch::nn::Module {
        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
        SEBlock se{nullptr};

        MBConvImpl(int in_channels, int out_channels, int expan, int stride, int kernel);
        torch::Tensor forward(torch::Tensor x);
    };
    TORCH_MODULE(MBConv);

    struct MBBlockImpl : public ForwardNet {
    protected:
        std::vector<MBConv> mbconvs;

    public:
        MBBlockImpl() = default;
        torch::Tensor forward(torch::Tensor x) override;
    };
    TORCH_MODULE(MBBlock);

    struct MBBlock1Impl : public MBBlockImpl { MBBlock1Impl(); };
    TORCH_MODULE(MBBlock1);

    struct MBBlock2Impl : public MBBlockImpl { MBBlock2Impl(); };
    TORCH_MODULE(MBBlock2);

    struct MBBlock3Impl : public MBBlockImpl { MBBlock3Impl(); };
    TORCH_MODULE(MBBlock3);

    struct MBBlock4Impl : public MBBlockImpl { MBBlock4Impl(); };
    TORCH_MODULE(MBBlock4);

    struct MBBlock5Impl : public MBBlockImpl { MBBlock5Impl(); };
    TORCH_MODULE(MBBlock5);

    struct MBBlock6Impl : public MBBlockImpl { MBBlock6Impl(); };
    TORCH_MODULE(MBBlock6);

    struct MBBlock7Impl : public MBBlockImpl { MBBlock7Impl(); };
    TORCH_MODULE(MBBlock7);

    struct EffUNet : public ForwardNet {
        EffUNet();
        torch::Tensor forward(torch::Tensor x);

        ConvBlock dconv5{nullptr}, dconv4{nullptr}, dconv3{nullptr}, dconv2{nullptr},
                  dconv1{nullptr};

        MBBlock1 mbblock1{nullptr};
        MBBlock2 mbblock2{nullptr};
        MBBlock3 mbblock3{nullptr};
        MBBlock4 mbblock4{nullptr};
        MBBlock5 mbblock5{nullptr};
        MBBlock6 mbblock6{nullptr};
        MBBlock7 mbblock7{nullptr};

        torch::nn::ConvTranspose2d upconv1{nullptr}, upconv2{nullptr},
                                   upconv3{nullptr}, upconv4{nullptr}, upconv5{nullptr};
        torch::nn::Conv2d first_conv{nullptr}, final_conv{nullptr};
    };

}

namespace utils {

    float calc_mIoU(const torch::Tensor& pred, const torch::Tensor& target, int class_index);
    void create_pred_image(torch::Tensor pred, const std::string& path);
    std::vector<float> evaluate_IoU(std::shared_ptr<models::ForwardNet> net, const std::string& milestone);
    std::string to_string_float(float value);

}

namespace train {

    void train_with_lr(float lr, const std::string& model_name);

}
