#include "header.h"

namespace models {

    ConvBlockImpl::ConvBlockImpl(int64_t in_channels, int64_t out_channels) {
        conv1 = register_module(
            "conv1",
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));
        conv2 = register_module(
            "conv2",
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
    }

    torch::Tensor ConvBlockImpl::forward(torch::Tensor x) {
        x = conv1->forward(x);
        torch::relu_(x);
        x = bn1->forward(x);

        x = conv2->forward(x);
        torch::relu_(x);
        x = bn2->forward(x);
        return x;
    }

    UNet::UNet(int channels) {
        econv1 = register_module("econv1", ConvBlock(3, channels));
        econv2 = register_module("econv2", ConvBlock(channels, 2 * channels));
        econv3 = register_module("econv3", ConvBlock(2 * channels, 4 * channels));
        econv4 = register_module("econv4", ConvBlock(4 * channels, 8 * channels));
        cconv = register_module("cconv", ConvBlock(8 * channels, 16 * channels));
        dconv4 = register_module(
            "dconv4", ConvBlock(8 * channels + 16 * channels, 8 * channels));
        dconv3 = register_module(
            "dconv3", ConvBlock(4 * channels + 8 * channels, 4 * channels));
        dconv2 = register_module(
            "dconv2", ConvBlock(2 * channels + 4 * channels, 2 * channels));
        dconv1 =
            register_module("dconv1", ConvBlock(channels + 2 * channels, channels));

        upconv4 = register_module(
            "upconv4", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(
                                                        16 * channels, 16 * channels, 2)
                                                        .stride(2)));

        upconv3 = register_module(
            "upconv3", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(
                                                        8 * channels, 8 * channels, 2)
                                                        .stride(2)));
        upconv2 = register_module(
            "upconv2", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(
                                                        4 * channels, 4 * channels, 2)
                                                        .stride(2)));
        upconv1 = register_module(
            "upconv1", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(
                                                        2 * channels, 2 * channels, 2)
                                                        .stride(2)));
        final_conv = register_module(
            "final_conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, 11, 3).padding(1)));
    }

    torch::Tensor UNet::forward(torch::Tensor x) {
        auto enc1 = econv1->forward(x);
        auto enc2 = econv2->forward(torch::max_pool2d(enc1, 2));
        auto enc3 = econv3->forward(torch::max_pool2d(enc2, 2));
        auto enc4 = econv4->forward(torch::max_pool2d(enc3, 2));

        auto center = cconv->forward(torch::max_pool2d(enc4, 2));

        auto dec4 = dconv4->forward(torch::cat({enc4, upconv4->forward(center)}, 1));
        auto dec3 = dconv3->forward(torch::cat({enc3, upconv3->forward(dec4)}, 1));
        auto dec2 = dconv2->forward(torch::cat({enc2, upconv2->forward(dec3)}, 1));
        auto dec1 = dconv1->forward(torch::cat({enc1, upconv1->forward(dec2)}, 1));

        return final_conv->forward(dec1);
    }

    SEBlockImpl::SEBlockImpl(int in_channels) {
        fc1 = register_module("fc1", torch::nn::Linear(in_channels, in_channels / 4));
        fc2 = register_module("fc2", torch::nn::Linear(in_channels / 4, in_channels));
    }

    torch::Tensor SEBlockImpl::forward(torch::Tensor x) {
        torch::Tensor z = torch::adaptive_avg_pool2d(x, {1, 1});
        z = z.view({-1, x.size(1)});
        z = fc1->forward(z);
        torch::relu_(z);
        z = fc2->forward(z);
        torch::sigmoid_(z);
        z = z.view({-1, x.size(1), 1, 1});
        return z;
    }

    MBConvImpl::MBConvImpl(int in_channels, int out_channels, int expan, int stride,
                           int kernel) {
        int echannels = in_channels * expan;
        int padding = (kernel - 1) / 2;
        conv1 = register_module(
            "conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, echannels, 1)
                                .stride(1)
                                .padding(0)
                                .bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(echannels));
        conv2 = register_module(
            "conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(echannels, echannels, kernel)
                                .stride(stride)
                                .padding(padding)
                                .bias(false)
                                .groups(echannels)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(echannels));
        conv3 = register_module(
            "conv3",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(echannels, out_channels, 1)
                                .stride(1)
                                .padding(0)
                                .bias(false)));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(out_channels));
        se = register_module("se", SEBlock(echannels));
    }

    torch::Tensor MBConvImpl::forward(torch::Tensor x) {
        auto pre = x;
        x = conv1->forward(x);
        x = bn1->forward(x);
        torch::relu_(x);

        x = conv2->forward(x);
        x = bn2->forward(x);
        torch::relu_(x);

        auto z = se->forward(x);
        x = z * x;

        x = conv3->forward(x);
        x = bn3->forward(x);

        if (pre.sizes() == x.sizes()) {
            x = x + pre;
        }
        return x;
    }

    torch::Tensor MBBlockImpl::forward(torch::Tensor x) {
        for (MBConv conv : mbconvs) {
            x = conv->forward(x);
        }
        return x;
    }

    MBBlock1Impl::MBBlock1Impl() {
        mbconvs.push_back(register_module("mbconv1", MBConv(64, 32, 1, 1, 3)));
        mbconvs.push_back(register_module("mbconv2", MBConv(32, 32, 1, 1, 3)));
        mbconvs.push_back(register_module("mbconv3", MBConv(32, 32, 1, 1, 3)));
        mbconvs.push_back(register_module("mbconv4", MBConv(32, 32, 1, 1, 3)));
    }

    MBBlock2Impl::MBBlock2Impl() {
        mbconvs.push_back(register_module("mbconv1", MBConv(32, 48, 6, 2, 3)));
        mbconvs.push_back(register_module("mbconv2", MBConv(48, 48, 6, 1, 3)));
        mbconvs.push_back(register_module("mbconv3", MBConv(48, 48, 6, 1, 3)));
        mbconvs.push_back(register_module("mbconv4", MBConv(48, 48, 6, 1, 3)));
        mbconvs.push_back(register_module("mbconv5", MBConv(48, 48, 6, 1, 3)));
        mbconvs.push_back(register_module("mbconv6", MBConv(48, 48, 6, 1, 3)));
        mbconvs.push_back(register_module("mbconv7", MBConv(48, 48, 6, 1, 3)));
    }

    MBBlock3Impl::MBBlock3Impl() {
        mbconvs.push_back(register_module("mbconv1", MBConv(48, 80, 6, 2, 5)));
        mbconvs.push_back(register_module("mbconv2", MBConv(80, 80, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv3", MBConv(80, 80, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv4", MBConv(80, 80, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv5", MBConv(80, 80, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv6", MBConv(80, 80, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv7", MBConv(80, 80, 6, 1, 5)));
    }

    MBBlock4Impl::MBBlock4Impl() {
        mbconvs.push_back(register_module("mbconv1", MBConv(80, 80, 6, 2, 3)));
        mbconvs.push_back(register_module("mbconv2", MBConv(80, 80, 6, 1, 3)));
        mbconvs.push_back(register_module("mbconv3", MBConv(80, 80, 6, 1, 3)));
        mbconvs.push_back(register_module("mbconv4", MBConv(80, 80, 6, 1, 3)));
        mbconvs.push_back(register_module("mbconv5", MBConv(80, 80, 6, 1, 3)));
        mbconvs.push_back(register_module("mbconv6", MBConv(80, 80, 6, 1, 3)));
        mbconvs.push_back(register_module("mbconv7", MBConv(80, 80, 6, 1, 3)));
        mbconvs.push_back(register_module("mbconv8", MBConv(80, 80, 6, 1, 3)));
        mbconvs.push_back(register_module("mbconv9", MBConv(80, 80, 6, 1, 3)));
        mbconvs.push_back(register_module("mbconv10", MBConv(80, 80, 6, 1, 3)));
    }

    MBBlock5Impl::MBBlock5Impl() {
        mbconvs.push_back(register_module("mbconv1", MBConv(80, 224, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv2", MBConv(224, 224, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv3", MBConv(224, 224, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv4", MBConv(224, 224, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv5", MBConv(224, 224, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv6", MBConv(224, 224, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv7", MBConv(224, 224, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv8", MBConv(224, 224, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv9", MBConv(224, 224, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv10", MBConv(224, 224, 6, 1, 5)));
    }

    MBBlock6Impl::MBBlock6Impl() {
        mbconvs.push_back(register_module("mbconv1", MBConv(224, 384, 6, 2, 5)));
        mbconvs.push_back(register_module("mbconv2", MBConv(384, 384, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv3", MBConv(384, 384, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv4", MBConv(384, 384, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv5", MBConv(384, 384, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv6", MBConv(384, 384, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv7", MBConv(384, 384, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv8", MBConv(384, 384, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv9", MBConv(384, 384, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv10", MBConv(384, 384, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv11", MBConv(384, 384, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv12", MBConv(384, 384, 6, 1, 5)));
        mbconvs.push_back(register_module("mbconv13", MBConv(384, 384, 6, 1, 5)));
    }

    MBBlock7Impl::MBBlock7Impl() {
        mbconvs.push_back(register_module("mbconv1", MBConv(384, 640, 6, 1, 3)));
        mbconvs.push_back(register_module("mbconv2", MBConv(640, 640, 6, 1, 3)));
        mbconvs.push_back(register_module("mbconv3", MBConv(640, 640, 6, 1, 3)));
        mbconvs.push_back(register_module("mbconv4", MBConv(640, 640, 6, 1, 3)));
    }

    EffUNet::EffUNet() {
        mbblock1 = register_module("mbblock1", MBBlock1());
        mbblock2 = register_module("mbblock2", MBBlock2());
        mbblock3 = register_module("mbblock3", MBBlock3());
        mbblock4 = register_module("mbblock4", MBBlock4());
        mbblock5 = register_module("mbblock5", MBBlock5());
        mbblock6 = register_module("mbblock6", MBBlock6());
        mbblock7 = register_module("mbblock7", MBBlock7());

        dconv5 = register_module("dconv5", ConvBlock(640, 512));
        dconv4 = register_module("dconv4", ConvBlock(512 + 224, 256));
        dconv3 = register_module("dconv3", ConvBlock(256 + 80, 128));
        dconv2 = register_module("dconv2", ConvBlock(128 + 48, 64));
        dconv1 = register_module("dconv1", ConvBlock(64 + 32, 16));
        upconv5 = register_module(
            "upconv5", torch::nn::ConvTranspose2d(
                           torch::nn::ConvTranspose2dOptions(512, 512, 2).stride(2)));
        upconv4 = register_module(
            "upconv4", torch::nn::ConvTranspose2d(
                           torch::nn::ConvTranspose2dOptions(256, 256, 2).stride(2)));

        upconv3 = register_module(
            "upconv3", torch::nn::ConvTranspose2d(
                           torch::nn::ConvTranspose2dOptions(128, 128, 2).stride(2)));
        upconv2 = register_module(
            "upconv2", torch::nn::ConvTranspose2d(
                           torch::nn::ConvTranspose2dOptions(64, 64, 2).stride(2)));
        upconv1 = register_module(
            "upconv1", torch::nn::ConvTranspose2d(
                           torch::nn::ConvTranspose2dOptions(16, 16, 2).stride(2)));
        first_conv = register_module(
            "first_conv",
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(3, 64, 3).stride(2).padding(1)));
        final_conv = register_module(
            "final_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 11, 1)));
    }

    torch::Tensor EffUNet::forward(torch::Tensor x) {
        x = first_conv(x);

        auto block1 = mbblock1->forward(x);
        auto block2 = mbblock2->forward(block1);
        auto block3 = mbblock3->forward(block2);
        auto block4 = mbblock4->forward(block3);
        auto block5 = mbblock5->forward(block4);
        auto block6 = mbblock6->forward(block5);
        auto block7 = mbblock7->forward(block6);

        auto dec5 = dconv5->forward(block7);
        auto dec4 = dconv4->forward(torch::cat({block5, upconv5->forward(dec5)}, 1));
        auto dec3 = dconv3->forward(torch::cat({block3, upconv4->forward(dec4)}, 1));
        auto dec2 = dconv2->forward(torch::cat({block2, upconv3->forward(dec3)}, 1));
        auto dec1 = dconv1->forward(torch::cat({block1, upconv2->forward(dec2)}, 1));

        return final_conv->forward(upconv1(dec1));
    }

}
