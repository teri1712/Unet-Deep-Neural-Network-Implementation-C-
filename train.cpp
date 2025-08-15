#include "header.h"
#include <chrono>
#include <fstream>
#include <iostream>

namespace train {

    void train_with_lr(float lr, const std::string& model_name) {
        std::shared_ptr<models::ForwardNet> net;
        if (model_name == "Unet") {
            net = std::make_shared<models::UNet>(64);
        } else {
            net = std::make_shared<models::EffUNet>();
        }

        auto data_loader =
            torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                std::move(data::CustomDataset("../dataset/final-dataset")
                              .map(torch::data::transforms::Stack<>())),
                torch::data::DataLoaderOptions().batch_size(4));

        auto val_loader =
            torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                data::CustomDataset(("../dataset/validate"))
                    .map(torch::data::transforms::Stack<>()),
                3);

        const std::string filename = model_name + "_lr_" + utils::to_string_float(lr);

        std::ofstream stat_iou("../stat/" + filename + ".csv");
        stat_iou << "Number of epoch,training-error,validate-error,Team B,Team A,GK "
                 << "B,GK A,Ground,Advertisement,Audience,Goal Bar,Ball,Referee,Coach"
                 << std::endl;

        torch::optim::Adam optimizer(net->parameters(),
                                     torch::optim::AdamOptions(lr));

        auto start = std::chrono::high_resolution_clock::now();
        int max_epoch = 50;

        std::cout << "Start training for lr = " << lr << std::endl;
        for (int epoch = 0; epoch < max_epoch; ++epoch) {
            std::pair<float, int> ce_loss = std::make_pair(0, 0);
            int cnt_batch = 0;
            for (auto& batch : *data_loader) {
                auto start_batch = std::chrono::high_resolution_clock::now();
                optimizer.zero_grad();
                torch::Tensor prediction = net->forward(batch.data);
                auto loss =
                    torch::nn::functional::cross_entropy(prediction, batch.target);
                loss.backward();
                ce_loss.first += loss.item<float>();
                ce_loss.second += batch.data.size(0);
                optimizer.step();

                auto end_batch = std::chrono::high_resolution_clock::now();
                std::cout << "Time of batch " + std::to_string(cnt_batch) << " : "
                          << std::chrono::duration_cast<std::chrono::seconds>(end_batch -
                                                                              start_batch)
                                 .count()
                          << " seconds" << std::endl;
                cnt_batch++;
            }

            float val_error = 0;
            for (auto& batch : *val_loader) {
                torch::Tensor pred_prob = net->forward(batch.data);
                auto loss = torch::nn::functional::cross_entropy(pred_prob, batch.target);
                val_error = loss.item<float>();
            }
            val_error /= 5;
            std::vector<float> mIoU_each_class =
                utils::evaluate_IoU(net, model_name + "_lr" + utils::to_string_float(lr) + "_epoch" +
                                          std::to_string(epoch));

            float error = ce_loss.first / ce_loss.second;
            std::cout << "End epoch " << epoch + 1 << " with error: " << error
                      << " , validation error: " << val_error << std::endl;

            stat_iou << epoch << "," << error << "," << val_error;
            for (int i = 0; i < 11; i++) {
                stat_iou << "," << mIoU_each_class[i];
            }
            stat_iou << std::endl;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        std::cout << "Time train with: " + std::to_string(max_epoch) + " :" << duration
                  << " seconds" << std::endl;
        std::cout << "End training" << std::endl;
        torch::save(net, "../models/" + filename + ".pt");

        stat_iou.close();
    }

}
