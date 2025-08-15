#include "header.h"

int color_index[256][256][256];
tuple<int, int, int> color_map[11];

/* Tham khảo:
 * https://krshrimali.github.io/posts/2019/07/custom-data-loading-using-pytorch-c-api/
 */
/* Đọc dư liệu theo đường dẫn */
torch::Tensor read_data(const string& filename) {
  /* Đọc ảnh */
  cv::Mat img = cv::imread(filename);
  /*Tạo một tensor dữ liệu từ ảnh đọc được, tensor có kích thước như ảnh là
   * (h,w,c)
   */
  torch::Tensor img_tensor =
      torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);

  /* Hoán vị các chiều lại thành (c,h,w) để phù hợp form, tính toán tích chập
   * trên mạng */
  img_tensor = img_tensor.permute({2, 0, 1});

  /*Chuẩn hóa dữ liệu, train model hiệu quả hơn*/
  return img_tensor.to(torch::kFloat).div_(255);
}
/* Đọc nhãn theo đường dẫn */
torch::Tensor read_label(const string& filename) {
  cv::Mat img = cv::imread(filename);
  /* Gọi một mạng temp để tính toán trước sau đó khởi tạo tensor với mảng này,
   * vì cập nhật giá trị theo chỉ mục tensor rất chậm*/
  vector<int8_t> temp(img.rows * img.cols);
  /*Tương ứng mỗi điểm dữ liệu ảnh, lấy số thứ tự của lớp tương ứng với màu*/
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
  /* Tạo tensor nhãn */
  torch::Tensor img_tensor =
      torch::from_blob(temp.data(), {img.rows, img.cols}, torch::kByte);
  /*Đổi thành kLong theo format yêu cầu của hàm cross-entropy loss của
   * libtorch*/
  return img_tensor.to(torch::kLong);
}

/*Đọc toàn bộ ảnh lên*/
vector<torch::Tensor> process_images(const vector<string>& list_images) {
  vector<torch::Tensor> states;
  for (const string& filename : list_images) {
    torch::Tensor img = read_data(filename);
    states.push_back(img);
  }
  return states;
}
/*Đọc toàn bộ nhãn lên*/
vector<torch::Tensor> process_labels(const vector<string>& list_labels) {
  vector<torch::Tensor> labels;
  for (const string& label_filename : list_labels) {
    torch::Tensor label = read_label(label_filename);
    labels.push_back(label);
  }
  return labels;
}

CustomDataset::CustomDataset(const string& dataset_location) {
  for (const auto& entry : filesystem::directory_iterator(dataset_location)) {
    if (entry.is_directory()) continue;
    string path = entry.path().string();
    string filename = entry.path().filename().string();
    /*ảnh nào có "___fuse.png" là hậu tố là ảnh nhãn, ảnh "save" không dùng
    ảnh không có cả 2 là ảnh dữ liệu
    */
    if (filename.find("fuse") == std::string::npos &&
        filename.find("save") == std::string::npos) {
      list_images.push_back(path);
      list_labels.push_back(path + "___fuse.png");
    }
  }
  images = process_images(list_images);
  labels = process_labels(list_labels);
};

/*Khi thừa kế lớp này cần override phương thức get
Trả về dữ liệu tương ứng chỉ mục*/
torch::data::Example<> CustomDataset::get(size_t index) {
  torch::Tensor sample_img = images.at(index);
  torch::Tensor sample_label = labels.at(index);
  /*Gọi hàm clone rất quan trọng, nếu không tensor trả về sẽ dùng data của
   * tensor gốc, thay đổi ở bên ngoài sẽ làm thay đổi data này*/
  return {sample_img.clone(), sample_label.clone()};
};

/*Khi thừa kế lớp này cần override phương thức size
Trả về số lượng dữ liệu*/
torch::optional<size_t> CustomDataset::size() const { return labels.size(); }

/* Block mạng con cho mỗi tầng 2 bên */
/* Số channels đầu vào của input và số channels đầu ra sau mạng con này */
ConvBlockImpl::ConvBlockImpl(int64_t in_channels, int64_t out_channels) {
  /* Gọi register_model để gán module conv vào mạng này*/
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

/*Khi tạo một lớp torch::nn::Module cần override phương thức này để truyền đầu
 * vào và tính đâu ra của mạng cho đầu vào, và libtorch tính toán, lưu trữ các
 * thứ cần thiết cho mạng*/
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
  /* Tầng 1 encoder */
  auto enc1 = econv1->forward(x);
  /* Tầng 2 encoder */
  auto enc2 = econv2->forward(torch::max_pool2d(enc1, 2));
  /* Tầng 3 encoder */
  auto enc3 = econv3->forward(torch::max_pool2d(enc2, 2));
  /* Tầng 4 encoder */
  auto enc4 = econv4->forward(torch::max_pool2d(enc3, 2));

  /* Tầng encoder center */
  auto center = cconv->forward(torch::max_pool2d(enc4, 2));

  /* Tầng 4 decoder */
  auto dec4 = dconv4->forward(torch::cat({enc4, upconv4->forward(center)}, 1));
  /* Tầng 3 decoder */
  auto dec3 = dconv3->forward(torch::cat({enc3, upconv3->forward(dec4)}, 1));
  /* Tầng 2 decoder */
  auto dec2 = dconv2->forward(torch::cat({enc2, upconv2->forward(dec3)}, 1));
  /* Tầng 1 decoder */
  auto dec1 = dconv1->forward(torch::cat({enc1, upconv1->forward(dec2)}, 1));

  /* Tích chập đầu ra cho 11 channels tương ứng mỗi lớp*/
  return final_conv->forward(dec1);
}
/* Tính toán mean IoU cho các ảnh cho lớp, đầu vào là tensor
 * (n,480,256)*/
float calc_mIoU(const torch::Tensor& pred, const torch::Tensor& target,
                int class_index) {
  /* Chuyển đổi sang boolean mask cho đầu ra dự đoán và nhãn */
  auto pred_mask = (pred == class_index);
  auto label_mask = (target == class_index);

  /* Tính phần giao của 2 mask này, ra TP*/
  auto inter = torch::sum((pred_mask & label_mask).to(torch::kFloat), {1, 2});
  /* Tính phần hợp của 2 mask này, ra TP + FP + FN*/
  auto uni = torch::sum((pred_mask | label_mask).to(torch::kFloat), {1, 2});

  /* Tính Tp/(TP + FP + FN) cho mỗi nhãn trong lô*/
  auto iou = inter / uni;
  int batch_size = iou.size(0);
  /* Lấy trung bình trong lô */
  return iou.sum().item<float>() / batch_size;
}

/*Tạo ảnh dự đoán, ảnh được lưu vào thư mục pred, cho mỗi thư mục của lớp (nằm
 * trong thư mục dataset/validate)*/
void create_pred_image(torch::Tensor pred, const string& path) {
  const int64_t* tensor_data = pred.data_ptr<int64_t>();

  cv::Mat segment_img(256, 480, CV_8UC3);
  /*Tương ứng mỗi lớp, lấy màu tương ứng*/
  for (int i = 0; i < 256; ++i) {
    for (int j = 0; j < 480; ++j) {
      tuple<int, int, int> color = color_map[tensor_data[i * 480 + j]];
      segment_img.at<cv::Vec3b>(i, j)[0] = get<2>(color);
      segment_img.at<cv::Vec3b>(i, j)[1] = get<1>(color);
      segment_img.at<cv::Vec3b>(i, j)[2] = get<0>(color);
    }
  }
  cv::imwrite(path + "__pred.jpg", segment_img);
}
/* Tính mean IoU cho mỗi lớp */
vector<float> evaluate_IoU(shared_ptr<ForwardNet> net,
                           const string& milestone) {
  vector<float> mIoU(11);

  /*Trong thư mục validate có thư mục chứa các ảnh dành để tính IoU cho mỗi lớp,
  lớp tên gì thư mục tên đó
  */
  for (const auto& entry :
       filesystem::directory_iterator("../dataset/validate")) {
    if (!entry.is_directory()) continue;
    string path = entry.path().string();
    string filename = entry.path().filename().string();

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
    CustomDataset dataset(path);
    auto loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            dataset.map(torch::data::transforms::Stack<>()),
            dataset.size().value());

    for (auto& batch : *loader) {
      /* Tính đầu ra dự đoán */
      torch::Tensor pred_prob = net->forward(batch.data);
      /* Bỏ chiều đầu tiên ra thành (C,W,H) */

      /*Tương ứng mỗi điểm dữ liệu ảnh, tìm lớp có xác suất cao nhất*/
      torch::Tensor pred = torch::argmax(pred_prob, 1).to(torch::kLong);
      float get = calc_mIoU(pred, batch.target, class_index);
      mIoU[class_index] = get;

      /* Tạo ảnh tất cả ảnh dự đoán trong trong thư mục của lớp và lưu trong thư
       * mục pred*/
      for (int i = 0; i < (int)dataset.size().value(); i++) {
        auto pred_tensor_of_image = pred.select(0, i);
        filesystem::path p(dataset.list_images[i]);

        create_pred_image(
            pred_tensor_of_image,
            path + "/pred/" + p.filename().string() + milestone + ".jpg");
      }
    }
  }
  return mIoU;
}

/* Đánh giá model cho từng learning rate */
void train_with_lr(float lr, const string& model_name) {
  shared_ptr<ForwardNet> net;
  if (model_name == "Unet") {
    net = make_shared<UNet>(64);
  } else{
    net = make_shared<EffUNet>();
  }
  /* Tải dữ liệu train lên */
  auto data_loader =
      torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
          std::move(CustomDataset("../dataset/final-dataset")
                        .map(torch::data::transforms::Stack<>())),
          torch::data::DataLoaderOptions().batch_size(4));

  /* Tải dữ liệu validation lên */
  auto val_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          CustomDataset(("../dataset/validate"))
              .map(torch::data::transforms::Stack<>()),
          3);

  const string filename = model_name + "_lr_" + to_string_float(lr);

  /* Tạo file csv, thống kê error, accuracy, dùng để plot và đánh giá khả năng
   * của model*/
  std::ofstream stat_iou("../stat/" + filename + ".csv");
  stat_iou << "Number of epoch,training-error,validate-error,Team B,Team A,GK "
           << "B,GK A,Ground,Advertisement,Audience,Goal Bar,Ball,Referee,Coach"
           << endl;
  /*Tham khảo : https://pytorch.org/cppdocs/frontend.html#end-to-end-example
   */

  /* tạo một data loader load sử dụng theo lô với kích thước 10 cho thuật
  toán
   * SGD và xóa trộn dữ liệu ngẫu nhiên vỡi mỗi epoch*/

  /* Khởi tạo thuật toán tối ưu SGD và Adam với momentum để cập nhật các tham số
   * trong mạng. */
  torch::optim::Adam optimizer(net->parameters(),
                               torch::optim::AdamOptions(lr));

  /* Bắt đầu train, với số lượng epoch cho trước, sau mỗi epoch tính toán
  lượng error (cross-entropy) và IoU tương ứng mỗi lớp*/

  auto start = std::chrono::high_resolution_clock::now();
  int max_epoch = 50;

  cout << "Start training for lr = " << lr << endl;
  for (int epoch = 0; epoch < max_epoch; ++epoch) {
    /*error của dữ liệu training, do khả năng của máy tính cá nhân có hạn nên
     * lấy trung bình cho các batch làm xấp xỉ*/
    pair<float, int> ce_loss = make_pair(0, 0);
    int cnt_batch = 0;
    /*Duyệt qua các lô dữ liệu đã được xáo trộn*/
    for (auto& batch : *data_loader) {
      auto start_batch = std::chrono::high_resolution_clock::now();
      /* Reset gradients về không, vì vòng lặp trước để lại giá trị cũ*/
      optimizer.zero_grad();
      /* Tính đầu ra dự đoán */
      torch::Tensor prediction = net->forward(batch.data);

      /* Đầu ra của mạng có kích thước (n,C,H,W) (với n là kích thước lô, C
       * là số lớp, W là chiều rộng ảnh và H là chiều cao ảnh) mỗi vị trí
       * tương ứng xác suất pixel đó thuộc về lớp nào, nhãn có kích thước
       * (n,H,W), mỗi vị trí cho biết pixel thuộc lớp nào.*/
      auto loss =
          torch::nn::functional::cross_entropy(prediction, batch.target);
      /* back propagation, tính gradient. */
      loss.backward();
      ce_loss.first += loss.item<float>();
      ce_loss.second += batch.data.size(0);
      /* Cập nhật các tham sô với gradient */
      optimizer.step();

      auto end_batch = std::chrono::high_resolution_clock::now();
      std::cout << "Time of batch " + to_string(cnt_batch) << " : "
                << std::chrono::duration_cast<std::chrono::seconds>(end_batch -
                                                                    start_batch)
                       .count()
                << " seconds" << endl;
      cnt_batch++;
    }

    /*error dữ liệu validation*/
    float val_error = 0;
    for (auto& batch : *val_loader) {
      torch::Tensor pred_prob = net->forward(batch.data);
      auto loss = torch::nn::functional::cross_entropy(pred_prob, batch.target);
      val_error = loss.item<float>();
    }
    val_error /= 5;
    /*tính IoU*/
    vector<float> mIoU_each_class =
        evaluate_IoU(net, model_name + "_lr" + to_string_float(lr) + "_epoch" +
                              to_string(epoch));

    float error = ce_loss.first / ce_loss.second;
    cout << "End epoch " << epoch + 1 << " with error: " << error
         << " , validation error: " << val_error << endl;

    stat_iou << epoch << "," << error << "," << val_error;
    for (int i = 0; i < 11; i++) {
      stat_iou << "," << mIoU_each_class[i];
    }
    stat_iou << endl;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      chrono::duration_cast<std::chrono::seconds>(end - start).count();
  cout << "Time train with: " + to_string(max_epoch) + " :" << duration
       << " seconds" << endl;
  cout << "End training" << endl;
  /*Lưu lại model cho người dùng, sử dụng file thực thi build/TestModel, đọc
   * hướng dẫn sử dụng trong report*/
  torch::save(net, "../models/" + filename + ".pt");

  stat_iou.close();
}

/* Hàm to_string của C++ cho float có thể tạo ra các số 0 ở cuối để đủ 6 chữ số,
viết lại hàm này để xóa các chữ số 0 ở cuối*/
string to_string_float(float value) {
  string s = to_string(value);
  if ((int)s.length() == 1) {
    return s;
  }
  while (s.back() == '0') {
    s.pop_back();
  }

  return s;
}
SEBlockImpl::SEBlockImpl(int in_channels) {
  fc1 = register_module("fc1", torch::nn::Linear(in_channels, in_channels / 4));
  fc2 = register_module("fc2", torch::nn::Linear(in_channels / 4, in_channels));
}

torch::Tensor SEBlockImpl::forward(torch::Tensor x) {
  /* Squeeze : (N,C,W,H) -> (N,C,1,1)*/
  torch::Tensor z = torch::adaptive_avg_pool2d(x, {1, 1});
  /*(N,C,1,1) -> (N,C) giảm chiều cho lớp kết nối đầy đủ 1D*/
  z = z.view({-1, x.size(1)});

  /*Excitation : 2 lớp kết nối đầy đủ, số node đầu ra của lớp đầu tiên là số
   * chiều được giảm từ C chiều, C/4 */
  z = fc1->forward(z);
  torch::relu_(z);
  z = fc2->forward(z);
  torch::sigmoid_(z);
  z = z.view({-1, x.size(1), 1, 1});
  return z;
}

MBConvImpl::MBConvImpl(int in_channels, int out_channels, int expan, int stride,
                       int kernel) {
  /* Số lượng channel được nông ra */
  int echannels = in_channels * expan;
  /* pading để giữ độ phân giải*/
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
  /* Để lại đầu vòa cho kết nối thặng dư*/
  auto pre = x;
  /* 1x1 tích chập, nông số lượng channel*/
  x = conv1->forward(x);
  x = bn1->forward(x);
  torch::relu_(x);

  /* chiều sâu tích chập, mỗi channel đầu ra chỉ tích chập với channel đầu vào
   * với độ sâu bằng với nó*/
  x = conv2->forward(x);
  x = bn2->forward(x);
  torch::relu_(x);

  /* lớp Squeeze Excitation */
  auto z = se->forward(x);
  x = z * x;

  /* Lớp tích chập giảm số channel thành số channel đầu ra*/
  x = conv3->forward(x);
  x = bn3->forward(x);

  /* Tổng thặng dư */
  if (pre.sizes() == x.sizes()) {
    x = x + pre;
  }
  return x;
}
torch::Tensor MBBlockImpl::forward(torch::Tensor x) {
  /* Tương ứng mỗi block, các lớp mbconv có số channel đầu vào là đầu ra từ lớp
   * trước, số channel đầu ra thì như đã liệt kê cho EffcientNetb7*/
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