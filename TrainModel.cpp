#include "header.h"

int main(int argc, char* argv[]) {
  string model(argv[1]);
  float lr = atof(argv[2]);
  /* Ánh xạ lớp tương ứng với màu */
  color_index[255][160][1] = 0;
  color_index[254][233][3] = 1;
  color_index[51][188][255] = 2;
  color_index[0][255][0] = 3;
  color_index[137][126][126] = 4;
  color_index[27][71][151] = 5;
  color_index[111][48][253] = 6;
  color_index[255][0][29] = 7;
  color_index[201][19][223] = 8;
  color_index[238][171][171] = 9;
  color_index[0][0][0] = 10;

  color_map[0] = make_tuple(255, 160, 1);
  color_map[1] = make_tuple(254, 233, 3);
  color_map[2] = make_tuple(51, 188, 255);
  color_map[3] = make_tuple(0, 255, 0);
  color_map[4] = make_tuple(137, 126, 126);
  color_map[5] = make_tuple(27, 71, 151);
  color_map[6] = make_tuple(111, 48, 253);
  color_map[7] = make_tuple(255, 0, 29);
  color_map[8] = make_tuple(201, 19, 223);
  color_map[9] = make_tuple(238, 171, 171);
  color_map[10] = make_tuple(0, 0, 0);

  train_with_lr(lr, model);
}