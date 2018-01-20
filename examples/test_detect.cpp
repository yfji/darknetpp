#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <darknet.h>

void test(char* input){
	list *options = read_data_cfg("./cfg/coco.data");
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);

	image **alphabet = load_alphabet();
	network *net = load_network("./cfg/yolo.cfg", "weights/yolo.weights", 0);
	set_batch_network(net, 1);
	for(auto i=0;i<net->n;++i)
		std::cout<<net->layers[i].batch<<std::endl;
	std::cout<<"test detect: "<<net->layers[net->n-1].classes<<","<<net->layers[net->n-1].coords<<std::endl;
	image im = load_image_color(input,0,0);
	image sized = letterbox_image(im, net->w, net->h);
	srand(2222222);
}
