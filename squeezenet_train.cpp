#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>

using namespace std;
using namespace dlib;

template <typename SUBNET> using fire_expand_a1 = relu<con<64,1,1,1,1,SUBNET>>;
template <typename SUBNET> using fire_expand_a2 = relu<con<64,3,3,1,1,SUBNET>>;
template <typename SUBNET> using fire_squeeze_a = inception2<fire_expand_a1,fire_expand_a2,SUBNET>;

template <typename SUBNET> using fire_expand_b1 = relu<con<128,1,1,1,1,SUBNET>>;
template <typename SUBNET> using fire_expand_b2 = relu<con<128,3,3,1,1,SUBNET>>;
template <typename SUBNET> using fire_squeeze_b = inception2<fire_expand_b1,fire_expand_b2,SUBNET>;

template <typename SUBNET> using fire_expand_c1 = relu<con<192,1,1,1,1,SUBNET>>;
template <typename SUBNET> using fire_expand_c2 = relu<con<192,3,3,1,1,SUBNET>>;
template <typename SUBNET> using fire_squeeze_c = inception2<fire_expand_c1,fire_expand_c2,SUBNET>;

template <typename SUBNET> using fire_expand_d1 = relu<con<256,1,1,1,1,SUBNET>>;
template <typename SUBNET> using fire_expand_d2 = relu<con<256,3,3,1,1,SUBNET>>;
template <typename SUBNET> using fire_squeeze_d = inception2<fire_expand_d1,fire_expand_d2,SUBNET>;

// -- Model ------------------------------------------------------------------>

using squeeze_net = loss_multiclass_log<
	avg_pool_everything<
	relu<con<2,1,1,1,1,
	dropout<
	fire_squeeze_d<
	relu<con<64,1,1,1,1,
	max_pool<3,3,2,2,
	fire_squeeze_d<
	relu<con<64,1,1,1,1,
	fire_squeeze_c<
	relu<con<48,1,1,1,1,
	fire_squeeze_c<
	relu<con<48,1,1,1,1,
	fire_squeeze_b<
	relu<con<32,1,1,1,1,
	max_pool<3,3,2,2,
	fire_squeeze_b<
	relu<con<32,1,1,1,1,
	fire_squeeze_a<
	relu<con<16,1,1,1,1,
	fire_squeeze_a<
	relu<con<16,1,1,1,1,
	max_pool<3,3,2,2,
	relu<con<96,7,7,2,2,
	input_rgb_image_sized<224>
	>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

// ----------------------------------------------------------------- !Model -->

// ----------------------------------------------------------------------------------------

rectangle make_random_cropping_rect_resnet(
		const matrix<rgb_pixel>& img,
		dlib::rand& rnd
)
{
	// figure out what rectangle we want to crop from the image
	double mins = 0.466666666, maxs = 0.875;
	auto scale = mins + rnd.get_random_double()*(maxs-mins);
	auto size = scale*std::min(img.nr(), img.nc());
	rectangle rect(size, size);
	// randomly shift the box around
	point offset(rnd.get_random_32bit_number()%(img.nc()-rect.width()),
	             rnd.get_random_32bit_number()%(img.nr()-rect.height()));
	return move_rect(rect, offset);
}

// ----------------------------------------------------------------------------------------

void randomly_crop_image (
		const matrix<rgb_pixel>& img,
		matrix<rgb_pixel>& crop,
		dlib::rand& rnd
)
{
	auto rect = make_random_cropping_rect_resnet(img, rnd);

	// now crop it out as a 227x227 image.
	extract_image_chip(img, chip_details(rect, chip_dims(224,224)), crop);

	// Also randomly flip the image
	if (rnd.get_random_double() > 0.5)
		crop = fliplr(crop);

	// And then randomly adjust the colors.
	apply_random_color_offset(crop, rnd);
}

void randomly_crop_images (
		const matrix<rgb_pixel>& img,
		dlib::array<matrix<rgb_pixel>>& crops,
		dlib::rand& rnd,
		long num_crops
)
{
	std::vector<chip_details> dets;
	for (long i = 0; i < num_crops; ++i)
	{
		auto rect = make_random_cropping_rect_resnet(img, rnd);
		dets.push_back(chip_details(rect, chip_dims(224,224)));
	}

	extract_image_chips(img, dets, crops);

	for (auto&& img : crops)
	{
		// Also randomly flip the image
		if (rnd.get_random_double() > 0.5)
			img = fliplr(img);

		// And then randomly adjust the colors.
		apply_random_color_offset(img, rnd);
	}
}

// ----------------------------------------------------------------------------------------

struct image_info
{
	string filename;
	string label;
	long numeric_label;
};

std::vector<image_info> get_image_listing(
		const std::string& images_folder,
		const std::string& label,
		const long& numeric_label
)
{
	std::vector<image_info> results;
	image_info temp;
	temp.numeric_label = numeric_label;

	auto dir = directory(images_folder);
	temp.label = label;

	for (auto image_file : dir.get_files()) {
		temp.filename = image_file;
		results.push_back(temp);
	}

	return results;
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
	if (argc != 4)
	{
		cout << "Usage: " << endl;
		cout << "./ddfd_train /path/to/positive/images /path/to/negative/images /path/to/validation/images" << endl;
		return 1;
	}

	cout << "\nSCANNING IMAGES\n" << endl;

	auto positive_images = get_image_listing(string(argv[1]), "face", 1);
	cout << "positive examples: " << positive_images.size() << endl;

	auto negative_images = get_image_listing(string(argv[2]), "not_face", 0);
	cout << "negative examples: " << negative_images.size() << endl;

	set_dnn_prefer_smallest_algorithms();

	const double initial_learning_rate = 0.04;
	const double weight_decay = 0.0002;
	const double momentum = 0.9;

	std::vector<matrix<rgb_pixel>> samples;
	std::vector<unsigned long> labels;

	squeeze_net net;

	dnn_trainer<squeeze_net> trainer(net,sgd(weight_decay, momentum));

	trainer.be_verbose();
	trainer.set_learning_rate(initial_learning_rate);
	trainer.set_synchronization_file("squeezenet_trainer_state_file.dat", std::chrono::minutes(10));
	// This threshold is probably excessively large.  You could likely get good results
	// with a smaller value but if you aren't in a hurry this value will surely work well.
	trainer.set_iterations_without_progress_threshold(20000);

	dlib::pipe<std::pair<image_info,matrix<rgb_pixel>>> n_data(200);
	auto fn = [&n_data, &negative_images](time_t seed)
	{
		dlib::rand rnd(time(0)+seed);
		matrix<rgb_pixel> img;
		std::pair<image_info, matrix<rgb_pixel>> temp;
		while(n_data.is_enabled())
		{
			auto n_rnd = rnd.get_random_32bit_number()%negative_images.size();
			temp.first = negative_images[n_rnd];

			try {
				load_image(img, temp.first.filename);
				randomly_crop_image(img, temp.second, rnd);
				n_data.enqueue(temp);
			} catch (dlib::image_load_error& e) {
				negative_images.erase(negative_images.begin() + n_rnd);
			}
		}
	};
	std::thread ndata_loader1([fn](){ fn(1); });
	std::thread ndata_loader2([fn](){ fn(2); });
	std::thread ndata_loader3([fn](){ fn(3); });
	std::thread ndata_loader4([fn](){ fn(4); });

	dlib::pipe<std::pair<image_info,matrix<rgb_pixel>>> p_data(200);
	auto fp = [&p_data, &positive_images](time_t seed)
	{
		dlib::rand rnd(time(0)+seed);
		matrix<rgb_pixel> img;
		std::pair<image_info, matrix<rgb_pixel>> temp;
		while(p_data.is_enabled())
		{
			auto p_rnd = rnd.get_random_32bit_number()%positive_images.size();
			temp.first = positive_images[p_rnd];

			try {
				load_image(img, temp.first.filename);
				randomly_crop_image(img, temp.second, rnd);
				p_data.enqueue(temp);
			} catch (dlib::image_load_error& e) {
				positive_images.erase(positive_images.begin() + p_rnd);
			}
		}
	};
	std::thread pdata_loader1([fp](){ fp(1); });
	std::thread pdata_loader2([fp](){ fp(2); });
	std::thread pdata_loader3([fp](){ fp(3); });
	std::thread pdata_loader4([fp](){ fp(4); });

	// The main training loop.  Keep making mini-batches and giving them to the trainer.
	// We will run until the learning rate has dropped by a factor of 1e-3.
	while(trainer.get_learning_rate() >= initial_learning_rate*1e-3)
	{
		samples.clear();
		labels.clear();

		// make a 128 image mini-batch, 32 positive, 96 negative samples
		std::pair<image_info, matrix<rgb_pixel>> img;

		for (int i = 0; i < 32; ++i) {
			p_data.dequeue(img);

			samples.push_back(std::move(img.second));
			labels.push_back(1);
		}

		for (int i = 0; i < 96; ++i) {
			n_data.dequeue(img);

			samples.push_back(std::move(img.second));
			labels.push_back(0);
		}

		trainer.train_one_step(samples, labels);
	}

	// Training done, tell threads to stop and make sure to wait for them to finish before
	// moving on.
	p_data.disable();
	pdata_loader1.join();
	pdata_loader2.join();
	pdata_loader3.join();
	pdata_loader4.join();

	n_data.disable();
	ndata_loader1.join();
	ndata_loader2.join();
	ndata_loader3.join();
	ndata_loader4.join();

	// also wait for threaded processing to stop in the trainer.
	trainer.get_net();

	net.clean();
	cout << "saving network" << endl;
	serialize("squeezenet.dnn") << net;

	// Now test the network on the validation dataset.  First, make a testing
	// network with softmax as the final layer.  We don't have to do this if we just wanted
	// to test the "top1 accuracy" since the normal network outputs the class prediction.
	// But this snet object will make getting the top5 predictions easy as it directly
	// outputs the probability of each class as its final output.
	softmax<squeeze_net::subnet_type> snet; snet.subnet() = net.subnet();

	cout << "Testing network on validation dataset..." << endl;
	int num_right = 0;
	int num_wrong = 0;
	int num_right_top1 = 0;
	int num_wrong_top1 = 0;
	dlib::rand rnd(time(0));
	// loop over all the validation images
	for (auto l : get_image_listing(string(argv[3]), "face", 1))
	{
		dlib::array<matrix<rgb_pixel>> images;
		matrix<rgb_pixel> img;
		load_image(img, l.filename);
		// Grab 16 random crops from the image.  We will run all of them through the
		// network and average the results.
		const int num_crops = 16;
		randomly_crop_images(img, images, rnd, num_crops);
		// p(i) == the probability the image contains object of class i.
		matrix<float,1,2> p = sum_rows(mat(snet(images.begin(), images.end())))/num_crops;

		// check top 1 accuracy
		if (index_of_max(p) == l.numeric_label)
			++num_right_top1;
		else
			++num_wrong_top1;

		// check top 5 accuracy
		bool found_match = false;
		for (int k = 0; k < 2; ++k)
		{
			long predicted_label = index_of_max(p);
			p(predicted_label) = 0;
			if (predicted_label == l.numeric_label)
			{
				found_match = true;
				break;
			}

		}
		if (found_match)
			++num_right;
		else
			++num_wrong;
	}
	cout << "val top2 accuracy:  " << num_right/(double)(num_right+num_wrong) << endl;
	cout << "val top1 accuracy:  " << num_right_top1/(double)(num_right_top1+num_wrong_top1) << endl;
}
catch(std::exception& e)
{
	cout << e.what() << endl;
}



