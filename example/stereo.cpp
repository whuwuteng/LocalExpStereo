
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <omp.h>
#include <time.h>

#include <opencv2/opencv.hpp>
#include "FastGCStereo.h"
#include "Evaluator.h"
#include "ArgsParser.h"
#include "CostVolumeEnergy.h"
#include "Utilities.hpp"

#ifdef __unix
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h> 
#else
#include <direct.h>
#endif

struct Options
{
	std::string mode = ""; // "MiddV2" or "MiddV3"
	std::string outputDir = "";
	std::string targetDir = "";

	int iterations = 5;
	int pmIterations = 2;
	bool doDual = false;

	int ndisp = 0;
	float smooth_weight = 1.0;
	float mc_threshold = 0.5;
	int filterRadious = 20;

	int threadNum = -1;

	void loadOptionValues(ArgsParser& argParser)
	{
		argParser.TryGetArgment("outputDir", outputDir);
		argParser.TryGetArgment("targetDir", targetDir);
		argParser.TryGetArgment("mode", mode);

		if (mode == "MiddV2")
			smooth_weight = 1.0;
		else if (mode == "MiddV3")
			smooth_weight = 0.5;

		argParser.TryGetArgment("threadNum", threadNum);
		argParser.TryGetArgment("doDual", doDual);
		argParser.TryGetArgment("iterations", iterations);
		argParser.TryGetArgment("pmIterations", pmIterations);

		argParser.TryGetArgment("ndisp", ndisp);
		argParser.TryGetArgment("filterRadious", filterRadious);
		argParser.TryGetArgment("smooth_weight", smooth_weight);
		argParser.TryGetArgment("mc_threshold", mc_threshold);
	}

	void printOptionValues(FILE * out = stdout)
	{
		fprintf(out, "----------- parameter settings -----------\n");
		fprintf(out, "mode           : %s\n", mode.c_str());
		fprintf(out, "outputDir      : %s\n", outputDir.c_str());
		fprintf(out, "targetDir      : %s\n", targetDir.c_str());

		fprintf(out, "threadNum      : %d\n", threadNum);
		fprintf(out, "doDual         : %d\n", (int)doDual);
		fprintf(out, "pmIterations   : %d\n", pmIterations);
		fprintf(out, "iterations     : %d\n", iterations);

		fprintf(out, "ndisp          : %d\n", ndisp);
		fprintf(out, "filterRadious  : %d\n", filterRadious);
		fprintf(out, "smooth_weight  : %f\n", smooth_weight);
		fprintf(out, "mc_threshold   : %f\n", mc_threshold);
	}
};

const Parameters paramsBF = Parameters(20, 20, "BF", 10);
const Parameters paramsGF = Parameters(1.0f, 20, "GF", 0.0001f);
const Parameters paramsGFfloat = Parameters(1.0f, 20, "GFfloat", 0.0001f); // Slightly faster

struct Calib
{
	float cam0[3][3];
	float cam1[3][3];
	float doffs;
	float baseline;
	int width;
	int height;
	int dispmin;
	int dispmax;
	int isint;
	int vmin;
	int vmax;
	float dyavg;
	float dymax;
	float gt_prec; // for V2 only

	// ----------- format of calib.txt ----------
	//cam0 = [2852.758 0 1424.085; 0 2852.758 953.053; 0 0 1]
	//cam1 = [2852.758 0 1549.445; 0 2852.758 953.053; 0 0 1]
	//doffs = 125.36
	//baseline = 178.089
	//width = 2828
	//height = 1924
	//ndisp = 260
	//isint = 0
	//vmin = 36
	//vmax = 218
	//dyavg = 0.408
	//dymax = 1.923

	Calib()
		: doffs(0)
		, baseline(0)
		, width(0)
		, height(0)
		, dispmin(0)
		, dispmax(0)
		, isint(0)
		, vmin(0)
		, vmax(0)
		, dyavg(0)
		, dymax(0)
		, gt_prec(-1)
	{
	}

	// not used
	Calib(std::string filename)
		: Calib()
	{
		FILE* fp = fopen(filename.c_str(), "r");
		char buff[512];

		if (fp != nullptr)
		{
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "cam0 = [%f %f %f; %f %f %f; %f %f %f]\n", &cam0[0][0], &cam0[0][1], &cam0[0][2], &cam0[1][0], &cam0[1][1], &cam0[1][2], &cam0[2][0], &cam0[2][1], &cam0[2][2]);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "cam1 = [%f %f %f; %f %f %f; %f %f %f]\n", &cam1[0][0], &cam1[0][1], &cam1[0][2], &cam1[1][0], &cam1[1][1], &cam1[1][2], &cam1[2][0], &cam1[2][1], &cam1[2][2]);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "doffs = %f\n", &doffs);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "baseline = %f\n", &baseline);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "width = %d\n", &width);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "height = %d\n", &height);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "dispmin = %d\n", &dispmin);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "dispmax = %d\n", &dispmax);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "isint = %d\n", &isint);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "vmin = %d\n", &vmin);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "vmax = %d\n", &vmax);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "dyavg = %f\n", &dyavg);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "dymax = %f\n", &dymax);
			fclose(fp);
		}
	}
};

void fillOutOfView(cv::Mat& volume, int mode, int margin = 0)
{
	int D = volume.size.p[0];
	int H = volume.size.p[1];
	int W = volume.size.p[2];

	if (mode == 0)
	for (int d = 0; d < D; d++)
	for (int y = 0; y < H; y++)
	{
		auto p = volume.ptr<float>(d, y);
		auto q = p + d + margin;
		float v = *q;
		while (p != q){
			*p = v;
			p++;
		}
	}
	else
	for (int d = 0; d < D; d++)
	for (int y = 0; y < H; y++)
	{
		auto q = volume.ptr<float>(d, y) + W;
		auto p = q - d - margin;
		float v = p[-1];
		while (p != q){
			*p = v;
			p++;
		}
	}
}

cv::Mat convertVolumeL2R(cv::Mat& volSrc, int margin = 0)
{
	int D = volSrc.size[0];
	int H = volSrc.size[1];
	int W = volSrc.size[2];
	cv::Mat volDst = volSrc.clone();

	for (int d = 0; d < D; d++)
	{
		cv::Mat_<float> s0(H, W, volSrc.ptr<float>() + H*W*d);
		cv::Mat_<float> s1(H, W, volDst.ptr<float>() + H*W*d);
		s0(cv::Rect(d, 0, W - d, H)).copyTo(s1(cv::Rect(0, 0, W - d, H)));

		cv::Mat edge1 = s0(cv::Rect(W - 1 - margin, 0, 1, H)).clone();
		cv::Mat edge0 = s0(cv::Rect(d + margin, 0, 1, H)).clone();
		for (int x = W - 1 - d - margin; x < W; x++)
			edge1.copyTo(s1.col(x));
		for (int x = 0; x < margin; x++)
			edge0.copyTo(s1.col(x));
	}
	return volDst;
}

bool loadData(const std::string inputDir, cv::Mat& im0, cv::Mat& im1, Calib& calib)
{
	std::string strImgL;
	std::string strImgR;

	std::ifstream infile;
	infile.open((inputDir + "info.txt").c_str());
	if ( infile.good()) {
		char strLine[256] = { 0 };
		infile.getline(strLine, 256);
		strImgL = std::string(strLine);
		
		infile.getline(strLine, 256);
		strImgR = std::string(strLine);
		
		int dispmin = 0;
		int dispmax = 0;
		
		infile.getline(strLine, 256);
		sscanf(strLine, "%d %d", & dispmin, & dispmax);
		calib.dispmin = dispmin;
		calib.dispmax = dispmax;
		
		infile.close();
	}
	else{
		printf("ndisp is not speficied.\n");
		return false;
	}

	im0 = cv::imread(inputDir + strImgL);
	im1 = cv::imread(inputDir + strImgR);

	if (im0.empty() || im1.empty()){
		printf("Image pairs (im0.png, im1.png) or (imL.png, imR.png) not found in\n");
		printf("%s\n", inputDir.c_str());
		return false;
	}
	
	return true;
}

void stereo_match(const std::string inputDir, const std::string outputDir, const Options& options)
{
	cv::Mat imL, imR;
	Calib calib;

	if (loadData(inputDir, imL, imR, calib) == false)
		return;
	printf("disp min = %d \n", calib.dispmin);
	printf("disp max = %d \n", calib.dispmax);

	float errorThresh = 0.5f;
	float vdisp = 0; // Purtubation of vertical displacement in the range of [-vdisp, vdisp]
	float maxdisp = (float)(calib.dispmax - calib.dispmin);

	Parameters param = paramsGF;
	param.windR = options.filterRadious;
	param.lambda = options.smooth_weight;

	{
#ifdef __unix
		if(access((outputDir + "debug").c_str(), R_OK | W_OK) != 0){
			mkdir((outputDir + "debug").c_str(), 0755);
		}
#else
		_mkdir((outputDir + "debug").c_str());
#endif

		FastGCStereo stereo(imL, imR, param, calib.dispmax, calib.dispmin, vdisp);
		stereo.saveDir = outputDir + "debug/";

		IProposer* prop1 = new ExpansionProposer(1);
		IProposer* prop2 = new RandomProposer(7, maxdisp);
		IProposer* prop3 = new ExpansionProposer(2);
		IProposer* prop4 = new RansacProposer(1);
		stereo.addLayer(5, { prop1, prop4, prop2 });
		stereo.addLayer(15, { prop3, prop4 });
		stereo.addLayer(25, { prop3, prop4 });

		cv::Mat labeling, rawdisp;
		if (options.doDual)
			stereo.run(options.iterations, { 0, 1 }, options.pmIterations, labeling, rawdisp);
		else
			stereo.run(options.iterations, { 0 }, options.pmIterations, labeling, rawdisp);

		delete prop1;
		delete prop2;
		delete prop3;
		delete prop4;

		cvutils::io::save_pfm_file(outputDir + "disp0.pfm", stereo.getEnergyInstance().computeDisparities(labeling));
		
		cv::Mat ucharImg;
		stereo.getEnergyInstance().computeDisparities(labeling).convertTo(ucharImg, CV_8U, 255.0);
		cv::imwrite(outputDir + "disp0.png", ucharImg);
		
		if (options.doDual){
			cvutils::io::save_pfm_file(outputDir + "disp0raw.pfm", stereo.getEnergyInstance().computeDisparities(rawdisp));
		
			cv::Mat ucharImg_raw;
			stereo.getEnergyInstance().computeDisparities(rawdisp).convertTo(ucharImg_raw, CV_8U, 255.0);
			cv::imwrite(outputDir + "disp0raw.png", ucharImg_raw);
		}
	}
}

int main(int argc, const char** args)
{
	ArgsParser parser(argc, args);
	Options options;
	options.loadOptionValues(parser);
	unsigned int seed = (unsigned int)time(NULL);
/*
#if 0
	// For debugging MiddV3
	//  1  99.4        262247  252693  9554    10.51   8.54
	options.targetDir = "../data/MiddV3/trainingH/Adirondack";
	options.outputDir = "../results/Adirondack";
	options.mode = "MiddV3";
	options.smooth_weight = 0.5;
	options.pmIterations = 2;
	//options.threadNum = 1;
	seed = 0;
#else 
	// For debugging MiddV2
	options.targetDir = "../data/MiddV2/vaihingen";
	options.outputDir = "../results/vaihingen";
	options.mode = "MiddV2";
	options.smooth_weight = 1;
	options.doDual = 1;
	//options.threadNum = 1;
	seed = 0;
#endif
*/
	options.printOptionValues();

	int nThread = omp_get_max_threads();
	#pragma omp parallel for
	for (int j = 0; j < nThread; j++)
	{
		srand(seed + j);
		cv::theRNG() = seed + j;
	}

	if (options.threadNum > 0)
		omp_set_num_threads(options.threadNum);

	if (options.outputDir.length()){
#ifdef __unix
		if(access((options.outputDir).c_str(), R_OK | W_OK) != 0){
			if(mkdir((options.outputDir).c_str(), 0755) == -1){
				printf("can not make directory %s \n", (options.outputDir).c_str());
				return -1;
			}
		}
#else
		_mkdir((options.outputDir).c_str());
#endif
	}

	printf("\n\n");
	
	stereo_match(options.targetDir + "/", options.outputDir + "/", options);

	return 0;
}
