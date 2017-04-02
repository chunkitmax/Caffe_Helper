/*
 * helper.cpp
 *
 *  Created on: Apr 1, 2017
 *      Author: petel__
 */

#include <vector>
#include <iostream>
#include <dirent.h>
#include <regex>
#include <string>

#include <cuda_runtime.h>

#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/solver.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/layers/memory_data_layer.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "boost/scoped_ptr.hpp"
#include "boost/shared_ptr.hpp"

using namespace std;
using namespace caffe;
using namespace boost;
using namespace db;

#define GetCurrentDir getcwd

void getGPUs(vector<int>* gpus);
void printHelp();

const string MODEL_PATH_PREFIX = "../nn/";
const string PROTOTXT_PATH_PREFIX = "../nn/prototxt/";
const string TRAININGSET_PATH_PREFIX = "../nn/training_set/";
const string TEST_DATA_PATH_PREFIX = "../nn/";

//const string MODEL_PATH_PREFIX = "../nn/";
//const string PROTOTXT_PATH_PREFIX = "../nn/prototxt/";
//const string TRAININGSET_PATH_PREFIX = "../nn/training_set/";
//const string TEST_DATA_PATH_PREFIX = "../nn/";

int main(int argc, char **argv)
{
//	cout << "Hello World!" << endl;
//	caffe::GlobalInit(&argc, &argv);

	if (argc < 2)
	{
		printHelp();
		return 0;
	}

	if (!strcmp(argv[1], "test"))
	{
		Caffe::SetDevice(0);
		Caffe::set_mode(Caffe::GPU);

		float loss = 0.0;
		Net<float> net(PROTOTXT_PATH_PREFIX + "net.prototxt", Phase::TEST);
		DIR *dir;
		struct dirent *ent;
		if ((dir = opendir("./")) != NULL)
		{
			while ((ent = readdir(dir)) != NULL)
			{
				if (regex_match(ent->d_name, regex(".+\.caffemodel")))
				{
					net.CopyTrainedLayersFrom(string("./") + ent->d_name);
					break;
				}
			}
			closedir(dir);
			if (ent == NULL)
			{
				printHelp();
				return 0;
			}
		}
		else
		{
			cout << "Error: Cannot find .caffemodel file in " + TRAININGSET_PATH_PREFIX << "." << endl << endl;
			return 0;
		}

		ifstream inputFile(TRAININGSET_PATH_PREFIX + "dataList.txt");
		int dataCount, labelCount, width, height, isGray;
		inputFile >> dataCount >> labelCount >> width >> height >> isGray;
		inputFile.close();

		Datum datum, lab_datum;
		if (argv[1] == NULL)
		{
			printHelp();
			return 0;
		}

//		lab_datum.set_channels(1);
//		lab_datum.set_height(1);
//		lab_datum.set_width(1);
//		lab_datum.add_float_data(1);
////		cout << "OK" << endl;
//		MemoryDataLayer<float> *memoryLabDataLayer = (MemoryDataLayer<float> *)net.layer_by_name("label").get();
////		cout << "OK" << endl;
//		memoryLabDataLayer->AddDatumVector(vector<Datum>(1, lab_datum));
//		cout << "OK" << endl;

		cout << (TEST_DATA_PATH_PREFIX + argv[2]) << " " << width << " " << height << " " << isGray << endl;
		bool status = ReadImageToDatum(TEST_DATA_PATH_PREFIX + argv[2], -1, width, height, &datum);
		if (!status)
		{
			cout << "Error: Test files cannot be loaded." << endl << endl;
			return 0;
		}
		MemoryDataLayer<float> *memoryDataLayer = (MemoryDataLayer<float> *)net.layer_by_name("data").get();
		vector<Datum> tmp(1);
		tmp[0] = datum;
		memoryDataLayer->AddDatumVector(tmp);
		cout << "OK" << endl;
		const vector<Blob<float> *> &result = net.Forward(&loss);

		cout << "OK" << endl;

		const boost::shared_ptr<Blob<float>> &probLayer = net.blob_by_name("prob");

		cout << "----------------------------------------------------------" << endl;
		const float *probs_out = probLayer->cpu_data();
		for (uint i = 0; i < 13; i++)
			cout << "Class: " << i << ", Prob = " << probs_out[i * probLayer->height()] << endl;

		cout << "----------------------------------------------------------" << endl;
	}
	else if (!strcmp(argv[1], "prepare"))
	{
		scoped_ptr<DB> dat_db(db::GetDB("lmdb"));
		dat_db->Open(TRAININGSET_PATH_PREFIX + "data_mdb", NEW);
		scoped_ptr<Transaction> dat_txn(dat_db->NewTransaction());
		scoped_ptr<DB> test_dat_db(db::GetDB("lmdb"));
		test_dat_db->Open(TRAININGSET_PATH_PREFIX + "test_data_mdb", NEW);
		scoped_ptr<Transaction> test_dat_txn(dat_db->NewTransaction());

		scoped_ptr<DB> lab_db(db::GetDB("lmdb"));
		lab_db->Open(TRAININGSET_PATH_PREFIX + "label_mdb", NEW);
		scoped_ptr<Transaction> lab_txn(lab_db->NewTransaction());
		scoped_ptr<DB> test_lab_db(db::GetDB("lmdb"));
		test_lab_db->Open(TRAININGSET_PATH_PREFIX + "test_label_mdb", NEW);
		scoped_ptr<Transaction> test_lab_txn(lab_db->NewTransaction());

		Datum dat_datum, lab_datum;
		int count = 0;
		ifstream inputFile(TRAININGSET_PATH_PREFIX + "dataList.txt");
		ifstream inputTestFile(TRAININGSET_PATH_PREFIX + "testDataList.txt");
		string curFileName, curTestFileName;
		bool status;
		int dataCount, labelCount, width, height, isGray;
		int testDataCount, testLabelCount, testWidth, testHeight, testIsGray;
		float label;
		string key_str, out;

		inputFile >> dataCount >> labelCount >> width >> height >> isGray;
		inputFile >> testDataCount >> testLabelCount >> testWidth >> testHeight >> testIsGray;

		if (dataCount != testDataCount ||
			labelCount != testLabelCount ||
			width != testWidth ||
			height != testHeight ||
			isGray != testIsGray)
		{
			cout << "Error: Test data size and train data size are not match." << endl;
			return 0;
		}

		lab_datum.set_channels(labelCount);
		lab_datum.set_height(1);
		lab_datum.set_width(1);

		cout << "Total " << dataCount << " images" << endl
			 << "with " << labelCount << " labels." << endl;

		for (int i = 0; i < dataCount; i++)
		{
			inputFile >> curFileName;
			inputTestFile >> curTestFileName;

			status = ReadImageToDatum(TRAININGSET_PATH_PREFIX + "data/" + curFileName, -1, width, height, (bool)isGray, &dat_datum);
			if (!status)
			{
				cout << "Error: One of data files cannot be loaded." << endl << endl;
				return 0;
			}
			lab_datum.clear_data();
			lab_datum.clear_float_data();
			for (int j = 0; j < labelCount; j++)
			{
				inputFile >> label;
				lab_datum.add_float_data(label);
			}

			key_str = format_int(i, 8);
			CHECK(dat_datum.SerializeToString(&out));
			dat_txn->Put(key_str, out);
			CHECK(lab_datum.SerializeToString(&out));
			lab_txn->Put(key_str, out);

			status = ReadImageToDatum(TRAININGSET_PATH_PREFIX + "data/" + curTestFileName, -1, width, height, (bool)isGray, &dat_datum);
			if (!status)
			{
				cout << "Error: One of test data files cannot be loaded." << endl << endl;
				return 0;
			}
			lab_datum.clear_data();
			lab_datum.clear_float_data();
			for (int j = 0; j < labelCount; j++)
			{
				inputTestFile >> label;
				lab_datum.add_float_data(label);
			}

			key_str = format_int(i, 8);
			CHECK(dat_datum.SerializeToString(&out));
			test_dat_txn->Put(key_str, out);
			CHECK(lab_datum.SerializeToString(&out));
			test_lab_txn->Put(key_str, out);

			if (!(++count % 1000))
			{
				dat_txn->Commit();
				dat_txn.reset(dat_db->NewTransaction());
				lab_txn->Commit();
				lab_txn.reset(lab_db->NewTransaction());
				test_dat_txn->Commit();
				test_dat_txn.reset(test_dat_db->NewTransaction());
				test_lab_txn->Commit();
				test_lab_txn.reset(test_lab_db->NewTransaction());
				cout << "Processed " << count << " files." << endl;
			}
		}
		if (count % 1000 != 0)
		{
			dat_txn->Commit();
			lab_txn->Commit();
			test_dat_txn->Commit();
			test_lab_txn->Commit();
			cout << "Processed " << count << " files." << endl;
		}
		inputFile.close();
		inputTestFile.close();
	}
	else if (!strcmp(argv[1], "train"))
	{
		vector<int> gpus;
		getGPUs(&gpus);
		if (gpus.size() == 0)
		{
			cout << "Using CPU." << endl;
			Caffe::set_mode(Caffe::CPU);
		}
		else
		{
			cudaDeviceProp device_prop;
			for (uint i = 0; i < gpus.size(); ++i)
			{
				cudaGetDeviceProperties(&device_prop, gpus[i]);
				cout << "GPU " << gpus[i] << ": " << device_prop.name << endl;
			}
			Caffe::SetDevice(0);
			Caffe::set_mode(Caffe::GPU);
			Caffe::set_solver_count(gpus.size());
		}

		SolverParameter solver_param;
		ReadSolverParamsFromTextFileOrDie(PROTOTXT_PATH_PREFIX + "solver.prototxt", &solver_param);

		caffe::SignalHandler signal_handler(SolverAction::STOP, SolverAction::SNAPSHOT);
		caffe::shared_ptr<Solver<float>> solver(SolverRegistry<float>::CreateSolver(solver_param));
		solver->SetActionFunction(signal_handler.GetActionFunction());

		DIR *dir;
		struct dirent *ent;
		if ((dir = opendir(MODEL_PATH_PREFIX.c_str())) != NULL)
		{
			while ((ent = readdir(dir)) != NULL)
			{
				if (regex_match(ent->d_name, regex(".+\.solverstate")))
				{
					solver->Restore((PROTOTXT_PATH_PREFIX + ent->d_name).c_str());
					break;
				}
			}
			closedir(dir);
		}
		else
		{
			cout << "Error: Cannot find .caffemodel file in " + TRAININGSET_PATH_PREFIX << "." << endl << endl;
			return 0;
		}
		cout << "Starting Optimiztion" << endl;
		if (gpus.size() > 1)
		{
#ifdef USE_NCCL
			caffe::NCCL<float> nccl(solver);
			nccl.Run(gpus, FLAGS_snapshot.size() > 0 ? FLAGS_snapshot.c_str() : NULL);
#else
			cout << "Multi-GPU execution not available - rebuild with USE_NCCL" << endl;
#endif
		}
		else
			solver->Solve();
		cout << "Optimization Done." << endl;
	}
	else
		cout << "wtf?" << endl;

    return 0;
}

void getGPUs(vector<int>* gpus)
{
	int count = 0;
#ifndef CPU_ONLY
	CUDA_CHECK(cudaGetDeviceCount(&count));
#else
	NO_GPU;
#endif
	for (int i = 0; i < count; ++i)
		gpus->push_back(i);
}

void printHelp()
{

	cout << endl << endl << "Usage: helper [prepare/train/test] {test_data_file_name}" << endl
		 << "prepare:" << endl
		 << "\tMake sure that there are dataList.txt and testDataList.txt file, in nn/training_set folder which contains a list of data file name and corresponding label in the following format" << endl
		 << "\t\t[Data count]\\t[Label count]\\t[Width]\\t[Height]\\t[gray{1/0}]\\n\t// First line" << endl
		 << "\t\t[Data file name]\\t[label]\\n" << endl
		 << "\t**All data files must be in nn/training_set/data folder" << endl
		 << "\t**All label files must be in nn/training_set/label folder." << endl
		 << endl
		 << "train:" << endl
		 << "\tMake sure that sovler.prototxt and net.prototxt are in nn/prototxt folder" << endl
		 << endl
		 << "test:" << endl
		 << "\tPlease enter target file name which is supposed to be in nn folder" << endl
		 << "\tMake sure that nn/model folder contains .caffemodel file" << endl << endl << endl;
}
