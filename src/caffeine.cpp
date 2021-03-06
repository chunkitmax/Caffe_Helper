/*
 * caffeine.cpp
 *
 *  Created on: Apr 1, 2017
 *      Author: petel__
 */

#include <vector>
#include <iostream>
#include <dirent.h>
#include <regex>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>
#include <stdlib.h>

#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/solver.hpp"
#include "caffe/sgd_solvers.hpp"
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
void printDataHelp();
void printImageHelp();

const string MODEL_PATH_PREFIX = "nn/model/";
const string PROTOTXT_PATH_PREFIX = "nn/prototxt/";
const string TRAININGSET_PATH_PREFIX = "nn/training_set/";
const string TEST_DATA_PATH_PREFIX = "nn/";

//const string MODEL_PATH_PREFIX = "../nn/";
//const string PROTOTXT_PATH_PREFIX = "../nn/prototxt/";
//const string TRAININGSET_PATH_PREFIX = "../nn/training_set/";
//const string TEST_DATA_PATH_PREFIX = "../nn/";

int main(int argc, char **argv)
{
//	cout << "Hello World!" << endl;
//	caffe::GlobalInit(&argc, &argv);

	if (argc < 3)
	{
		printHelp();
		return 0;
	}

	if (!strcmp(argv[1], "help") || !strcmp(argv[2], "help"))
	{
		if (argc > 2)
		{
			if (!strcmp(argv[1], "image"))
				printImageHelp();
			else if (!strcmp(argv[1], "data"))
				printDataHelp();
			else
				printHelp();
		}
		else
			printHelp();
		return 0;
	}
	else if (!strcmp(argv[1], "image"))
	{
		if (!strcmp(argv[2], "test"))
		{
			Caffe::SetDevice(0);
			Caffe::set_mode(Caffe::GPU);

			float loss = 0.0;
			Net<float> net(PROTOTXT_PATH_PREFIX + "test.prototxt", Phase::TEST);
			DIR *dir;
			struct dirent *ent;
			if ((dir = opendir(MODEL_PATH_PREFIX.c_str())) != NULL)
			{
				while ((ent = readdir(dir)) != NULL)
				{
					if (regex_match(ent->d_name, regex("[^\\.]+\\.caffemodel")))
					{
						net.CopyTrainedLayersFrom(MODEL_PATH_PREFIX + ent->d_name);
						cout << "Finish loading caffe model: " << (MODEL_PATH_PREFIX + ent->d_name) << endl;
						break;
					}
				}
				closedir(dir);
				if (ent == NULL)
				{
					printImageHelp();
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

			Datum datum;
			if (argv[2] == NULL)
			{
				printImageHelp();
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

			cout << (TEST_DATA_PATH_PREFIX + argv[3]) << " " << width << " " << height << " " << (bool)isGray << endl;
			bool status = ReadImageToDatum(TEST_DATA_PATH_PREFIX + argv[3], -1, width, height, &datum);
			if (!status)
			{
				cout << "Error: Test files cannot be loaded." << endl << endl;
				printImageHelp();
				return 0;
			}
			MemoryDataLayer<float> *memoryDataLayer = (MemoryDataLayer<float> *)net.layer_by_name("data").get();
			vector<Datum> tmp(1);
			tmp[0] = datum;
			memoryDataLayer->AddDatumVector(tmp);
			const vector<Blob<float> *> &result = net.Forward(&loss);

			const boost::shared_ptr<Blob<float>> &probLayer = net.blobs().back();

			cout << "----------------------------------------------------------" << endl;
			const float *probs_out = probLayer->cpu_data();
			for (uint i = 0; i < probLayer->channels(); i++)
				cout << "Class: " << i << ", Prob = " << probs_out[i * probLayer->height()] << endl;

			cout << "----------------------------------------------------------" << endl;
		}
		else if (!strcmp(argv[2], "prepare"))
		{
			scoped_ptr<DB> dat_db(db::GetDB("lmdb"));
			dat_db->Open(TRAININGSET_PATH_PREFIX + "data_mdb", NEW);
			scoped_ptr<Transaction> dat_txn(dat_db->NewTransaction());
			scoped_ptr<DB> test_dat_db(db::GetDB("lmdb"));
			test_dat_db->Open(TRAININGSET_PATH_PREFIX + "test_data_mdb", NEW);
			scoped_ptr<Transaction> test_dat_txn(test_dat_db->NewTransaction());

			scoped_ptr<DB> lab_db(db::GetDB("lmdb"));
			lab_db->Open(TRAININGSET_PATH_PREFIX + "label_mdb", NEW);
			scoped_ptr<Transaction> lab_txn(lab_db->NewTransaction());
			scoped_ptr<DB> test_lab_db(db::GetDB("lmdb"));
			test_lab_db->Open(TRAININGSET_PATH_PREFIX + "test_label_mdb", NEW);
			scoped_ptr<Transaction> test_lab_txn(test_lab_db->NewTransaction());

			Datum dat_datum, lab_datum;
			int count = 0;
			ifstream inputFile(TRAININGSET_PATH_PREFIX + "dataList.txt");
			ifstream inputTestFile(TRAININGSET_PATH_PREFIX + "testDataList.txt");
			string curFileName;
			bool status;
			int dataCount, labelCount, width, height, isGray;
			int testDataCount, testLabelCount, testWidth, testHeight, testIsGray;
			vector<float> tmpVector;
			float label;
			string key_str, out;

			vector<pair<string, vector<float>> > lines;
			vector<pair<string, vector<float>> > testLines;

			inputFile >> dataCount >> labelCount >> width >> height >> isGray;
			inputTestFile >> testDataCount >> testLabelCount >> testWidth >> testHeight >> testIsGray;

	//		if (dataCount != testDataCount ||
	//			labelCount != testLabelCount ||
	//			width != testWidth ||
	//			height != testHeight ||
	//			isGray != testIsGray)
	//		{
	//			cout << "Error: Test data size and train data size are not match." << endl;
	//			return 0;
	//		}

			cout << "Total " << dataCount << " images" << endl
				 << "with " << labelCount << " labels." << endl;

			for (int i = 0; i < dataCount; i++)
			{
				tmpVector.clear();
				inputFile >> curFileName;
				for (int j = 0; j < labelCount; j++)
				{
					inputFile >> label;
					tmpVector.push_back(label);
				}
				lines.push_back(make_pair(curFileName, tmpVector));
			}
			for (int i = 0; i < testDataCount; i++)
			{
				tmpVector.clear();
				inputTestFile >> curFileName;
				for (int j = 0; j < testLabelCount; j++)
				{
					inputTestFile >> label;
					tmpVector.push_back(label);
				}
				testLines.push_back(make_pair(curFileName, tmpVector));
			}

			random_shuffle(lines.begin(), lines.end());
			random_shuffle(testLines.begin(), testLines.end());

			count = 0;
			lab_datum.set_channels(labelCount);
			lab_datum.set_height(1);
			lab_datum.set_width(1);
			for (uint32_t lineId = 0; lineId < lines.size(); lineId++)
			{
				dat_datum.clear_data();
				lab_datum.clear_float_data();
				status = ReadImageToDatum(TRAININGSET_PATH_PREFIX + "data/" + lines[lineId].first, -1, width, height, !((bool)isGray), &dat_datum);
				if (!status)
				{
					cout << "Error: One of data files cannot be loaded." << endl << endl;
					return 0;
				}
				for (uint32_t labelId = 0; labelId < lines[lineId].second.size(); labelId++)
					lab_datum.add_float_data(lines[lineId].second[labelId]);

				key_str = format_int(lineId, 8);
				CHECK(dat_datum.SerializeToString(&out));
				dat_txn->Put(key_str, out);
				CHECK(lab_datum.SerializeToString(&out));
				lab_txn->Put(key_str, out);

				if (!(++count % 1000))
				{
					dat_txn->Commit();
					dat_txn.reset(dat_db->NewTransaction());
					lab_txn->Commit();
					lab_txn.reset(lab_db->NewTransaction());
					cout << "Processed " << count << " data files." << endl;
				}
			}
			if (count % 1000 != 0)
			{
				dat_txn->Commit();
				lab_txn->Commit();
				cout << "Total " << count << " data files." << endl;
			}

			count = 0;
			lab_datum.set_channels(labelCount);
			lab_datum.set_height(1);
			lab_datum.set_width(1);
			for (uint32_t lineId = 0; lineId < testLines.size(); lineId++)
			{
				dat_datum.clear_data();
				lab_datum.clear_float_data();
				status = ReadImageToDatum(TRAININGSET_PATH_PREFIX + "test_data/" + testLines[lineId].first, -1, width, height, !((bool)testIsGray), &dat_datum);
				if (!status)
				{
					cout << "Error: One of data files cannot be loaded. (File name: " << testLines[lineId].first << ")" << endl << endl;
					return 0;
				}
				for (uint32_t labelId = 0; labelId < testLines[lineId].second.size(); labelId++)
					lab_datum.add_float_data(testLines[lineId].second[labelId]);

				key_str = format_int(lineId, 8);
				CHECK(dat_datum.SerializeToString(&out));
				test_dat_txn->Put(key_str, out);
				CHECK(lab_datum.SerializeToString(&out));
				test_lab_txn->Put(key_str, out);

				if (!(++count % 1000))
				{
					test_dat_txn->Commit();
					test_dat_txn.reset(test_dat_db->NewTransaction());
					test_lab_txn->Commit();
					test_lab_txn.reset(test_lab_db->NewTransaction());
					cout << "Processed " << count << " test data files." << endl;
				}
			}
			if (count % 1000 != 0)
			{
				test_dat_txn->Commit();
				test_lab_txn->Commit();
				cout << "Total " << count << " test data files." << endl;
			}

			dat_db->Close();
			lab_db->Close();
			test_dat_db->Close();
			test_lab_db->Close();
			inputFile.close();
			inputTestFile.close();
		}
		else if (!strcmp(argv[2], "train"))
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
			caffe::Solver<float>* solver = new caffe::SGDSolver<float>(solver_param);
			solver->SetActionFunction(signal_handler.GetActionFunction());

			DIR *dir;
			struct dirent *ent;
			if ((dir = opendir(MODEL_PATH_PREFIX.c_str())) != NULL)
			{
				while ((ent = readdir(dir)) != NULL)
				{
					if (regex_match(ent->d_name, regex(".+\.solverstate")))
					{
						solver->Restore((MODEL_PATH_PREFIX + ent->d_name).c_str());
						break;
					}
				}
				closedir(dir);
			}
			else
			{
				cout << "Error: Cannot find nn/model folder in " + TRAININGSET_PATH_PREFIX << "." << endl << endl;
				printImageHelp();
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
	}
	else if (!strcmp(argv[1], "data"))
	{
		if (!strcmp(argv[2], "test"))
		{
			Caffe::SetDevice(0);
			Caffe::set_mode(Caffe::GPU);

			float loss = 0.0;
			Net<float> net(PROTOTXT_PATH_PREFIX + "test.prototxt", Phase::TEST);
			DIR *dir;
			struct dirent *ent;
			if ((dir = opendir(MODEL_PATH_PREFIX.c_str())) != NULL)
			{
				while ((ent = readdir(dir)) != NULL)
				{
					if (regex_match(ent->d_name, regex("[^\\.]+\\.caffemodel")))
					{
						net.CopyTrainedLayersFrom(MODEL_PATH_PREFIX + ent->d_name);
						cout << "Finish loading caffe model: " << (MODEL_PATH_PREFIX + ent->d_name) << endl;
						break;
					}
				}
				closedir(dir);
				if (ent == NULL)
				{
					printDataHelp();
					return 0;
				}
			}
			else
			{
				cout << "Error: Cannot find .caffemodel file in " + TRAININGSET_PATH_PREFIX << "." << endl << endl;
				printDataHelp();
				return 0;
			}

			ifstream inputFile(TRAININGSET_PATH_PREFIX + "dataList.txt");
			int dataCount, inputCount, labelCount;
			inputFile >> dataCount >> inputCount >> labelCount;
			inputFile.close();

			Datum datum;
			if (argv[3] == NULL)
			{
				printDataHelp();
				return 0;
			}

			datum.set_channels(inputCount);
			datum.set_height(1);
			datum.set_width(1);
			cout << (TEST_DATA_PATH_PREFIX + argv[3]) << inputCount << endl;
			ifstream targetFile(TEST_DATA_PATH_PREFIX + argv[3]);
			float tmpData;
			for (int i = 0; i < inputCount; i++)
			{
				targetFile >> tmpData;
				datum.add_float_data(tmpData);
			}
			targetFile.close();
//			bool status = ReadImageToDatum(TEST_DATA_PATH_PREFIX + argv[2], -1, width, height, &datum);
			MemoryDataLayer<float> *memoryDataLayer = (MemoryDataLayer<float> *)net.layer_by_name("data").get();
			vector<Datum> tmp(1);
			tmp[0] = datum;
			memoryDataLayer->AddDatumVector(tmp);
			const vector<Blob<float> *> &result = net.Forward(&loss);

			const boost::shared_ptr<Blob<float>> &probLayer = net.blobs().back();

			cout << "----------------------------------------------------------" << endl;
			const float *probs_out = probLayer->cpu_data();
			for (uint i = 0; i < probLayer->channels(); i++)
				cout << "Class: " << i << ", Prob = " << probs_out[i * probLayer->height()] << endl;

			cout << "----------------------------------------------------------" << endl;
		}
		else if (!strcmp(argv[2], "prepare"))
		{
			scoped_ptr<DB> dat_db(db::GetDB("lmdb"));
			dat_db->Open(TRAININGSET_PATH_PREFIX + "data_mdb", NEW);
			scoped_ptr<Transaction> dat_txn(dat_db->NewTransaction());
			scoped_ptr<DB> test_dat_db(db::GetDB("lmdb"));
			test_dat_db->Open(TRAININGSET_PATH_PREFIX + "test_data_mdb", NEW);
			scoped_ptr<Transaction> test_dat_txn(test_dat_db->NewTransaction());

			scoped_ptr<DB> lab_db(db::GetDB("lmdb"));
			lab_db->Open(TRAININGSET_PATH_PREFIX + "label_mdb", NEW);
			scoped_ptr<Transaction> lab_txn(lab_db->NewTransaction());
			scoped_ptr<DB> test_lab_db(db::GetDB("lmdb"));
			test_lab_db->Open(TRAININGSET_PATH_PREFIX + "test_label_mdb", NEW);
			scoped_ptr<Transaction> test_lab_txn(test_lab_db->NewTransaction());

			Datum dat_datum, lab_datum;
			int count = 0;
			ifstream inputFile(TRAININGSET_PATH_PREFIX + "dataList.txt");
			ifstream inputTestFile(TRAININGSET_PATH_PREFIX + "testDataList.txt");
			float inputData;
			vector<float> inputVector, testInputVector;
			vector<float> labelVector, testLabelVector;
			int dataCount, inputCount, labelCount;
			int testDataCount, testInputCount, testLabelCount;
			float label;
			string key_str, out;

			vector<pair<vector<float>, vector<float>> > lines;
			vector<pair<vector<float>, vector<float>> > testLines;

			inputFile >> dataCount >> inputCount >> labelCount;
			inputTestFile >> testDataCount >> testInputCount >> testLabelCount;

			cout << "Total " << dataCount << " images" << endl
				 << "with " << labelCount << " labels." << endl;

			for (int i = 0; i < dataCount; i++)
			{
				inputVector.clear();
				labelVector.clear();
				for (int k = 0; k < inputCount; k++)
				{
					inputFile >> inputData;
					inputVector.push_back(inputData);
				}
				for (int j = 0; j < labelCount; j++)
				{
					inputFile >> label;
					labelVector.push_back(label);
				}
				lines.push_back(make_pair(inputVector, labelVector));
			}
			for (int i = 0; i < testDataCount; i++)
			{
				testInputVector.clear();
				testLabelVector.clear();
				for (int k = 0; k < testInputCount; k++)
				{
					inputTestFile >> inputData;
					testInputVector.push_back(inputData);
				}
				for (int j = 0; j < testLabelCount; j++)
				{
					inputTestFile >> label;
					testLabelVector.push_back(label);
				}
				testLines.push_back(make_pair(testInputVector, testLabelVector));
			}

			random_shuffle(lines.begin(), lines.end());
			random_shuffle(testLines.begin(), testLines.end());

			count = 0;
			lab_datum.set_channels(labelCount);
			lab_datum.set_height(1);
			lab_datum.set_width(1);
			dat_datum.set_channels(inputCount);
			dat_datum.set_height(1);
			dat_datum.set_width(1);
			for (uint32_t lineId = 0; lineId < lines.size(); lineId++)
			{
				dat_datum.clear_float_data();
				lab_datum.clear_float_data();
				for (uint32_t dataId = 0; dataId < lines[lineId].first.size(); dataId++)
					dat_datum.add_float_data(lines[lineId].first[dataId]);
				for (uint32_t labelId = 0; labelId < lines[lineId].second.size(); labelId++)
					lab_datum.add_float_data(lines[lineId].second[labelId]);

//				dat_datum.set_label(lines[lineId].second[0]);

				key_str = format_int(lineId, 8);
				CHECK(dat_datum.SerializeToString(&out));
				dat_txn->Put(key_str, out);
				CHECK(lab_datum.SerializeToString(&out));
				lab_txn->Put(key_str, out);

				if (!(++count % 1000))
				{
					dat_txn->Commit();
					dat_txn.reset(dat_db->NewTransaction());
					lab_txn->Commit();
					lab_txn.reset(lab_db->NewTransaction());
					cout << "Processed " << count << " data files." << endl;
				}
			}
			if (count % 1000 != 0)
			{
				dat_txn->Commit();
				lab_txn->Commit();
				cout << "Total " << count << " data files." << endl;
			}

			count = 0;
			lab_datum.set_channels(testLabelCount);
			lab_datum.set_height(1);
			lab_datum.set_width(1);
			dat_datum.set_channels(testInputCount);
			dat_datum.set_height(1);
			dat_datum.set_width(1);
			for (uint32_t lineId = 0; lineId < testLines.size(); lineId++)
			{
				dat_datum.clear_float_data();
				lab_datum.clear_float_data();
				for (uint32_t dataId = 0; dataId < testLines[lineId].first.size(); dataId++)
					dat_datum.add_float_data(testLines[lineId].first[dataId]);
				for (uint32_t labelId = 0; labelId < testLines[lineId].second.size(); labelId++)
					lab_datum.add_float_data(testLines[lineId].second[labelId]);

//				dat_datum.set_label(lines[lineId].second[0]);

				key_str = format_int(lineId, 8);
				CHECK(dat_datum.SerializeToString(&out));
				test_dat_txn->Put(key_str, out);
				CHECK(lab_datum.SerializeToString(&out));
				test_lab_txn->Put(key_str, out);

				if (!(++count % 1000))
				{
					test_dat_txn->Commit();
					test_dat_txn.reset(test_dat_db->NewTransaction());
					test_lab_txn->Commit();
					test_lab_txn.reset(test_lab_db->NewTransaction());
					cout << "Processed " << count << " test data files." << endl;
				}
			}
			if (count % 1000 != 0)
			{
				test_dat_txn->Commit();
				test_lab_txn->Commit();
				cout << "Total " << count << " test data files." << endl;
			}

			dat_db->Close();
			lab_db->Close();
			test_dat_db->Close();
			test_lab_db->Close();
			inputFile.close();
			inputTestFile.close();
		}
		else if (!strcmp(argv[2], "train"))
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
			caffe::Solver<float>* solver = new caffe::SGDSolver<float>(solver_param);
			solver->SetActionFunction(signal_handler.GetActionFunction());

			DIR *dir;
			struct dirent *ent;
			if ((dir = opendir(MODEL_PATH_PREFIX.c_str())) != NULL)
			{
				while ((ent = readdir(dir)) != NULL)
				{
					if (regex_match(ent->d_name, regex(".+\.solverstate")))
					{
						solver->Restore((MODEL_PATH_PREFIX + ent->d_name).c_str());
						break;
					}
				}
				closedir(dir);
			}
			else
			{
				cout << "Error: Cannot find nn/model folder in " + TRAININGSET_PATH_PREFIX << "." << endl << endl;
				printDataHelp();
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
	ostringstream oStr;
	oStr << endl << endl << "Usage: caffeine [image/data] [prepare/train/test/help] {test_data_file_name}" << endl
		 << endl
		 << "For more information, enter 'caffeine [image/data] help'" << endl
		 << endl
		 << "." << endl
		 << "└── nn" << endl
		 << "    ├── model" << endl
		 << "    │   ├── ?.caffemodel" << endl
		 << "    │   └── ?.solverstate" << endl
		 << "    ├── prototxt" << endl
		 << "    │   ├── mean.binaryproto" << endl
		 << "    │   ├── train.prototxt" << endl
		 << "    │   ├── solver.prototxt" << endl
		 << "    │   └── test.prototxt" << endl
		 << "    └── training_set" << endl
		 << "        ├── data" << endl
		 << "        │   └── ?.jpg ..." << endl
		 << "        ├── dataList.txt" << endl
		 << "        ├── data_mdb" << endl
		 << "        │   ├── data.mdb" << endl
		 << "        │   └── lock.mdb" << endl
		 << "        ├── label_mdb" << endl
		 << "        │   ├── data.mdb" << endl
		 << "        │   └── lock.mdb" << endl
		 << "        ├── test_data" << endl
		 << "        │   └── ?.jpg ..." << endl
		 << "        ├── dataList.txt" << endl
		 << "        ├── test_data_mdb" << endl
		 << "        │   ├── data.mdb" << endl
		 << "        │   └── lock.mdb" << endl
		 << "        └── test_label_mdb" << endl
		 << "            ├── data.mdb" << endl
		 << "            └── lock.mdb" << endl
		 << endl
		 << endl;
	std::system(("echo \"" + oStr.str() + "\" | more").c_str());
}
void printImageHelp()
{
	ostringstream oStr;
	oStr << endl << endl << "Usage: caffeine image/data [prepare/train/test/help] {test_data_file_name}" << endl
		 << "prepare:" << endl
		 << "\tMake sure that there are dataList.txt and testDataList.txt file, in nn/training_set folder" << endl
		 << "\twhich contains a list of data file name and corresponding labels in the following format:" << endl
		 << endl
		 << "\tLine 1: [Line count]\\t[Label count]\\t[Width]\\t[Height]\\t[gray{1/0}]\\n\t" << endl
		 << "\tLine ?: [Data file name]\\t[label]\\n" << endl
		 << "\t**All data files must be in nn/training_set/data folder" << endl
		 << endl
		 << "train:" << endl
		 << "\tMake sure that sovler.prototxt and train.prototxt are in nn/prototxt folder" << endl
		 << endl
		 << "test:" << endl
		 << "\tPlease enter target file name which is supposed to be in nn folder" << endl
		 << "\tMake sure that test.prototxt is placed in nn/prototxt folder" << endl
		 << "\tMake sure that nn/model folder contains .caffemodel file" << endl << endl
		 << "." << endl
		 << "└── nn" << endl
		 << "    ├── model" << endl
		 << "    │   ├── ?.caffemodel" << endl
		 << "    │   └── ?.solverstate" << endl
		 << "    ├── prototxt" << endl
		 << "    │   ├── mean.binaryproto" << endl
		 << "    │   ├── train.prototxt" << endl
		 << "    │   ├── solver.prototxt" << endl
		 << "    │   └── test.prototxt" << endl
		 << "    └── training_set" << endl
		 << "        ├── data" << endl
		 << "        │   └── ?.jpg ..." << endl
		 << "        ├── dataList.txt" << endl
		 << "        ├── data_mdb" << endl
		 << "        │   ├── data.mdb" << endl
		 << "        │   └── lock.mdb" << endl
		 << "        ├── label_mdb" << endl
		 << "        │   ├── data.mdb" << endl
		 << "        │   └── lock.mdb" << endl
		 << "        ├── test_data" << endl
		 << "        │   └── ?.jpg ..." << endl
		 << "        ├── dataList.txt" << endl
		 << "        ├── test_data_mdb" << endl
		 << "        │   ├── data.mdb" << endl
		 << "        │   └── lock.mdb" << endl
		 << "        └── test_label_mdb" << endl
		 << "            ├── data.mdb" << endl
		 << "            └── lock.mdb" << endl
		 << endl
		 << endl;
	std::system(("echo \"" + oStr.str() + "\" | more").c_str());
}
void printDataHelp()
{
	ostringstream oStr;
	oStr << endl << endl << "Usage: caffeine data [prepare/train/test/help] {test_data_file_name}" << endl
		 << "prepare:" << endl
		 << "\tMake sure that there are dataList.txt and testDataList.txt file, in nn/training_set folder" << endl
		 << "\twhich contains a list of input data and corresponding labels in the following format:" << endl
		 << endl
		 << "\tLine 1: [Line count]\\t[Input Data count]\\t[Label count]\\t\\n\t" << endl
		 << "\tLine ?: [Data]\\t[Data]\\t...\\t\\t[label]\\t[label]\\t...\\t\\n" << endl
		 << "\t**All data files must be in nn/training_set/data folder" << endl
//		 << "\t**All label files must be in nn/training_set/label folder." << endl
		 << endl
		 << "train:" << endl
		 << "\tMake sure that sovler.prototxt and train.prototxt are in nn/prototxt folder" << endl
		 << endl
		 << "test:" << endl
		 << "\tPlease enter target txt file name which is supposed to be in nn folder" << endl
		 << "\tMake sure that test.prototxt is placed in nn/prototxt folder" << endl
		 << "\tMake sure that nn/model folder contains .caffemodel file" << endl << endl
		 << "." << endl
		 << "└── nn" << endl
		 << "    ├── model" << endl
		 << "    │   ├── ?.caffemodel" << endl
		 << "    │   └── ?.solverstate" << endl
		 << "    ├── prototxt" << endl
		 << "    │   ├── mean.binaryproto" << endl
		 << "    │   ├── train.prototxt" << endl
		 << "    │   ├── solver.prototxt" << endl
		 << "    │   └── test.prototxt" << endl
		 << "    └── training_set" << endl
		 << "        ├── data" << endl
		 << "        │   └── ?.jpg ..." << endl
		 << "        ├── dataList.txt" << endl
		 << "        ├── data_mdb" << endl
		 << "        │   ├── data.mdb" << endl
		 << "        │   └── lock.mdb" << endl
		 << "        ├── label_mdb" << endl
		 << "        │   ├── data.mdb" << endl
		 << "        │   └── lock.mdb" << endl
		 << "        ├── test_data" << endl
		 << "        │   └── ?.jpg ..." << endl
		 << "        ├── dataList.txt" << endl
		 << "        ├── test_data_mdb" << endl
		 << "        │   ├── data.mdb" << endl
		 << "        │   └── lock.mdb" << endl
		 << "        └── test_label_mdb" << endl
		 << "            ├── data.mdb" << endl
		 << "            └── lock.mdb" << endl
		 << endl
		 << endl;
	std::system(("echo \"" + oStr.str() + "\" | more").c_str());
}
