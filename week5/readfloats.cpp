#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

using namespace std;

vector<float> readFloats(const string &fname)
{
	vector<float> data;
	ifstream f(fname, std::ios::binary);

	f.seekg(0, ios_base::end);
	int pos = f.tellg();
	f.seekg(0, ios_base::beg);
	if (pos <= 0)
		throw runtime_error("Can't seek in file " + fname + " or file has zero length");

	if (pos % sizeof(float) != 0)
		throw runtime_error("File " + fname + " doesn't contain an integer number of float32 values");

	int num = pos/sizeof(float);
	data.resize(num);

	f.read(reinterpret_cast<char*>(data.data()), pos);
	if (f.gcount() != pos)
		throw runtime_error("Incomplete read: " + to_string(f.gcount()) + " vs " + to_string(pos));
	return data;
}

