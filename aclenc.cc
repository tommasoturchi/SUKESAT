/*
 * aclenc.cc
 *
 *  Created on: Jun 21, 2012
 *      Author: Tommaso Turchi
 */

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/finder.hpp>
#include <boost/algorithm/string/formatter.hpp>
#include <boost/chrono.hpp>
#include <boost/filesystem.hpp>
#include <boost/functional/hash.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <iostream>
#include <stdio.h>

#define MAXLENGTH 4096
#define MAXWORDS 256

using namespace std;

namespace bfs = boost::filesystem;

// Hash function used to encode strings
unsigned long hash(char *str) {
	unsigned long hash = 5381;
	int c;
	while (c = *str++)
		hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
	return hash;
}

int main(int argc, char const *argv[]) {
	// Positive reviews path to be read
	bfs::path ppath(argv[1]);
	// Negative reviews path to be read
	bfs::path npath(argv[2]);
	// File to be write
	FILE* output = fopen(argv[3], "w+");

	// Start measuring
	cout << "Starting... ";
	boost::chrono::system_clock::time_point start =
			boost::chrono::system_clock::now();

	if (!bfs::exists(ppath) || !bfs::exists(npath)) {
		cout << "Error" << endl;
		return 1;
	}

	int psize = 0;

	bfs::directory_iterator end_itr;
	for (bfs::directory_iterator itr(ppath); itr != end_itr; ++itr) {
		double rating =
				atoi(
						bfs::basename(itr->path().leaf()).substr(
								bfs::basename(itr->path().leaf()).find_last_of(
										'_') + 1).c_str());
		fwrite(&rating, sizeof rating, 1, output);

		ifstream curr((ppath / itr->path().leaf().string()).c_str());
		string review;
		getline(curr, review);
		curr.close();

		string body_stripped(review.c_str(), MAXLENGTH);
		review.clear();
		// Strip punctuation
		boost::find_format_all(body_stripped,
				boost::token_finder(boost::is_punct(locale())),
				boost::const_formatter(""));

		char* dup = strdup(body_stripped.c_str());
		char* saveptr;
		char* token = strtok_r(dup, " ", &saveptr);

		// Stops the tokenization when reached the end or MAXWORDS
		unsigned int z = 0;
		unsigned long* tmp = new unsigned long[MAXWORDS];
		while ((token != NULL) && (z < MAXWORDS)) {
			boost::to_lower(token);
			tmp[z++] = hash(token);
			token = strtok_r(NULL, " ", &saveptr);
		}
		free(dup);
		body_stripped.clear();

		fwrite(&z, sizeof z, 1, output);

		for (unsigned int j = 0; j < z; ++j)
			fwrite(&tmp[j], sizeof tmp[j], 1, output);

		delete[] tmp;

		++psize;

	}

	int nsize = 0;

	for (bfs::directory_iterator itr(npath); itr != end_itr; ++itr) {
		double rating =
				atoi(
						bfs::basename(itr->path().leaf()).substr(
								bfs::basename(itr->path().leaf()).find_last_of(
										'_') + 1).c_str());
		fwrite(&rating, sizeof rating, 1, output);

		ifstream curr((npath / itr->path().leaf().string()).c_str());
		string review;
		getline(curr, review);
		curr.close();

		string body_stripped(review.c_str(), MAXLENGTH);
		// Strip punctuation
		boost::find_format_all(body_stripped,
				boost::token_finder(boost::is_punct(locale())),
				boost::const_formatter(""));

		char* dup = strdup(body_stripped.c_str());
		char* saveptr;
		char* token = strtok_r(dup, " ", &saveptr);

		// Stops the tokenization when reached the end or MAXWORDS
		unsigned int z = 0;
		unsigned long* tmp = new unsigned long[MAXWORDS];
		while ((token != NULL) && (z < MAXWORDS)) {
			boost::to_lower(token);
			tmp[z++] = hash(token);
			token = strtok_r(NULL, " ", &saveptr);
		}
		free(dup);
		body_stripped.clear();

		fwrite(&z, sizeof z, 1, output);

		for (unsigned int j = 0; j < z; ++j)
			fwrite(&tmp[j], sizeof tmp[j], 1, output);

		delete[] tmp;

		++nsize;

	}

	fclose(output);

	cout << psize + nsize << " reviews successfully processed." << endl;

	// End measuring
	boost::chrono::duration<double> sec = boost::chrono::system_clock::now()
			- start;
	cout << "Elapsed time: " << sec.count() << " secs." << endl;

	return 0;
}
