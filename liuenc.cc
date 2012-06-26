/*
 * liuenc.cc
 *
 *  Created on: Jun 21, 2012
 *      Author: Tommaso Turchi
 */

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/finder.hpp>
#include <boost/algorithm/string/formatter.hpp>
#include <boost/chrono.hpp>
#include <boost/functional/hash.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <iostream>
#include <stdio.h>

#define MAXLENGTH 4096
#define MAXWORDS 256
#define MAXREVIEWS 1048576

using namespace std;

// Hash function used to encode strings
unsigned long hash(char *str) {
	unsigned long hash = 5381;
	int c;
	while (c = *str++)
		hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
	return hash;
}

int main(int argc, char const *argv[]) {
	// Amazon reviews file to be read
	ifstream input(argv[1]);
	// File to be write
	FILE* output = fopen(argv[2], "w+");

	// Start measuring
	cout << "Starting... ";
	boost::chrono::system_clock::time_point start =
			boost::chrono::system_clock::now();

	int size = 0;

	streamsize maxssize = numeric_limits<streamsize>::max();

	while (!input.eof() && size < MAXREVIEWS) {
		// Member ID
		input.ignore(maxssize, '\t');
		// Product ID
		input.ignore(maxssize, '\t');
		// Date
		input.ignore(maxssize, '\t');
		// Helpful Feedbacks
		input.ignore(maxssize, '\t');
		// Feedbacks
		input.ignore(maxssize, '\t');
		// Rating
		double rating;
		if (!(input >> rating)) {
			input.ignore(maxssize, '\n');
			continue;
		}
		fwrite(&rating, sizeof rating, 1, output);
		// Title
		input.ignore(maxssize, '\t');
		// Body
		string review;
		if (!getline(input, review))
			continue;

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

		++size;

	}

	input.close();
	fclose(output);

	cout << size << " reviews successfully processed." << endl;

	// End measuring
	boost::chrono::duration<double> sec = boost::chrono::system_clock::now()
			- start;
	cout << "Elapsed time: " << sec.count() << " secs." << endl;

	return 0;
}
