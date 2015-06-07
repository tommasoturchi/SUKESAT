/*
 * sukesan.cc
 *
 *  Created on: Jun 6, 2012
 *      Author: Tommaso Turchi
 */

#include <boost/chrono.hpp>
#include <boost/progress.hpp>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <vector>

#define LAMBDA 0.8
#define LAMBDA2 0.64
#define MU 0.5
#define EPOCHS 30
#define KERNELFUNCTION ssqkernel
#define ONLYPN
/*
// Threshold for positive ratings
#define PTHOLD 5
// Threshold for negative ratings
#define NTHOLD 1
// Threshold for positive tests (used to count errors)
#define TPTHOLD 4
// Threshold for negative tests (used to count errors)
#define TNTHOLD 2
*/
// IMDB
#define PTHOLD 10
#define NTHOLD 1
#define TPTHOLD 8
#define TNTHOLD 3

#define THREADS 8

using namespace std;

// Encoded input review
struct rev {
    unsigned long* hashes;
    unsigned int hlen;
    double score;
    int label;
};

// Efficiently compute the substring kernel between s and t where substrings of exactly length n are considered
double sstkernel(rev *u, rev *v, unsigned int n) {
    unsigned int p = u->hlen, q = v->hlen;
    register unsigned int i, j, k;
    double ker, tmp;

    /* Computes the substring kernel */
    for (ker = 0.0, i = 0; i < p; i++)
        for (j = 0; j < q; j++) {
            for (k = 0, tmp = LAMBDA2;
                    (i + k < p) && (j + k < q)
                    && (u->hashes[i + k] == v->hashes[j + k]) && (k < n);
                    k++, tmp *= LAMBDA2)
                ;
            // Update features in case of full match
            if (k == n)
                ker += tmp;
        }

    /* Return the computed value */
    return (ker);
}

/* Efficiently compute the bag of word kernel between s and t where substrings of exactly length n are considered */
double bowkernel(rev *u, rev *v, unsigned int n) {
    unsigned int p = u->hlen, q = v->hlen;
    register unsigned int i, j, k;
    double ker, tmp;

    /* Computes the bag-of-words kernel */
    for (ker = 0.0, i = 0; i < p; i++) {
        for (j = 0; j < q; j++) {
            for (k = 0, tmp = LAMBDA2; (i + k < p) && (j + k < q) && (u->hashes[i + k] == v->hashes[j + k]) && (k < n); k++, tmp *= LAMBDA2);

            /* Substring detected */
            if (k == n)
                ker += tmp;

            /* Skip the whole word in v */
            for (; j < q; j++)
                ;
        }

        /* Skip the whole word in u */
        for (; i < p; i++)
            ;
    }

    /* Return the computed value */
    return (ker);
}

/* This dynamic variable saves the auxillary kernel values computed */
//static double ***cache;
double Kprime(rev *u, int p, rev *v, int q, int n, double*** cache) {
    register int j;
    double tmp;

    /* Case 1: if a full substring length is processed, return */
    if (n <= 0)
        return (1.0);

    /* Check, if the value was already computed in a previous computation */
    if (cache[n][p][q] != -1.0)
        return (cache[n][p][q]);

    /* Case 2: at least one substring is to short */
    if (p < n || q < n)
        return (0.0);

    /* Case 3: recursion */
    for (j = 0, tmp = 0; j < q; j++)
        if (v->hashes[j] == u->hashes[p - 1])
            tmp += Kprime(u, p - 1, v, j, n - 1, cache)
                   * pow(LAMBDA, (float) (q - j + 1));

    cache[n][p][q] = LAMBDA * Kprime(u, p - 1, v, q, n, cache) + tmp;
    return (cache[n][p][q]);
}

double K(rev *u, int p, rev *v, int q, int n, double*** cache) {
    register int j;
    double KP;

    /* The simple case: (at least) one string is too short */
    if (p <= n || q < n)
        return (0.0);

    /* The recursion: use Kprime for the t'th substrings */
    for (j = 0, KP = 0.0; j < q; ++j)
        if (v->hashes[j] == u->hashes[p - 1])
            KP += Kprime(u, p - 1, v, j, n - 1, cache) * LAMBDA2;

    return (K(u, p - 1, v, q, n, cache) + KP);
}

/* Recursively computes the subsequence kernel between s and t where subsequences of exactly length n are considered */
double ssqkernel(rev *u, rev *v, int n) {
    int p = u->hlen, q = v->hlen, i, j, k;
    double ker;

    /* Allocate memory for auxiallary cache variable */
    double*** cache = (double ***) malloc(n * sizeof(double **));
    for (i = 1; i < n; i++) {
        cache[i] = (double **) malloc(p * sizeof(double *));
        for (j = 0; j < p; j++) {
            cache[i][j] = (double *) malloc(q * sizeof(double));
            for (k = 0; k < q; k++)
                cache[i][j][k] = -1.0;
        }
    }

    /* Invoke recursion */
    ker = K(u, p, v, q, n, cache);

    /* Free memory */
    for (i = 1; i < n; i++) {
        for (j = 0; j < p; j++)
            free(cache[i][j]);
        free(cache[i]);
    }
    free(cache);

    /* Return the computed value */
    return (ker);
}

int main(int argc, char const *argv[]) {
    // File to be read
    FILE* input = fopen(argv[1], "r");
    // Start measuring
    cout << "Starting... ";
    boost::chrono::system_clock::time_point start =
        boost::chrono::system_clock::now();

    // Reviews array
    vector<rev> reviews(1048576);
    // Total processed reviews array size
    unsigned int size;
    //  Negative+Pegative reviews array
    vector<rev*> prevs, nrevs;
    // Negative+Positive reviews array size
    unsigned int nsize, psize;

    // Read reviews from input file
    for (size = 0, nsize = 0, psize = 0; !feof(input); ++size) {
        // Create a new review
        rev r = rev();
        // Read score
        fread(&r.score, sizeof r.score, 1, input);
        // Read body length
        fread(&r.hlen, sizeof r.hlen, 1, input);
        // Read body
        r.hashes = new unsigned long[r.hlen];
        for (unsigned int i = 0; i < r.hlen; ++i)
            fread(&r.hashes[i], sizeof r.hashes[i], 1, input);
        // Append to reviews array
        reviews[size] = r;
        // Check and store if could be chosen as a training example
        if (r.score <= NTHOLD)
            reviews[size].label = -1, nrevs.push_back(&reviews[size]), nsize++;
        else if (r.score >= PTHOLD)
            reviews[size].label = +1, prevs.push_back(&reviews[size]), psize++;
    }
    // Resize reviews arrays to fit processed ones
    reviews.resize(size), nrevs.resize(nsize), prevs.resize(psize);
    fclose(input);
    cout << "OK!" << endl;

    // Negative+Positive training example array size
    unsigned int ntsize = atoi(argv[2]), ptsize = atoi(argv[3]);
    // Training examples array size
    unsigned int tsize = ntsize + ptsize;
    // Training examples array
    vector<rev*> train(tsize);
    // Substring kernel length
    unsigned int klen = atoi(argv[4]);
    // Reviews to be tested
    unsigned int tests = atoi(argv[5]);

    for (int i = 0; i < atoi(argv[6]); ++i)
        try {
            // Output Filename
            std::stringstream outname;
            outname << ntsize << "ntrain-" << ptsize << "ptrain-" << tests
                    << "tests-" << klen << "ker." << i;
            // File to be written
            ofstream output(outname.str().c_str());

            // Shuffle random numbers
            srand(time(NULL));

            // Now extracting negative training data
            for (unsigned int i = 0; i < ntsize; ++i)
                train[i] = nrevs[fmod(rand(), nsize)];
            // Now extracting positive training data
            for (unsigned int i = ntsize; i < tsize; ++i)
                train[i] = prevs[fmod(rand(), psize)];

            // Gram Matrix
            double** gram = new double*[tsize];
            for (unsigned int i = 0; i < tsize; ++i)
                gram[i] = new double[tsize + klen];

            cout << "Now computing kernels between " << tsize
                 << " training examples...";
            boost::progress_display kernels_progress(tsize);

            // Compute the substring kernels for normalization
            for (unsigned int k = 0; k < klen; ++k)
                for (unsigned int s = 0; s < tsize; ++s)
                    gram[s][k] = KERNELFUNCTION(train[s], train[s], k);

            // Now compute the substring kernels
            #pragma omp parallel for if(THREADS>1) num_threads(THREADS) shared(gram,kernels_progress,klen,train,tsize) schedule(static)
            for (unsigned int s = 0; s < tsize; ++s) {

                /* Computes the substring kernel K(s,t) */
                for (unsigned int t = 0; t < tsize; ++t) {

                    if (t == s) {
                        gram[s][klen + t] = 1;
                        continue;
                    }

                    gram[s][klen + t] = 0;

                    for (unsigned int k = 0; k < klen; ++k) {
                        gram[s][klen + t] += pow(MU, 1 - k)
                                             * KERNELFUNCTION(train[s], train[t], k)
                                             / sqrt(gram[s][k] * gram[t][k]);

                    }
                }
                ++kernels_progress;
            }

            cout << "Now training neural network... ";

            // Now it's time to train our network
            unsigned int m(0);
            vector<int> u(tsize * EPOCHS), c(tsize * EPOCHS);

            for (unsigned int t = 0; t < EPOCHS; ++t)
                for (unsigned int i = 0; i < tsize; ++i) {
                    double pred = 0.0;
                    for (unsigned int j = 0; j < m; ++j)
                        pred += train[u[j]]->label * gram[u[j]][i + klen];
                    if (pred * train[i]->label > 0)
                        c[m]++;
                    else
                        u[m + 1] = i, c[m + 1] = 1, m++;
                }
            u.resize(m), c.resize(m);

            cout << "Done!" << endl << "Testing " << tests << " reviews...";

            boost::progress_display testing_progress(tests);

            int posize(0), poserr(0), posum(0), negsize(0), negerr(0), negsum(
                0);

#ifdef ONLYPN
            bool pos = true;
            bool neg = false;
#endif

            // Now test the results
            #pragma omp parallel for if(THREADS>1) num_threads(THREADS) shared(c,gram,testing_progress,klen,m,reviews,train,tsize,u) firstprivate(pos,neg) schedule(static)
            for (unsigned int s = 0; s < tests; ++s) {

                double* skernels = new double[klen + tsize];

                unsigned int current = rand() % size;

#ifdef ONLYPN
                while (((reviews[current].score < TPTHOLD) && pos) || ((reviews[current].score > TNTHOLD) && neg)) current = rand() % size;
                pos ^= true, neg ^= true;
#endif

                for (unsigned int k = 0; k < klen; ++k)
                    skernels[k] = KERNELFUNCTION(&reviews[current],
                                                 &reviews[current], k);

                for (unsigned int t = 0; t < tsize; ++t) {

                    skernels[klen + t] = 0;

                    for (unsigned int k = 0; k < klen; ++k)
                        skernels[klen + t] += pow(MU, 1 - k)
                                              * KERNELFUNCTION(&reviews[current], train[t], k)
                                              / sqrt(skernels[k] * gram[t][k]);

                }

                double predicted(0);
                for (unsigned int i = 0; i < m; ++i) {
                    double curprediction(0);
                    for (unsigned int j = 0; j < i; ++j)
                        curprediction += train[u[j]]->label
                                         * skernels[klen + u[j]];
                    predicted += c[i] * curprediction;
                }

                #pragma omp critical
                {

                    output << reviews[current].score << " " << predicted
                           << endl;

                    // Count #errors
                    if (reviews[current].score <= TNTHOLD) {
                        negsize++;
                        negsum += predicted;
                        if (predicted > 0)
                            negerr++;
                    } else if (reviews[current].score >= TPTHOLD) {
                        posize++;
                        posum += predicted;
                        if (predicted < 0)
                            poserr++;
                    }

                    ++testing_progress;
                }

            }

            output.close();

            // Advertise errors
            if (posize > 0)
                cout << "+reviews mean: " << posum / posize << " - " << poserr
                     << " errors / " << posize << " total ("
                     << poserr * 100 / posize << "%)" << endl;
            if (negsize > 0)
                cout << "-reviews mean: " << negsum / negsize << " - " << negerr
                     << " errors / " << negsize << " total ("
                     << negerr * 100 / negsize << "%)" << endl;

            if ((posize > 0) && ((posize - poserr + negerr) > 0))
                cout << "Precision: " << (double)(posize - poserr)/(posize - poserr + negerr) << " / Recall: " << (double)(posize - poserr)/posize << endl;
        }
        catch (...) {
            cout << endl << "An error occurred, moving on..." << endl;
        }

    boost::chrono::duration<double> sec = boost::chrono::system_clock::now()
                                          - start;
    cout << "Elapsed time: " << sec.count() << " secs." << endl;
    return 0;
}
