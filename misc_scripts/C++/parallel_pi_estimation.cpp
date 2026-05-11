#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <thread>
#include <vector>

using namespace std;

// Function that checks if the point (x,y) is inside the unit circle at (1,1).
bool in_circle(float x, float y) {
    float distance_squared = (x - 1.0f) * (x - 1.0f) + (y - 1.0f) * (y - 1.0f);
    return distance_squared <= 1.0f;
}

// Main function to estimate pi using uniform sampling in the square [0,2] x [0,2]. Utilize parallelism to speed up the computation.
float pi_hat(int n_points) {

    int inside_count = 0;

    auto start_time = chrono::high_resolution_clock::now();

    unsigned int n_threads = thread::hardware_concurrency();

    vector<thread> threads;
    threads.reserve(n_threads);
    vector<long long> local_counts(n_threads, 0);

    long long points_per_thread = n_points / n_threads;
    long long remainder = n_points % n_threads;

    for (unsigned int i = 0; i < n_threads; i++) {
        threads.emplace_back([&, i]() {
            long long local_count = 0;
            long long points_to_generate = points_per_thread + (i < remainder ? 1 : 0);

                random_device rd;
                mt19937 gen(rd());
                uniform_real_distribution<float> uniform_0_2(0.0f, 2.0f);

             for (long long j = 0; j < points_to_generate; j++) {

                float x = uniform_0_2(gen);
                float y = uniform_0_2(gen);

                if (in_circle(x, y)) {
                    local_count++;
                }
            }
            local_counts[i] = local_count;
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    for (const auto& count : local_counts) {
        inside_count += count;
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<float> duration = end_time - start_time;

    float pi_estimate = 4.0f * static_cast<float>(inside_count) / n_points;

    cout << "-- Results -- \n";
    cout << "Samples: " << n_points << "\n";
    cout << "Fraction of points inside circle: " << static_cast<float>(inside_count) / n_points << "\n";
    cout << "Pi estimate: " << pi_estimate << "\n";
    cout << "Threads used: " << n_threads << "\n";
    cout << "Time taken: " << duration.count() << " seconds\n";

    return pi_estimate;
}

// Run with user input for number of samples.
int main() {

    double n_points;
    char again = 'y';

    while (again == 'y' || again == 'Y') {

        cout << "Enter number of samples: ";
        cin >> n_points;

        pi_hat(n_points);

        cout << "\nDo you want to run again? (y/n): ";
        cin >> again;

        cout << endl;
    }

    cout << "Program finished." << endl;

    return 0;
}
