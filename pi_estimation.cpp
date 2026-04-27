#include <iostream>
#include <random>
#include <cmath>

using namespace std;

// Function that checks if the point (x,y) is inside the unit circle at (1,1).
bool in_circle(float x, float y) {
    float distance_squared = (x - 1.0f) * (x - 1.0f) + (y - 1.0f) * (y - 1.0f);
    return distance_squared <= 1.0f;
}

// Main function to estimate pi using uniform sampling in the square [0,2] x [0,2].
float pi_hat(int n_points) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> uniform_0_2(0.0f, 2.0f);

    int inside_count = 0;

    for (int i = 0; i < n_points; i++) {
        float x = uniform_0_2(gen);
        float y = uniform_0_2(gen);

        if (in_circle(x, y)) {
            inside_count++;
        }
    }

    float pi_estimate = 4.0f * static_cast<float>(inside_count) / n_points;

    cout << "-- Results -- \n";
    cout << "Samples: " << n_points << "\n";
    cout << "Fraction of points inside circle: " << static_cast<float>(inside_count) / n_points << "\n";
    cout << "Pi estimate: " << pi_estimate << "\n";

    return pi_estimate;
}

// Run with user input for number of samples.
int main() {

    int n_points;
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