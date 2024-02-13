from numpy import *
import matplotlib.pyplot as pt


def compute_error_for_line_given_points(b, m, points):
    total_error = 0
    # for every point in points
    for i in range(0, len(points)):
        # get x-value
        x = points[i, 0]
        # get y-value
        y = points[i, 1]
        # get difference and add to total error
        # summing the diffence in height between every point and the line
        # squared because total_error remains positive
        total_error += (y - (m * x + b)) ** 2

        # returns the average of total_error
        return total_error / float(len(points))


def gradient_descent(points, starting_b, starting_m, learning_rate, num_iterations):
    for i in range(num_iterations):
        # update b and m with more values closer to line of bast fit
        b, m = step_gradiant(starting_b, starting_m, points, learning_rate) # check if not array, still works
    return [b, m]


def step_gradiant(b_current, m_current, points, learning_rate):
    # starting points for gradients
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # direction with respects to b and m. computing partial derivative of error function
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))

    # update our b and m valus using partial derivatives
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def run():
    points_data = genfromtxt("data.csv", delimiter=",")  # collects data from csv file
    # print(points_data)
    x = []
    y = []
    for i in range(0, len(points_data)):
        x.append(points_data[i, 0])
        y.append(points_data[i, 1])
    pt.plot(array(x), array(y), "o")


    # STEP 2 - DEFINING OUR HYPERPARAMETERS
    # how fast should out model converge
    learning_rate = 0.0001

    # y = mx+b (slope formula)
    initial_m = 0
    initial_b = 0
    num_iterations = 1000

    # STEP 3- TRAIN OUR MODEL
    print("starting gradiant descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m,
                                                                              compute_error_for_line_given_points
                                                                              (initial_b, initial_m, points_data)))
    [b, m] = gradient_descent(points_data, initial_b, initial_m, learning_rate, num_iterations)

    print("ending point at b = {1}, m = {2}, error = {3}".format(num_iterations, b, m,
                                                                 compute_error_for_line_given_points
                                                                 (b, m, points_data)))
    x_val = array(x)
    y_val = m * x_val + b
    pt.plot(x_val, y_val, "--")
    pt.show()

if __name__ == '__main__':
    run()
