# A perceptron learning tool that demonstrates learning on D=2 dataset
# For Homework #1 from "Learning from Data" / Professor Yaser Abu-Mostafa, Caltech
# http://work.caltech.edu/homework/hw1.pdf
# Questions 7-10 can be answered with this implementation

# In this problem, you will create your own target function f and data set D to see
# how the Perceptron Learning Algorithm works. Take d = 2 so you can visualize the
# problem, and assume X = [−1, 1] × [−1, 1] with uniform probability of picking each
# x ∈ X .
#
# In each run, choose a random line in the plane as your target function f (do this by
# taking two random, uniformly distributed points in [−1, 1] × [−1, 1] and taking the
# line passing through them), where one side of the line maps to +1 and the other maps
# to −1. Choose the inputs xn of the data set as random points (uniformly in X), and
# evaluate the target function on each xn to get the corresponding output yn.
#
# Now, in each run, use the Perceptron Learning Algorithm to find g. Start the PLA
# with the weight vector w being all zeros (consider sign(0) = 0, so all points are initially
# misclassified), and at each iteration have the algorithm choose a point randomly
# from the set of misclassified points. We are interested in two quantities: the number
# of iterations that PLA takes to converge to g, and the disagreement between f and
# g which is P[f(x) != g(x)] (the probability that f and g will disagree on their classification
# of a random point). You can either calculate this probability exactly, or
# approximate it by generating a sufficiently large, separate set of points to estimate it.
#
# In order to get a reliable estimate for these two quantities, you should repeat the
# experiment for 1000 runs (each run as specified above) and take the average over
# these runs.
#

# Date: 1/4/15
# Author: Brett Bond

class Perceptron
  attr_reader :probability_of_error
  attr_reader :iterations

  def initialize
    # The input data as sets of (x,y) coordinates
    @x = Array.new(NUM_INPUTS)

    # The boolean output results for the input data as -1 or 1
    @y = Array.new(NUM_INPUTS)

    # The weight vector, with the 3rd element as the bias (b)
    @w = [0.0, 0.0, 0.0]

    # The target function, a hash of two points in the plane
    @target = {}

    # The number of iterations this run
    @iterations = 0

    # Probability of out of sample error this run
    @probability_of_error = 0

    randomize_target
    randomize_inputs
    calculate_outputs

    if VERBOSE
      print "Target "
      pp @target

      print "X "
      print_vector(@x)

      print "Y "
      pp @y
    end
  end

  def to_point(x,y)
    {x:x, y:y}
  end

  def random_point
    {x:(2.0*Random.rand-1.0),y:(2.0*Random.rand-1.0)}
  end

  def randomize_target
    @target[:a] = random_point
    @target[:b] = random_point
  end

  def randomize_inputs
    for i in 0..NUM_INPUTS-1 do
      @x[i] = random_point
    end
  end

  def calculate_outputs
    for i in 0..NUM_INPUTS-1 do
      @y[i] = which_side_of_target @x[i]
    end
  end

  # calculate which side of the target line an input point lies
  def which_side_of_target(test_point)
    # From: http://www.gamedev.net/topic/542870-determine-which-side-of-a-line-a-point-is/
    # A and B are points forming the target line. C is the test point
    # (Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)
    ((@target[:b][:x] - @target[:a][:x]) *
        (test_point[:y]-@target[:a][:y]) -
    (@target[:b][:y] - @target[:a][:y]) *
        (test_point[:x] - @target[:a][:x]) > 0) ? 1 : -1
  end

  def h(point)
    # compute the sign of w-transpose[i] * x[i]
    ((@w[0]*point[:x] + @w[1]*point[:y] + @w[2]) > 0) ? 1 : -1
  end

  def g(point)
    # the chosen hypothesis is called g but is just another name for the h function
    h(point)
  end

  def compute_weights
    misclassified = []

    for i in 0..NUM_INPUTS-1 do
      if h(@x[i]) != @y[i]
        misclassified << i
      end
    end
    if misclassified.length == 0
      return 0
    end
    # randomly select a misclassified point
    adjust = misclassified[Random.rand(misclassified.length)]

    # adjust weight
    @w[0] += @y[adjust] * @x[adjust][:x]
    @w[1] += @y[adjust] * @x[adjust][:y]
    @w[2] += @y[adjust]
    return misclassified.length
  end

  def print_vector(v)
    for i in 0..v.length-1 do
      printf "(%.2f, %.2f) ", v[i][:x], v[i][:y]
    end
    puts
  end

  def test_learning
    for i in 0..NUM_INPUTS-1 do
      test_point = random_point
      eq = g(test_point) == which_side_of_target(test_point)
      if eq then @probability_of_error += 1 end
      printf "Testing point (%.2f, %.2f): %s\n", test_point[:x], test_point[:y], (eq) ? "✓" : "☓" if VERBOSE
    end
    @probability_of_error /= NUM_INPUTS.to_f
    printf "Out of sample correct: %.4f.\n", @probability_of_error if VERBOSE
  end

  def run
    for i in 0..NUM_ITERATIONS-1 do
      num_misclassified = compute_weights
      if num_misclassified == 0
        break
      end
      if VERBOSE
        printf "w: (%.4f, %.4f, %.4f)\n", @w[0], @w[1], @w[2]
        printf "Iteration %d: Misclassified %d points.\n", i, num_misclassified
      end
    end

    if VERBOSE
      if num_misclassified == 0
        printf "Done in %d iterations.\n", i+1
      else
        printf "Done but PLA didn't converge after %d iterations.\n", i+1
      end
    end
    test_learning
    @iterations = i
    return i
  end
end


# Main script

# Number of times to start perceptron learning from scratch and test results
RUNS = 1000

# Number of training points
NUM_INPUTS = 100

# Maximum number of times to iterate training. If perceptron does not converge within this
# number of iterations, training stops and testing begins
NUM_ITERATIONS = 1000

# Print lots of intermediate progress updates to the console
VERBOSE = false

# The average number of iterations it took for perceptron to converge over all runs
average_iterations = 0

# The average probability of error on test points after training
# (ie. how well did perceptron perform out of sample)
average_probability_of_error = 0

for i in 0..RUNS-1 do
  printf "Run %d\n", i
  perceptron = Perceptron.new
  average_iterations += perceptron.run
  average_probability_of_error += perceptron.probability_of_error
end

average_iterations /= RUNS.to_f
average_probability_of_error = 1.0 - (average_probability_of_error / RUNS.to_f)

printf "On average it takes %.2f iterations to converge.\n", average_iterations+1
printf "Average probability of error: %.4f.\n", average_probability_of_error
