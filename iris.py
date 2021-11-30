import numpy as np
from sklearn.datasets import load_iris
import sys
import math
np.set_printoptions(threshold=sys.maxsize)

#load iris dataset and extract the columns for petal length, petal width, and the classifications
iris = load_iris()
length = iris.data[:, 2]
width = iris.data[:, 3]
output = iris.target

aggregation_type = 0 	#Set to 0 for max aggregation or set to 1 for sum aggregation			

def trapezoidal_membership_function(length, width, output, aggregation_type):
	length_mem_values = np.zeros((150, 3))		#array of membership values for small, medium, large for length
	width_mem_values = np.zeros((150, 3))		#array of membership values for small, medium, large for length

	#output membership functions for the 3 iris classes
	setosa_mf 	  = np.array([0,1,0,0,0,0,0])
	versicolor_mf = np.array([0,0,0,1,0,0,0])
	virginica_mf  = np.array([0,0,0,0,0,1,0])

	aggregation_vals = np.zeros((150, 7))
	results = np.zeros(150)
	counter = 0
	accuracy = 0

	#for each entry in the iris data set
	for i in range(len(length)):
		centroid_numerator = 0.0
		centroid_denominator = 0.0
		centroid = 0.0

		#define membership functions for petal length and width
		length_mem_values[i][0] = max(min(((length[i]-0)/(0.1-0)), 1, (2.75-length[i])/(2.75-2.25)), 0)
		length_mem_values[i][1] = max(min(((length[i]-2.5)/(2.75-2.5)), 1, (5.2-length[i])/(5.2-4.9)), 0)
		length_mem_values[i][2] = max(min(((length[i]-4.7)/(5-4.7)), 1, (7.5-length[i])/(7.5-7)), 0)

		width_mem_values[i][0] = max(min(((width[i]-0)/(0.1-0)), 1, (1-width[i])/(1-0.75)), 0)
		width_mem_values[i][1] = max(min(((width[i]-0.8)/(1-0.8)), 1, (2-width[i])/(2-1.5)), 0)
		width_mem_values[i][2] = max(min(((width[i]-1.3)/(1.8-1.3)), 1, (3-width[i])/(3-2.5)), 0)

		#define rule set
		rule1_val = min(length_mem_values[i][0], width_mem_values[i][0])	#if petal length & width is small
		rule2_val = min(length_mem_values[i][1], width_mem_values[i][1])	#if petal length & width is medium
		rule3_val = min(length_mem_values[i][2], width_mem_values[i][2])	#if petal length & width is large

		for j in range(len(setosa_mf)):
			#using correlation min as implication operator
			rule1_corr_min = min(rule1_val, setosa_mf[j])
			rule2_corr_min = min(rule2_val, versicolor_mf[j])
			rule3_corr_min = min(rule3_val, virginica_mf[j])

			#if aggregation is max
			if(aggregation_type == 0):
				aggregation_vals[i][j] = max(rule1_corr_min, rule2_corr_min, rule3_corr_min)

			#if aggregation is sum
			elif(aggregation_type == 1):
				aggregation_vals[i][j] = rule1_corr_min + rule2_corr_min + rule3_corr_min

			#calculate centroid 
			centroid_numerator += (j*aggregation_vals[i][j])
			centroid_denominator += aggregation_vals[i][j]

		#divide by 0 sanity check
		#if centroid_denominator == 0:
			#counter += 1
			#print(centroid_numerator, centroid_denominator, counter, i)

		centroid = centroid_numerator / centroid_denominator

		#classify as setosa
		if(centroid >= 0 and centroid < 2):
			results[i] = 0

		#classify as versicolor
		if(centroid >= 2 and centroid < 4):
			results[i] = 1

		#classify as virginica
		if(centroid >= 4):
			results[i] = 2

		#if classification is the correct, +1 for accuracy
		if(results[i] == output[i]):
			accuracy += 1

	accuracy = ((accuracy / len(results)) * 100)
	print("\nAccuracy using a trapezoidal membership function = " + str(accuracy) + " %")


def guassian_membership_function(length, width, output, aggregation_type):
	length_mem_values = np.zeros((150, 3))		#array of membership values for small, medium, large for length
	width_mem_values = np.zeros((150, 3))		#array of membership values for small, medium, large for length

	setosa_mf 	  = np.array([0,1,0,0,0,0,0])
	versicolor_mf = np.array([0,0,0,1,0,0,0])
	virginica_mf  = np.array([0,0,0,0,0,1,0])

	aggregation_vals = np.zeros((150, 7))
	results = np.zeros(150)
	counter = 0
	accuracy = 0

	#splits length column into the 3 classes: setosa, versicolor, & viginica
	seg1_length = length[:50]
	seg2_length = length[50:100]
	seg3_length = length[100:]

	#finds standard deviation for length for each class -> for gaussian mf calculation
	seg1_std_length = np.std(seg1_length)
	seg2_std_length = np.std(seg2_length)
	seg3_std_length = np.std(seg3_length)

	#print(seg1_std_length, seg2_std_length, seg3_std_length)

	#finds mean for length for each class -> for guassian mf calculation
	seg1_avg_length = np.average(seg1_length)
	seg2_avg_length = np.average(seg2_length)
	seg3_avg_length = np.average(seg3_length)

	#print(seg1_avg_length, seg2_avg_length, seg3_avg_length)

	#splits width column into the 3 classes: setosa, versicolor, & virginica
	seg1_width = width[:50]
	seg2_width = width[50:100]
	seg3_width = width[100:]

	#finds standard deviation for the width for each class -> for gaussian mf calculation
	seg1_std_width = np.std(seg1_width)
	seg2_std_width = np.std(seg2_width)
	seg3_std_width = np.std(seg3_width)

	#print(seg1_std_width, seg2_std_width, seg3_std_width)

	#finds mean for width for each class -> for gaussian mf calculation
	seg1_avg_width = np.average(seg1_width)
	seg2_avg_width = np.average(seg2_width)
	seg3_avg_width = np.average(seg3_width)

	#print(seg1_avg_width, seg2_avg_width, seg3_avg_width)

	for i in range(len(length)):
		centroid_numerator = 0.0
		centroid_denominator = 0.0
		centroid = 0.0

		#define membership functions
		length_mem_values[i][0] = math.exp(-((length[i] - seg1_avg_length)**2)/(2*(seg1_std_length**2)))
		length_mem_values[i][1] = math.exp(-((length[i] - seg2_avg_length)**2)/(2*(seg2_std_length**2)))
		length_mem_values[i][2] = math.exp(-((length[i] - seg3_avg_length)**2)/(2*(seg3_std_length**2)))

		width_mem_values[i][0] = math.exp(-((width[i] - seg1_avg_width)**2)/(2*(seg1_std_width**2)))
		width_mem_values[i][1] = math.exp(-((width[i] - seg2_avg_width)**2)/(2*(seg2_std_width**2)))
		width_mem_values[i][2] = math.exp(-((width[i] - seg3_avg_width)**2)/(2*(seg3_std_width**2)))

		#define rule set
		rule1_val = min(length_mem_values[i][0], width_mem_values[i][0])	#if petal length & width is small
		rule2_val = min(length_mem_values[i][1], width_mem_values[i][1])	#if petal length & width is medium
		rule3_val = min(length_mem_values[i][2], width_mem_values[i][2])	#if petal length & width is large

		for j in range(len(setosa_mf)):
			#using correlation min as implication operator
			rule1_corr_min = min(rule1_val, setosa_mf[j])
			rule2_corr_min = min(rule2_val, versicolor_mf[j])
			rule3_corr_min = min(rule3_val, virginica_mf[j])

			#if max aggregation
			if(aggregation_type == 0):
				aggregation_vals[i][j] = max(rule1_corr_min, rule2_corr_min, rule3_corr_min)

			#if sum aggregation
			elif(aggregation_type == 1):
				aggregation_vals[i][j] = rule1_corr_min + rule2_corr_min + rule3_corr_min

			#calculate centroid
			centroid_numerator += (j*aggregation_vals[i][j])
			centroid_denominator += aggregation_vals[i][j]

		#divide by 0 sanity check
		#if centroid_denominator == 0:
		#	counter += 1
		#	print(centroid_numerator, centroid_denominator, counter, i)

		centroid = centroid_numerator / centroid_denominator

		#classify as setosa
		if(centroid >= 0 and centroid < 2):
			results[i] = 0

		#classify as versicolor
		if(centroid >= 2 and centroid < 4):
			results[i] = 1

		#classify as virginica
		if(centroid >= 4):
			results[i] = 2

		#if correct +1 to accuracy
		if(results[i] == output[i]):
			accuracy += 1

	accuracy = ((accuracy / len(results)) * 100)
	print("\nAccuracy using a Gaussian membership function = " + str(accuracy) + " %\n")

### END GAUSSIAN FUNCTION

#call functions
trapezoidal_membership_function(length, width, output, aggregation_type)
guassian_membership_function(length, width, output, aggregation_type)