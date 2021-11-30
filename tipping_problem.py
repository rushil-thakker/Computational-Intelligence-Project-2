import numpy as np
import sys
import math
np.set_printoptions(threshold=sys.maxsize)

#food_quality is from 0 to 10
#service_quality is from 0 to 10
#tip_amount is from 0 to 25%

### CHANGE THESE VALUES TO CHANGE TIP ###
user_food_rating = 2.3					#0 is the worst and 10 is the best	
user_service_rating = 6.3 				#0 is the worst and 10 is the best
aggregation_type = 1					#Set to 0 for max aggregation, set to 1 for sum aggregation			
#########################################


def trapezoidal_membership_function(user_food_rating, user_service_rating, aggregation_type):
	food_quality_mf = np.zeros(3)			#array of membership values for bad, decent, great for food quality
	service_quality_mf = np.zeros(3)		#array of membership values for poor, acceptable, amazing for service quality

	#a low tip is defined as 0 to 9% with a 10% tip having 50% membership
	tip_low_mf 	  = np.array([1,1,1,1,1,1,1,1,1,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

	# a medium tip is defined as 11% to 18% with 10% and 19% having 50% membership
	tip_medium_mf = np.array([0,0,0,0,0,0,0,0,0,0.5,1,1,1,1,1,1,1,1,1,0.5,0,0,0,0,0,0])

	#a high tip is defined as 20% to 25% with 19% having 50% membership
	tip_high_mf   = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,1,1,1,1,1,1])

	aggregation_vals = np.zeros(26)
	tip = 0.0

	centroid_numerator = 0.0
	centroid_denominator = 0.0
	centroid = 0.0

	#define trapezoidal membership functions
	food_quality_mf[0] = max(min(1, (4-user_food_rating)/(4-3)), 0)
	food_quality_mf[1] = max(min((user_food_rating-3)/(4-3), 1, (8-user_food_rating)/(8-7)), 0)
	food_quality_mf[2] = max(min((user_food_rating-7)/(8-7), 1), 0)

	service_quality_mf[0] = max(min(1, (4-user_service_rating)/(4-3)), 0)
	service_quality_mf[1] = max(min((user_service_rating-3)/(4-3), 1, (8-user_service_rating)/(8-7)), 0)
	service_quality_mf[2] = max(min((user_service_rating-7)/(8-7), 1), 0)

	#define rule set
	rule1_val = min(food_quality_mf[0], service_quality_mf[0])		#if food bad 	& service poor 			tip low
	rule2_val = min(food_quality_mf[1], service_quality_mf[1])		#if food decent & service acceptable 	tip medium
	rule3_val = min(food_quality_mf[2], service_quality_mf[2])		#if food great 	& service amazing		tip high
	rule4_val = min(food_quality_mf[0], service_quality_mf[2])		#if food bad 	& service amazing 		tip high
	rule5_val = min(food_quality_mf[2], service_quality_mf[0])		#if food great 	& service poor 			tip medium
	rule6_val = min(food_quality_mf[2], service_quality_mf[1])		#if food great 	& service acceptable 	tip high
	rule7_val = min(food_quality_mf[0], service_quality_mf[1])		#if food bad 	& service acceptable 	tip medium
	rule8_val = min(food_quality_mf[1], service_quality_mf[2])		#if food decent & service amazing	 	tip high
	rule9_val = min(food_quality_mf[1], service_quality_mf[0])		#if food decent & service poor		 	tip low

	for j in range(len(tip_low_mf)):
		#take the correlation min as implication operator
		rule1_corr_min = min(rule1_val, tip_low_mf[j])
		rule2_corr_min = min(rule2_val, tip_medium_mf[j])
		rule3_corr_min = min(rule3_val, tip_high_mf[j])
		rule4_corr_min = min(rule4_val, tip_high_mf[j])
		rule5_corr_min = min(rule5_val, tip_medium_mf[j])
		rule6_corr_min = min(rule6_val, tip_high_mf[j])
		rule7_corr_min = min(rule7_val, tip_medium_mf[j])
		rule8_corr_min = min(rule8_val, tip_high_mf[j])
		rule9_corr_min = min(rule9_val, tip_low_mf[j])

		#if max aggregation
		if(aggregation_type == 0):
			aggregation_vals[j] = max(rule1_corr_min, rule2_corr_min, rule3_corr_min, rule4_corr_min, rule5_corr_min,
										rule6_corr_min, rule7_corr_min, rule8_corr_min, rule9_corr_min)
		#if sum aggregation
		elif(aggregation_type == 1):
			aggregation_vals[j] = rule1_corr_min + rule2_corr_min + rule3_corr_min + rule4_corr_min + rule5_corr_min + rule6_corr_min + rule7_corr_min + rule8_corr_min + rule9_corr_min

		#calculate the centriod
		centroid_numerator += (j*aggregation_vals[j])
		centroid_denominator += aggregation_vals[j]

	centroid = centroid_numerator / centroid_denominator

	#the tip is the centroid for this example
	tip = round(centroid, 3)
	print("\n(Trapezoidal Membership Function) Tip should be = " + str(tip) + " %")


def guassian_membership_function(user_food_rating, user_service_rating, aggregation_type):
	food_quality_mf = np.zeros(3)			
	service_quality_mf = np.zeros(3)		

	#same as for trapezoidal function
	tip_low_mf 	  = np.array([1,1,1,1,1,1,1,1,1,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
	tip_medium_mf = np.array([0,0,0,0,0,0,0,0,0,0.5,1,1,1,1,1,1,1,1,1,0.5,0,0,0,0,0,0])
	tip_high_mf   = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,1,1,1,1,1,1])

	aggregation_vals = np.zeros(26)
	tip = 0.0

	centroid_numerator = 0.0
	centroid_denominator = 0.0
	centroid = 0.0

	#standard deviation needed to do gaussian membership function calculations
	std = 1.8

	#print(seg1_std_length, seg2_std_length, seg3_std_length)

	#mean for the guassian membership function calculations
	low_avg = 0
	med_avg = 5
	high_avg = 10

	#define gaussian membership functions
	food_quality_mf[0] = math.exp(-((user_food_rating - low_avg)**2)/(2*(std**2)))
	food_quality_mf[1] = math.exp(-((user_food_rating - med_avg)**2)/(2*(std**2)))
	food_quality_mf[2] = math.exp(-((user_food_rating - high_avg)**2)/(2*(std**2)))

	service_quality_mf[0] = math.exp(-((user_service_rating - low_avg)**2)/(2*(std**2)))
	service_quality_mf[1] = math.exp(-((user_service_rating - med_avg)**2)/(2*(std**2)))
	service_quality_mf[2] = math.exp(-((user_service_rating - high_avg)**2)/(2*(std**2)))

	#define rule set
	rule1_val = min(food_quality_mf[0], service_quality_mf[0])		#if food bad 	& service poor 			tip low
	rule2_val = min(food_quality_mf[1], service_quality_mf[1])		#if food decent & service acceptable 	tip medium
	rule3_val = min(food_quality_mf[2], service_quality_mf[2])		#if food great 	& service amazing		tip high
	rule4_val = min(food_quality_mf[0], service_quality_mf[2])		#if food bad 	& service amazing 		tip high
	rule5_val = min(food_quality_mf[2], service_quality_mf[0])		#if food great 	& service poor 			tip medium
	rule6_val = min(food_quality_mf[2], service_quality_mf[1])		#if food great 	& service acceptable 	tip high
	rule7_val = min(food_quality_mf[0], service_quality_mf[1])		#if food bad 	& service acceptable 	tip medium
	rule8_val = min(food_quality_mf[1], service_quality_mf[2])		#if food decent & service amazing	 	tip high
	rule9_val = min(food_quality_mf[1], service_quality_mf[0])		#if food decent & service poor		 	tip low

	for j in range(len(tip_low_mf)):
		#calculate using correlation min as implication operator
		rule1_corr_min = min(rule1_val, tip_low_mf[j])
		rule2_corr_min = min(rule2_val, tip_medium_mf[j])
		rule3_corr_min = min(rule3_val, tip_high_mf[j])
		rule4_corr_min = min(rule4_val, tip_high_mf[j])
		rule5_corr_min = min(rule5_val, tip_medium_mf[j])
		rule6_corr_min = min(rule6_val, tip_high_mf[j])
		rule7_corr_min = min(rule7_val, tip_medium_mf[j])
		rule8_corr_min = min(rule8_val, tip_high_mf[j])
		rule9_corr_min = min(rule9_val, tip_low_mf[j])

		#if max aggregation
		if(aggregation_type == 0):
			aggregation_vals[j] = max(rule1_corr_min, rule2_corr_min, rule3_corr_min, rule4_corr_min, rule5_corr_min,
										rule6_corr_min, rule7_corr_min, rule8_corr_min, rule9_corr_min)

		#if sum aggregation
		elif(aggregation_type == 1):
			aggregation_vals[j] = rule1_corr_min + rule2_corr_min + rule3_corr_min + rule4_corr_min + rule5_corr_min + rule6_corr_min + rule7_corr_min + rule8_corr_min + rule9_corr_min

		#calculate centroid
		centroid_numerator += (j*aggregation_vals[j])
		centroid_denominator += aggregation_vals[j]

	centroid = centroid_numerator / centroid_denominator

	#tip is the centroid for this example
	tip = round(centroid, 3)
	print("\n(Gaussian Membership Function) Tip should be = " + str(tip) + " %\n")

### END GAUSSIAN FUNCTION

#call the functions
trapezoidal_membership_function(user_food_rating, user_service_rating, aggregation_type)
guassian_membership_function(user_food_rating, user_service_rating, aggregation_type)