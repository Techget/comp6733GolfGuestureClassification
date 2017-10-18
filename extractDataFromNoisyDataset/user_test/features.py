#! /usr/bin/python

    
import sys, math
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import normalize

# if len(sys.argv) != 2:
#     sys.stderr.write("Usage: %s <file>\n" % sys.argv[0])
#     sys.exit(1)

#####################################################################################
joints={}

X = 'X'
Y = 'Y'
Z = 'Z'
SpineBase 		= 1
SpineMid 		= 2
Neck 			= 3
Head			= 4
ShoulderLeft	= 5
ElbowLeft		= 6
WristLeft		= 7
HandLeft    	= 8
ShoulderRight	= 9
ElbowRight		= 10
WristRight		= 11
HandRight       = 12
HipLeft         = 13
KneeLeft        = 14
AnkleLeft       = 15
FootLeft		= 16
HipRight        = 17
KneeRight		= 18
AnkleRight		= 19
FootRight		= 20
SpineShoulder	= 21
HandTipLeft		= 22
ThumbLeft		= 23
HandTipRight	= 24
ThumbRight		= 25

setup			= 1
topOfSwing		= 2
impact			= 3
followThrough	= 4
finish			= 5

#####################################################################################

def readJointValues(file):
	j = SpineBase
	for line in open(file):
		line = line.strip()
		values = line.split(',')
		if not line:
			continue
		joints[j] = {'X': float(values[0].strip()), 'Y': float(values[1].strip()), 'Z': float(values[2].strip())}
		if j < 25:
			j += 1
		else:
			j=1

# get distance between 2 joints in metres
def getDistance(joint1, joint2):
	distance = math.sqrt(math.pow(joint2[X]-joint1[X],2)+math.pow(joint2[Y]-joint1[Y],2)+math.pow(joint2[Z]-joint1[Z],2))
	return distance
	
def getKneeFlexion(Knee, Hip, Ankle):
	HipRightPos = np.array([Hip[X], Hip[Y], Hip[Z]])
	KneeRightPos = np.array([Knee[X], Knee[Y], Knee[Z]])
	AnkleRightPos = np.array([Ankle[X], Ankle[Y], Ankle[Z]])

	transA = HipRightPos - KneeRightPos
	transB = AnkleRightPos - KneeRightPos
	angles = np.arccos(np.sum(transA * transB, axis = 0)/(np.sqrt(np.sum(transA ** 2, axis = 0)) * np.sqrt(np.sum(transB ** 2, axis = 0))))

	return (np.pi - angles) * (180/np.pi)

def getAngle(Joint1, Joint2, Joint3):
	Joint1Pos = np.array([Joint1[X], Joint1[Y], Joint1[Z]])
	Joint2Pos = np.array([Joint2[X], Joint2[Y], Joint2[Z]])
	Joint3Pos = np.array([Joint3[X], Joint3[Y], Joint3[Z]])

	vectorJ2toJ1 = Joint2Pos - Joint1Pos
	vectorJ2toJ3 = Joint2Pos - Joint3Pos

	vectorJ2toJ1 = vectorJ2toJ1.reshape(1,-1) 
	vectorJ2toJ3 = vectorJ2toJ3.reshape(1,-1)

	vectorJ2toJ1 = preprocessing.normalize(vectorJ2toJ1, norm='l2')
	vectorJ2toJ3 = preprocessing.normalize(vectorJ2toJ3, norm='l2')

	vectorJ2toJ1 = np.squeeze(vectorJ2toJ1)
	vectorJ2toJ3 = np.squeeze(vectorJ2toJ3)

	dotProduct = np.dot(vectorJ2toJ1, vectorJ2toJ3)

	return np.arccos(dotProduct) / np.pi * 180;


def determineType(file):
	if "setup" in file:
		type = setup
	elif "top" in file:
		type = topOfSwing
	elif "impact" in file:
		type = impact
	elif "follow" in file:
		type = followThrough
	elif "finish" in file:
		type = finish

	return type

# return the angle of 
#	    joint1
#      /
#	  /
#    joint2-----joint3

def getAngle_test(joint1, joint2, joint3):
	joint1 = np.array(joint1.values())
	joint2 = np.array(joint2.values())
	joint3 = np.array(joint3.values())

	vec12 = joint1 - joint2
	vec32 = joint3 - joint2

	cosine_angle = np.dot(vec12, vec32) / (np.linalg.norm(vec12) * np.linalg.norm(vec32))
	angle = np.arccos(cosine_angle)

	return np.degrees(angle)

#################################################################

# if __name__ == "__main__":
# 	type = determineType(sys.argv[1])
# 	readJointValues(sys.argv[1])

# 	if type == setup:	
# 		print "<=== Setup ===>"
# 	#	print "Stance width: %.2f cm" % (getDistance(joints[AnkleLeft],joints[AnkleRight])*100)
# 	elif type == topOfSwing:
# 		print "<=== Top of swing ===>"
# 	elif type == impact:
# 		print "<=== Impact ===>"
# 	elif type == followThrough:
# 		print "<=== Follow Through ===>"
# 	elif type == finish:
# 		print "<=== Finish ===>"
    	
# 	print "Stance width: %.2f cm" % (getDistance(joints[AnkleLeft],joints[AnkleRight])*100)
# 	print "Neck Angle: %.2f Degrees" % (getDistance(joints[ElbowLeft],joints[ElbowRight])*100)
# 	print "Elbow Distance: %.2f cm" % (getAngle(joints[Head],joints[Neck],joints[SpineBase]))
# 	print "Elbow Angle(Right): %.2f Degrees" % (getAngle(joints[ShoulderRight],joints[ElbowRight],joints[WristRight])) 
# 	print "Elbow Angle(Left): %.2f Degrees" % (getAngle(joints[ShoulderLeft],joints[ElbowLeft],joints[WristLeft])) 
# 	print "Hip bend: %.2f Degrees" % (getAngle(joints[SpineShoulder],joints[SpineBase],joints[AnkleRight])) 
# 	print "Knee Flexion(Right): %.2f Degrees" % (getKneeFlexion(joints[HipRight],joints[KneeRight],joints[AnkleRight])) 
# 	print "Knee Flexion(Left): %.2f Degrees" % (getKneeFlexion(joints[HipLeft],joints[KneeLeft],joints[AnkleLeft]))
