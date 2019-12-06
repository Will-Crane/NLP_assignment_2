#extracts review text from raw webpages and puts each review in a file in either POR or NEG folders
#reviews of fewer than 300 words are not considered, all numbers in the text are removed (not as fancy as Petal)
#stratification within the reviews - an equal number of pos/neg reviews for each film chosen (randomized which reviews make it)
#total dataset size capped at 400
import os
import random
from string import digits

def is_month(tocheck):
	if tocheck in ['January','February','March','April','May','June','July','August','September','October','November','December']:
		return True
	else:
		return False

def is_year(tocheck):
	yearnum = int(tocheck)
	if yearnum > 1990 and yearnum < 2020:
		return True
	else:
		return False

def is_out_of_line(line):
	#checks to see if the given line is of the format "X out of Y" where X,Y ints with X<=Y 
	#returns true if it is
	if len(line.split()) != 4:
		return False
	else:
		if line.split()[1] != 'out' or line.split()[2] != 'of':
			try:
				if int(line.split()[0].replace(",","")) > int(line.split()[3].replace(",","")):
					return False
				else:
					return True
			except:
				return False
			return False
		else:
			return True

def process_files():
	totalreviews = 0
	#get list of raw files:
	raw_file_list = ['assignment_2_dply_dataset/Raw_files/' + f for f in os.listdir('assignment_2_dply_dataset/Raw_files') if not f.startswith('.')]
	for filename in raw_file_list:
		#stores the text for all previously collected reviews (needed as IMDB sometimes has duplicate reviews)
		previous_reviews = []
		posreviews = []
		negreviews = []
		#read the file in:
		with open(filename, encoding='utf8') as f:
			fulltext = f.read()
		reviews = fulltext.split(" found this helpful. Was this review helpful? Sign in to vote.")
		for review in reviews:
			reviewtext = ''
			review_score = -1
			start_reading = False
			lines = review.splitlines()
			#iterate through all non-empty lines:
			for line in [i for i in lines if i]:
				if start_reading == True:
					#we are in the body of the review
					if line != 'Warning: Spoilers' and not is_out_of_line(line):
						reviewtext = reviewtext + ''.join(filter(lambda c: not c.isdigit(), line)) + " "
				elif review_score != -1:
					#we have the review score already, parse in the review text:
					#is this the [USERNAME MONTH YEAR] line that directly precedes the review text?
					if len(line.split()) == 3 and is_month(line.split()[1]) and is_year(line.split()[2]):
						start_reading = True
				#is this line the numbered score??
				elif line[0] == ' ':
					#verify that this line actually is a score out of 10:
					out_of = line.split('/')
					if len(out_of) == 2 and out_of[1] == '10':
						#collect review_score:
						review_score = int(out_of[0].strip())
						#verify review score is between 0 and 11:
						if review_score <= 0 or review_score > 10:
							#we haven't got the right line/review score
							review_score = -1
			#if the review has a score and is not less than 400 words and the rating is extreme, write it to a file
			if reviewtext and len(reviewtext.split()) >= 300 and reviewtext not in previous_reviews:
				if review_score >= 8:
					posreviews.append(reviewtext)
					#combat review duplication that sometimes happens from IMDB website
					previous_reviews.append(reviewtext)
				if review_score <= 3:
					negreviews.append(reviewtext)
					#combat review duplication that sometimes happens from IMDB website
					previous_reviews.append(reviewtext)

			random.shuffle(posreviews)
			random.shuffle(negreviews)
			#stratification within reviews:
			for i in range(0,min(len(posreviews),len(negreviews))):
				#write pos file
				with open('assignment_2_dply_dataset/POS/' + filename[42:] + "_" + "POS" + str(i), "w+") as pf:
					pf.write(posreviews[i])
				#write neg file:
				with open('assignment_2_dply_dataset/NEG/' + filename[42:] + "_" + "NEG" + str(i), "w+") as nf:
					nf.write(negreviews[i])
				totalreviews += 2

process_files()