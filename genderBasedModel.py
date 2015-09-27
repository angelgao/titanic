import csv
import numpy as np

#reading from CSV file
csv_file_object = csv.reader(open('./csv/train.csv', 'rb'))

# Skip header of file
header = csv_file_object.next()

data = []

for row in csv_file_object:      # Run through each row in the csv file,
    data.append(row)

data = np.array(data)            # Convert from a list to an array

number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

women_only_stats = data[0::,4] == "female"  # Refer to all female rows
men_only_stats = data[0::,4] != "female"    # Refer to all male rows

# Using mask from above to select the females and males
women_onboard = data[women_only_stats,1].astype(np.float)     
men_onboard = data[men_only_stats,1].astype(np.float)

proportion_women_survive = np.sum(women_onboard) / np.size(women_onboard)
proportion_men_survive = np.sum(men_onboard) / np.size(men_onboard)

print "women who survive: ", proportion_women_survive
print "men who survive: ", proportion_men_survive

"""
Writing to the Prediction file
"""

#reading from CSV file
test_file = open('./csv/test.csv', 'rb')
test_file_object = csv.reader(test_file)

# Skip header of file
header = test_file_object.next()

prediction_file = open("./csv/genderbasedmodel.csv", "wb")
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:       # For each row in test.csv
    if row[3] == 'female':         # is it a female, if yes then                                       
        prediction_file_object.writerow([row[0],'1'])    # predict 1
    else:                              # or else if male,       
        prediction_file_object.writerow([row[0],'0'])    # predict 0

test_file.close()
prediction_file.close()
