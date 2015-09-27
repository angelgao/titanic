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

# Change all ticket price above $40 to $39 for easy  binning
fare_ceiling = 40
above40 = data[0::, 9].astype(np.float) >= fare_ceiling
data[above40, 9] = fare_ceiling - 1.0

fare_bracket_size = 10
num_of_fare_brackets = fare_ceiling / fare_bracket_size

num_classes = len(np.unique(data[0::, 2]))

# Survival table is array with 0 of 1 for 
# their corresponding gender, class, and ticket price
survival_table = np.zeros((2, num_classes, num_of_fare_brackets))

for i in xrange(1, num_classes+1):
    for j in xrange(num_of_fare_brackets):
        women_only_stats = data[                                     \
                                (data[0::,4] == "female") &          \
                                (data[0::,2].astype(np.float) == i) &\
                                (data[0:,9].astype(np.float) >= j*fare_bracket_size) &\
                                (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size),\
                                1]
                                
        men_only_stats = data[                                     \
                              (data[0::,4] != "female") &          \
                              (data[0::,2].astype(np.float) == i) &\
                              (data[0:,9].astype(np.float) >= j*fare_bracket_size) &\
                              (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size),\
                              1]

        survival_table[0,i-1,j] = np.mean(women_only_stats.astype(np.float)) 
        survival_table[1,i-1,j] = np.mean(men_only_stats.astype(np.float))
        # If no passenger for a given category set to 0 rather than nan for mean
        survival_table[ survival_table != survival_table ] = 0

"""
Sample survival_table: 

array([[[ 0.        ,  0.        ,  0.83333333,  0.97727273],
        [ 0.        ,  0.91428571,  0.9       ,  1.        ],
        [ 0.59375   ,  0.58139535,  0.33333333,  0.125     ]],

        [[ 0.        ,  0.        ,  0.4       ,  0.38372093],
        [ 0.        ,  0.15873016,  0.16      ,  0.21428571],
        [ 0.11153846,  0.23684211,  0.125     ,  0.24      ]]])

E.g) Female passenger of 2nd class with fare $10-$19 has 91.4% survival
"""

# Assume any probability greater than 0.5 should result in survival
survival_table[ survival_table < 0.5 ] = 0
survival_table[ survival_table >= 0.5 ] = 1 

"""
Writing to the Prediction file
"""

test_file = open('./csv/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()
predictions_file = open("./csv/genderclassmodel.csv", "wb")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])

for row in test_file_object:
    for j in xrange(num_of_fare_brackets):
        # Some passengers have no fare data
        try:
            row[8] = float(row[8])
        except:
            bin_fare = 3 - float(row[1])
            break

        if row[8] > fare_ceiling:
            bin_fare = num_of_fare_brackets - 1
            break
        if row[8] >= j * fare_bracket_size \
           and row[8] < (j+1) * fare_bracket_size:
            bin_fare = j
            break

    if row[3] == 'female':
        p.writerow([row[0], int(survival_table[0, float(row[1])-1, bin_fare])])
    else:
        p.writerow([row[0], int(survival_table[1, float(row[1])-1, bin_fare])])

test_file.close()
predictions_file.close()
