from thesis import data


dummy = data.HorizonTSData('dummy', 2, 6, 2, l=2)

print dummy.x_train
print dummy.x_test
print dummy.y_train
print dummy.y_test
