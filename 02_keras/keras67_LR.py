weight = 0.5
input = 0.5
goal_prediction = 0.8
lr = 0.1 # 0.1 / 1 / 0.001 / 0.0001
epoch = 12

for iteration in range(epoch) :
    prediction = input * weight
    error = (prediction - goal_prediction) **2   # mse

    print("Error : " + str(error) + "\tprediction : " + str(prediction))

    up_prediction = input * (weight + lr)
    up_error = (goal_prediction - up_prediction) ** 2    # mse
    down_prediction = input * (weight - lr)
    down_error = (goal_prediction - down_prediction) **2

    if(down_error <= up_error) :
        weight = weight - lr
    if(down_error > up_error) :
        weight = weight + lr