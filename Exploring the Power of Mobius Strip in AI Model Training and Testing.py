import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Generating virtual Mobius strip data
theta = np.linspace(0, 2 * np.pi, 1000)  # Fix the typos in the code
w = 0.5  # Set the width of the strip
strip_data = np.vstack([(1 + w * np.cos(theta)) * np.cos(theta),
                        (1 + w * np.cos(theta)) * np.sin(theta),
                        w * np.sin(theta)]).T

# Generating population data from 1950 to 2023
years = np.arange(1950, 2024)  # Fix the typos in the code
population = np.random.randint(200, 1000, size=len(years))  # Fix the typos in the code

# Splitting data into train and test sets
train_years = years[:-10]  # Use 10 years for training
train_population = population[:-10]
test_years = years[-10:]  # Use the last 10 years for testing
test_population = population[-10:]

# Training and testing the model on the Mobius strip
knn_model = KNeighborsRegressor(n_neighbors=1)

# Arrays to store predicted population and actual population
predicted_population = []
actual_population = []

for i in range(len(test_population)):
    # Training the model on one loop around the Mobius strip
    train_X = strip_data[i].reshape(1, -1)  # Reshape the strip data
    train_y = np.array([train_population[i]])

    knn_model.fit(train_X, train_y)

    #testing the model on the next loop around Mobius strip
    test_x=strip_data[(i+1)%len(strip_data)].reshape(1,-1)
    test_y=np.array([test_population[i]])
    predicted_y=knn_model.predict(test_x)
    #appending predicted and actual population to the arrays
    predicted_population.append(predicted_y[0])
    actual_population.append(test_y[0])
    #printing the predicted and actual population for each loop
    print(f"Loop {i+1}: Predicted Population = {predicted_y}, Actual Population = {test_y}")

# Plotting the predicted vs. actual population
plt.plot(years,population, 'bo-', label='Actual Population')
plt.plot(test_years, predicted_population, 'ro-', label='Predicted Population')
plt.xlabel('Years')
plt.ylabel('Population')
plt.title('Population Growth')
plt.legend()
plt.show()
