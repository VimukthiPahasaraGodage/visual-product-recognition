import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

training_dataset_train = pd.read_csv('training-dataset/train.csv', low_memory=False)
training_dataset_test = pd.read_csv('training-dataset/test.csv', low_memory=False)

train_df = pd.concat([training_dataset_train, training_dataset_test], ignore_index=True)

train_df.to_csv('trainn.csv', index=False)

# Create an empty DataFrame to store the results
results_1 = pd.DataFrame(columns=['image1', 'image2', 'similarity', 'class'])

# Create a copy of the train_df DataFrame
train_df_copy = train_df.copy()

# Get the number of iterations as manual input from the user
num_iterations = 400000

# Create a list to store the individual DataFrames
dfs = []
available_classes = train_df_copy['class'].unique()

for i in range(num_iterations):
    # Filter train_df to get rows with the same class
    class_choice = random.choice(available_classes)

    class_rows = train_df_copy[train_df_copy['class'] == class_choice]

    if len(class_rows) >= 2:
        random_indices = random.sample(class_rows.index.tolist(), 2)
        random_rows = class_rows.loc[random_indices]

        # Get the image names and classes
        image1_name = random_rows.iloc[0]['name']
        image2_name = random_rows.iloc[1]['name']
        class1 = random_rows.iloc[0]['class']
        class2 = random_rows.iloc[1]['class']

        # Determine whether the classes are the same
        if class1 == class2:
            similarity = 1

        # Create a new DataFrame with the information
        result_df = pd.DataFrame({
            'image1': [image1_name],
            'image2': [image2_name],
            'similarity': [similarity],
            'class': [class1]
        })

        # Append the result DataFrame to the list
        dfs.append(result_df)

        # # Remove the selected rows from train_df_copy
        # train_df_copy.drop(random_indices[0], inplace=True)

        # # Remove the class from available_classes if it has <= 1 rows
        # if len(class_rows) <= 1:
        #     available_classes = [c for c in available_classes if c != class_choice]

# Concatenate all individual DataFrames into the final results DataFrame
results_1 = pd.concat(dfs, ignore_index=True)

print(f"Generated {num_iterations} similarity pairs and saved to 'same_image_similarity_results.csv'")

# Create a new column with sorted image pairs
results_1['sorted_images'] = results_1.apply(lambda row: tuple(sorted([row['image1'], row['image2']])), axis=1)

# Drop duplicate rows based on sorted image pairs
filtered_results_1 = results_1.drop_duplicates(subset='sorted_images')

# Drop the temporary sorted_images column
results_1 = filtered_results_1.drop(columns=['sorted_images'])

# Create an empty DataFrame to store the results
results_2 = pd.DataFrame(columns=['image1', 'image2', 'similarity', 'class'])

# Create a copy of the train_df DataFrame
train_df_copy = train_df.copy()

# Create an empty set to store the unique class names
unique_classes = set()

# Get the number of iterations as manual input from the user
num_iterations = 900000

# Create a list to store the individual DataFrames
dfs = []

for i in range(num_iterations):
    # Choose two random rows from the DataFrame
    random_indices = random.sample(range(len(train_df)), 2)
    random_rows = train_df.iloc[random_indices]

    # Get the image names and classes
    image1_name = random_rows.iloc[0]['name']
    image2_name = random_rows.iloc[1]['name']
    class1 = random_rows.iloc[0]['class']
    class2 = random_rows.iloc[1]['class']

    unique_classes.update([class1])
    unique_classes.update([class2])

    # Determine whether the classes are the same
    if class1 == class2:
        similarity = 1
    else:
        similarity = 0

    # Create a new DataFrame with the information
    result_df = pd.DataFrame({
        'image1': [image1_name],
        'image2': [image2_name],
        'similarity': [similarity],
        'class': [class1]
    })

    # Append the result DataFrame to the list
    dfs.append(result_df)

# Concatenate all individual DataFrames into the final results DataFrame
results_2 = pd.concat(dfs, ignore_index=True)

print(f"Generated {num_iterations} similarity pairs and saved to 'random_image_similarity_results.csv'")

# Create a new column with sorted image pairs
results_2['sorted_images'] = results_2.apply(lambda row: tuple(sorted([row['image1'], row['image2']])), axis=1)

# Drop duplicate rows based on sorted image pairs
filtered_results_2 = results_2.drop_duplicates(subset='sorted_images')

# Drop the temporary sorted_images column
results_2 = filtered_results_2.drop(columns=['sorted_images'])

similarity_results = pd.concat([results_1, results_2], ignore_index=True)
similarity_results = shuffle(similarity_results)

# Get the unique classes in the dataset
unique_classes = similarity_results['class'].unique()

# Initialize lists to store the data for train and validation sets
train_data = []
val_data = []

# Iterate through each unique class
for class_name in unique_classes:
    class_data = similarity_results[similarity_results['class'] == class_name]

    # Split the class data into training and validation sets
    train_class_data, val_class_data = train_test_split(class_data, test_size=0.1, random_state=42)

    train_data.append(train_class_data)
    val_data.append(val_class_data)

# Concatenate the data for train and validation sets
train_df = pd.concat(train_data, ignore_index=True)
val_df = pd.concat(val_data, ignore_index=True)

train_df = train_df.drop(columns=['class'])
val_df = val_df.drop(columns=['class'])

# Save the training and validation datasets to CSV files
train_df.to_csv('train_dataset.csv', index=False)
val_df.to_csv('validation_dataset.csv', index=False)
