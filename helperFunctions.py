import os
import requests
import numpy as np
from matplotlib import pyplot as plt

def download_dataset(url, file_path):
    if not os.path.exists(file_path):
        print("Downloading dataset...")
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print("Download completed!")
    else:
        print("Dataset already downloaded.")
        
def clean_data_set(data):
    # Fill missing values in 'carbo' and 'sugars' with their respective means
    data['carbo'].fillna(data['carbo'].mean(), inplace=True)
    data['sugars'].fillna(data['sugars'].mean(), inplace=True)

    # Fill missing values in 'potass' with the median
    data['potass'].fillna(data['potass'].median(), inplace=True)
     
def plot_cereal_manufacturer_frequency(cereal_data, manufacturer_map):
    #map manufacturer codes to names
    cereal_data['manufacturer_name'] = cereal_data['mfr'].map(manufacturer_map)

    #calculate the frequency of each manufacturer
    manufacturer_counts = cereal_data['manufacturer_name'].value_counts()

    #create a bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(manufacturer_counts.index, manufacturer_counts.values, 
                   color=['blue', 'green', 'orange', 'red', 'white', 'purple', 'pink'],
                   edgecolor='black')

    #chart title and labels
    plt.title('Frequency of Cereal Manufacturers', fontsize=16)
    plt.xlabel('Manufacturer', fontsize=14)
    plt.ylabel('Number of Products', fontsize=14)

    #  x and y ticks
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.show()
    
def plot_calpro_vs_rating(cereal_data):
    fig, ax = plt.subplots(figsize=(10, 6))

    custom_colors = ['blue', 'green', 'orange', 'red', 'white', 'purple', 'pink']
    manufacturers = cereal_data['manufacturer_name'].unique()
    colors = np.resize(custom_colors, len(manufacturers))

    for i, manufacturer in enumerate(manufacturers):
        subset = cereal_data[cereal_data['manufacturer_name'] == manufacturer]
        ax.scatter(subset['calpro'], subset['rating'], color=colors[i], label=manufacturer,
                   alpha=0.7, edgecolors='k', s=50)

    ax.set_xlabel('Calories per gram of protein (Calpro)', fontsize=12)
    ax.set_ylabel('Rating', fontsize=12)
    ax.set_title('Scatter Plot of Calpro vs. Rating Colored by Manufacturer', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title='Manufacturer', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize=11)

    plt.tight_layout()
    plt.show()
    
def plot_fiber_vs_rating(cereal_data):
    fig, ax = plt.subplots(figsize=(10, 6))

    custom_colors = ['blue', 'green', 'orange', 'red', 'white', 'purple', 'pink']
    manufacturers = cereal_data['manufacturer_name'].unique()
    colors = np.resize(custom_colors, len(manufacturers))

    for i, manufacturer in enumerate(manufacturers):
        subset = cereal_data[cereal_data['manufacturer_name'] == manufacturer]
        ax.scatter(subset['fiber'], subset['rating'], color=colors[i], label=manufacturer,
                   alpha=0.7, edgecolors='k', s=50)

    ax.set_xlabel('Fiber In Cereal', fontsize=12)
    ax.set_ylabel('Rating', fontsize=12)
    ax.set_title('Fiber In Cereal vs. Rating Colored by Manufacturer', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title='Manufacturer', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize=11)

    plt.tight_layout()
    plt.show()

def plot_fiber_distribution(cereal_data):
    fig, ax = plt.subplots(figsize=(8, 6))
    boxprops = dict(linestyle='-', linewidth=2, color='darkgoldenrod')
    whiskerprops = dict(linestyle='--', linewidth=2, color='orange')
    medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
    
    cereal_data.boxplot(column='fiber', by='type', ax=ax, grid=False,
                        boxprops=boxprops, whiskerprops=whiskerprops, medianprops=medianprops)

    ax.set_title('Distribution of Fiber Content by Cereal Type', fontsize=16, fontweight='bold')
    ax.set_xlabel('Cereal Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fiber (grams per serving)', fontsize=14, fontweight='bold')
    plt.suptitle('')  # Remove the default 'Boxplot grouped by type' title

    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_facecolor('whitesmoke')

    plt.tight_layout()
    plt.show()

def plot_fiber_by_shelf_and_manufacturer(cereal_data):
    grouped_data = cereal_data.groupby(['manufacturer_name', 'shelf'])['fiber'].mean().unstack()

    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 8))
    grouped_data.plot(kind='bar', ax=ax)

    # Adding plot aesthetics
    ax.set_title('Average Fiber Content by Shelf Level and Manufacturer', fontsize=15)
    ax.set_xlabel('Manufacturer', fontsize=12)
    ax.set_ylabel('Average Fiber (grams per serving)', fontsize=12)
    ax.legend(title='Shelf Level', title_fontsize='13', fontsize='11')
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_sugars_fat_vs_fiber(cereal_data):
    # Calculate Sugars + Fat for each cereal
    cereal_data['sugars_fat'] = cereal_data['sugars'] + cereal_data['fat']

    plt.figure(figsize=(14, 8))
    x_ticks = list(range(len(cereal_data)))

    # Plot Sugars + Fat
    plt.plot(x_ticks, cereal_data['sugars_fat'], label='Sugars + Fat', color='r', marker='o')
    # Plot Fiber
    plt.plot(x_ticks, cereal_data['fiber'], label='Fiber', color='g', marker='x')

    plt.title('Comparison of Sugars + Fat and Fiber Across Cereals')
    plt.xlabel('Cereal Index')
    plt.ylabel('Nutritional Content')
    plt.xticks(x_ticks, cereal_data['name'], rotation=90) 

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_health_values(cereal_data):
    #normalize the relevant columns
    cereal_data['fiber_norm'] = cereal_data['fiber'] / cereal_data['fiber'].max()
    cereal_data['protein_norm'] = cereal_data['protein'] / cereal_data['protein'].max()
    cereal_data['vitamins_norm'] = cereal_data['vitamins'] / cereal_data['vitamins'].max()
    cereal_data['sugars_norm'] = 1 - cereal_data['sugars'] / cereal_data['sugars'].max()
    cereal_data['sodium_norm'] = 1 - cereal_data['sodium'] / cereal_data['sodium'].max()

    #calculate the 'health_value'
    cereal_data['health_value'] = cereal_data[['fiber_norm', 'protein_norm', 'vitamins_norm', 'sugars_norm', 'sodium_norm']].mean(axis=1)

    # sortt by 'health_value'
    cereal_data.sort_values('health_value', ascending=False, inplace=True)

    # plot the health values
    plt.figure(figsize=(19, 8))
    bars = plt.bar(cereal_data['name'], cereal_data['health_value'], color='skyblue', edgecolor='black')

    plt.title('Health Value of Breakfast Cereals', fontsize=16)
    plt.xlabel('Cereal Name', fontsize=16)
    plt.ylabel('Health Value', fontsize=16)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)

    #labeling
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom', fontsize=11, rotation=90)

    plt.tight_layout()
    plt.show()

def plot_health_value_vs_rating(cereal_data):
    #create a scatter plot of health value vs. rating
    plt.figure(figsize=(10, 6))
    plt.scatter(cereal_data['health_value'], cereal_data['rating'], color='blue', edgecolor='black')
    plt.title('Health Value vs. Rating of Breakfast Cereals', fontsize=16)
    plt.xlabel('Health Value', fontsize=14)
    plt.ylabel('Rating', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_health_value_by_shelf(cereal_data):
    #create a box plot of health value vs. shelf level
    fig, ax = plt.subplots(figsize=(8, 6))
    boxprops = dict(linestyle='-', linewidth=2, color='darkgoldenrod')
    whiskerprops = dict(linestyle='--', linewidth=2, color='orange')
    medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
    cereal_data.boxplot(column='health_value', by='shelf', ax=ax, grid=False, 
                        boxprops=boxprops, whiskerprops=whiskerprops, medianprops=medianprops)

    #set plot titles and labels with enhanced fonts
    ax.set_title('Distribution of Health Value by Shelf Level', fontsize=16, fontweight='bold')
    ax.set_xlabel('Shelf Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('Health Value', fontsize=14, fontweight='bold')
    plt.suptitle('')  #remove the default 'Boxplot grouped by type' title

    #improving the tick labels for better readability
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    #adding a light background grid for better readability
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    #set the background color
    ax.set_facecolor('whitesmoke')

    #show the plot
    plt.tight_layout()
    plt.show()

def plot_rating_by_shelf(cereal_data):
    # compare rating with shelf level to see if higher rated cereals are placed on higher shelves
    grouped_data = cereal_data.groupby(['shelf'])['rating'].mean()

    plt.figure(figsize=(8, 6))
    plt.bar(grouped_data.index, grouped_data.values, color='skyblue', edgecolor='black')

    plt.title('Average Rating by Shelf Level', fontsize=16)
    plt.xlabel('Shelf Level', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_fiber_vs_rating(cereal_data):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(cereal_data['fiber'], cereal_data['rating'], color='blue', alpha=0.7, edgecolors='k', s=50)

    ax.set_xlabel('Fiber In Cereal', fontsize=12)
    ax.set_ylabel('Rating', fontsize=12)
    ax.set_title('Fiber In Cereal vs. Rating', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def plot_fiber_vs_calorie_efficiency(cereal_data):
    # calculate coefficients for the line of best fit
    m, b = np.polyfit(cereal_data['fiber'], cereal_data['calorie_efficiency'], 1)

    # plot the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(cereal_data['fiber'], cereal_data['calorie_efficiency'], color='blue', alpha=0.7, edgecolors='k', s=50)

    # plot the line of best fit
    ax.plot(cereal_data['fiber'], m*cereal_data['fiber'] + b, color='red')

    ax.set_xlabel('Fiber In Cereal', fontsize=12)
    ax.set_ylabel('Calorie Efficiency', fontsize=12)
    ax.set_title('Fiber In Cereal vs. Calorie Efficiency', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def plot_calorie_efficiency_by_shelf(cereal_data):
    # group data by shelf level and calculate average calorie efficiency
    grouped_data = cereal_data.groupby(['shelf'])['calorie_efficiency'].mean()

    plt.figure(figsize=(8, 6))
    plt.bar(grouped_data.index, grouped_data.values, color='skyblue', edgecolor='black')

    plt.title('Average Calorie Efficiency by Shelf Level', fontsize=16)
    plt.xlabel('Shelf Level', fontsize=14)
    plt.ylabel('Average Calorie Efficiency', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()

