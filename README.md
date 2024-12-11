## Introduction

### Introduction and Question Identification

League of Legends is a popular multiplayer online battle arena, or MOBA, game developed by Riot Games in 2009. With over 100 million registered players worldwide, it is one of the most influental games in esports history. To analyze competitive matches, we will be working with a professional data set developed by Oracle’s Elixir containing official League matches from 2022. This dataset includes various identifications and gameplay metrics from these matches, like player position, champion choices, kills, and earned gold. These variables are useful for analyzing gameplay viability and goals. 

While playing League of Legends, one of the main ways to get gold is by killing monsters around the map, including respawning minions and shared objectives teams fight over. The statistical metric "creep score" keeps track of every monster killed, with a general rule of thumb that every basic minion killed awards 1 creep score. With more monsters killed and thus more gold, the player can purchase stat increasing items, which could theoretically improve chances of victory and change gameplay strategies.

The central question we are interested in the relationship creep score per minute (cspm) has to other gameplay variables and metrics, including win rate. CSPM is used over creep score in general to account for varying game lengths. Using data analysis, the impact cspm and other statistics have on each other can be measured and tested. We can then utilize these given features to predict cspm. This predictive model can be used to strategize around high cspm objectives and picks, as well as improve the user's gameplay.

#### Introduction of Columns

This Oracle's Elixir dataset includes various identifications and gameplay metrics from competitive League matches in 2022. There are 150180 rows in this dataset, as well as 161 columns/variables. Here is a list of the key columns analyzed:

- gameid: Contains a unique string identifier representing an individual match.

- participantid: Contains an integer representing the participant's id. 1-5 represents a Red team player, 6-10 represents a Blue team player, 100 represents the Red team itself, and 200 represents the Blue team itself.

- result: Contains an integer representing the outcome of a match for the corresponding participant. 1 represents a win, 0 represents a loss.

- patch: Contains a string representing the game patch the match was played in.

- cspm: Contains a float representing the average monsters + minions killed per minute by a participant.

- kills: Contains an integer representing how many times a participant lands a finishing blow on an enemy champion.

- position: Contains a string represents the participant's position (top, jg, mid, adc, sup, or team itself).

- champion: Contains a string represents the champion picked by the participant.

- totalgold: Contains a int representing the overall amount of gold a player gained throughout the match from individual and shared actions.

- earnedgold: Contains a int representing the overall amount of gold a player gained throughout the match solely from individual actions.

- league: Contains a string represents the specific league tournament the match was held in, such as LEC for European matches.

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning
To clean the dataset, some columns are in a string format instead of a numerical format, which makes it harder to graph and find measures of central tendency. Therefore, multiple columns, including cspm, are converted to floats or integers with astype(). In addition, to account for missing values, every empty value in the dataset is converted to np.nan using replace(). In addition, a column 'cspm_missing' is recorded as "Yes" if a "cspm" value is missing and "No" if not to further analyze its missingness. Finally, for participants, two new dataframes are created, one holding only players and the other holding only teams for further analysis on both. This is because team stats are aggregates of each of its players. Here is the head of the cleaned dataset with some relevant columns:
```py
print(df[0:5][['gameid', 'participantid', 'playername', 'teamname', 'position', 'cspm', 'cspm_missing']])
```
### Univariate Data Analysis
To start off, univariate data analysis is performed on the cspm column for the full cleaned dataset using a histogram.

<iframe src="plots/cspm_plot.html" width=800 height=600 frameBorder=0></iframe>

This plot shows three different normal curves with their own peaks from 0-3, 5-12, and a smaller one at 25-40 cspm, which could be because different positions (including teams) on average have different cspm. However, for the overall dataset, it has a slight right skew. 

Univariate data analysis is performed again on the earned gold column.

<iframe src="plots/earned_plot.html" width=800 height=600 frameBorder=0></iframe>

This plot shows a high normal (though arguably very slightly right skewed) distribution from 0-20k earned gold, which is followed by a smaller, spread out distribution from 20k-60k earned gold. This could be because the first distribution is oriented towards the players' earned gold, while the second could represent the team's total earned gold.

### Bivariate Data Analysis
To further analyze the relationship between cspm and earned gold, a scatterplot is formed between them for bivariate analysis.
<iframe src="plots/cspm_gold_scatter.html" width=800 height=600 frameBorder=0></iframe>
In this scatterplot, cspm and earned gold have a positive correlation, and the data is organized into two clusters: one from 0-15 cspm, and the other from 20-50 cspm.
Another scatterplot is formed on the relationship between cspm and the participant's kills.
<iframe src="plots/cspm_kills_scatter.html" width=800 height=600 frameBorder=0></iframe>
In this scatterplot, the data is again organized into two clusters: one from 0-15 cspm, and the other from 20-50 cspm. The latter cluster has a greater average kill count compared to the 0-15 cspm cluster.
From these findings, we learned that cspm has a positive correlation with earned gold and kills, which directly result in a greater win percentage.

### Interesting Aggregates
For aggregates, we grouped cspm by position and analyzed the mean of each group.
```py
print(pivot_position)
```
From the pivot table, we can observe that the means are significantly different for each position. Even accounting for team being the sum of each participant there, mid and bot have the highest mean cspm while sup(port) has by far the lowest. Thus, we should keep in mind position's impact when trying to predict cspm.
## Assessment of Missingness

### Problem Identification
From the last part, we've learned that cspm has a considerable impact on winning results for competitive League of Legends matches. To prioritize this statistic for better games, creating a prediction model based off other columns/variables could be useful, such as league or champion. For this problem, we can utilize **regression** to predict cspm, since it's a numerical variable. So, a prediction problem now arises: **Can we predict a player's creep score/minute based off of other game statistics?**

As stated earlier, creep score/minute will be utilized as the response variable, since it's a reliable measure for creep score that accounts for varying game lengths. To evaluate our model, we will be using the numerical regression metric RMSE (Root Mean Squared Error) for interpretability compared to similar counterparts like MSE and balancing outliers.

## Hypothesis Testing
## Framing a Prediction Problem
## Baseline Model
