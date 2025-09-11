<div align="center">  

# NFLGameML Predictive Model  

----- By: Lance Kluge ----  

</div>  

---

## OVERVIEW  

This is a machine learning model utilizing **XGBoost**, **Pandas**, and **SKLearn**.  
It focuses on predicting the winner of the NFL football game given only stats available before the game was played.  

This is accomplished by utilizing data from:  
[https://www.kaggle.com/datasets/cviaxmiwnptr/nfl-team-stats-20022019-espn](url)  

This dataset is an accumulation of all the games played from 2002 season until the end of the 2024. The dataset contains things like the two teams that played, the week they played in, the season they played in, and then all the team stats you would think of.  

Things like:  
- Points scored by the teams  
- Total yards  
- Total yards from rushing  
- Total yards from passing  
- Interceptions  
- Turnovers  
- Sacks  
- First downs  
- And much more  

All of these stats were wrangled using some Pandas data wrangling in order to create what I will refer to as **"Rolling Stats"**.  

---

## ROLLING STATS 

These rolling stats focus on creating data that would be available to you and I before an NFL game would ever be played. That means that we are omitting week 1 predictions as we don't have the greatest ability to predict those stats from this season given no games had been played.  

I considered using the team’s season averages from the end of the last season but decided that teams change slightly too much for that to be something worth doing. Therefore we have transformed the data into rolling stats.  

- Week 2 game → averages from the previous week (week 1)  
- Week 10 game → averages from weeks 1–9  

The key to creating these rolling stats was to:  
1. Mark each game with an ID  
2. Split up the two teams and bring their stats into one long dataframe  
3. Sort by team and season  
4. Apply the rolling stats method  
5. Zip it all back together based on the game_ID  

---

## TRAIN VS TESTING DATA  

Originally I wanted to use just one season to test and the rest to train (from 2005 onwards).  

I chose these two cutoffs as for 1, the game of football does actually change over time. These past decade or two has started to let passing attacks shine when compared to (relatively) more run focused offenses. The 2005 cutoff was chosen fairly arbitrarily but that was when some of the oldest qbs still in the league were drafted Aaron Rodgers.  

This gives a nice balance of still being closer to the modern game, but also giving enough training data to use.  

I end up switching to testing on the 2023 and 2024 season after model 1 just to get double of games we need to predict so we get a better sense of how accurate the model is going to be on seasons (and games) that it has not seen yet in its training phase.  

---

## FIRST MODEL  

Onto the modeling part of the project and we first start out doing zero transformation and including about 50 columns in the model.  

- Accuracy: **~59%**  
- Comparison: just picking the home team every single time = **57%**  

It was sloppy and not very well thought out, but a nice confidence boost that I was going to be able to be better than just picking the home team all the time.  

---

## SECOND MODEL  

The second model we started to dabble with diff columns instead of home and away columns. We wanted to see if one team had a significant advantage over the other team in terms of a certain given category.  

- Accuracy: **~63%**  

---

## THIRD MODEL

The third model started looking at which columns we should actually be including and it came out to the following:  

`['score_diff', 'points_allowed_diff', 'pass_att_diff', 'pen_yards_diff', 'first_downs_diff', 'sacks_yards_diff', 'first_downs_from_penalty_diff']`

All of these columns ended up yielding an **accuracy of 69.5%** which is even better than Vegas. I then looked to using a grid search to find the optimal model parameters to try and squeeze a little more accuracy out of the model.  

---

## GRID SEARCH

I ended up grid searching **972 possible combinations** using `GridSearchCV` from `sklearn`. This ended up yielding the following parameters:  

`{'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.01, 
 'max_depth': 4, 'n_estimators': 300, 'subsample': 0.6}`


Although this ended up yielding a best CV ROC AUC score of **0.67** and testing on the test data at an **accuracy of 66%**. This was slightly confusing at first given I just had a model that had an accuracy of 69.5%, so I decided to do some more digging.  

---

## SEASONAL CV EXPLORATION

The seasonal CV exploration was aimed at figuring out if I just got lucky/was overfitting to the 2023 and 2024 season. It looked at the cross-validation scores for years **2010 - 2024** and trained a model with the same attributes, but different parameters.  

For each season, it trained on all the other seasons (i.e., 2010 would train on 2011–2024) and then test on the season and record the accuracy and ROC AUC.  

It found that **Model 3 was particularly good at predicting the 2024 season, but not great on other seasons.** This led me to choosing the grid search model to continue with as it was more applicable to all seasons rather than being great at just one or two.  

---

## FINAL MODEL

The final model was just the best model found by the grid searched model utilizing the features found in the third model description. The final accuracy ended up being about **65%** for predicting NFL games that it had never seen based off the rolling, difference stats that were constructed for this model.  

I ended up doing some exploration into how accurate the model was across weeks and the rolling ROC and accuracy across the weeks.  

<p align="center">
  <img width="653" height="455" alt="image" src="https://github.com/user-attachments/assets/2988d83c-1b0e-4290-b896-818db4ee1c01" />
</p>

<p align="center">
  <img width="644" height="455" alt="image" src="https://github.com/user-attachments/assets/7834dc64-7314-4367-96df-ad07f80d2e75" />
</p>

The graphs are above and are slightly interesting to see that our **Week 2 predictions are so good given we only have 1 week of data** on the teams that season. One potential cause that I see for this is that you almost always have a healthy team to start the season and around **40% of injuries happen in the first 4 weeks.**  

So teams can be healthy for most of Week 1 and 2, but end up being injured later down the line which is not reflected in the teams’ rolling stats. Please also note that this analysis was done on data that the model trained on already so the accuracy numbers are not the ones that I would expect on an unseen NFL game.  

---

## HOW TO UTILIZE

I have provided the **`final_nfl_rolling_stats.csv`** file in case anyone would like to expand on this work, I would love to see what other people end up coming up with.  

You are also welcome to use the model, it needs the following inputs:  


`['score_diff', 'points_allowed_diff', 'pass_att_diff', 
 'pen_yards_diff', 'first_downs_diff', 'sacks_yards_diff', 
 'first_downs_from_penalty_diff']`

 Each column is going to be a rolling average as described above. The diff is calculated with:  

```
df[home_col] - df[away_col]
```

I also ended up predicting probabilities and therefore the code to predict outcomes would be:  

```
y_pred_proba = best_model.predict_proba(X)[:, 1]
y_pred = (y_pred_proba > 0.51).astype(int)  # Note: 0.51 was a better cutoff than 0.5
```

This will generate a `1` or `0` for each game, in which:  

- `1` = home team wins  
- `0` = away team wins  

---

I do not plan on using this model to sports bet, although I am going to have a little friendly competition with myself in terms of predicting games for the upcoming **2025 NFL season**, starting in Week 2.  

All together this was a whole lot of fun for me as someone who really loves the game of football and computer science. It was great to effectively take the training wheels off and pursue a project that was not for a grade or for Kaggle competitions in the machine learning field.  

Please feel free to reach out if you have any questions or comments and feel free to continue tweaking what I already came up with!
