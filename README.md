by: Jalen Li

### <a name="frame">Framing the Problem</a>

In my [exploratory data analysis](https://breezyinterwebs.github.io/Analyzing-League/), I looked at potential correlations between a team's end statistics and the outcome of their match. While we found that there is a strong correlation, knowing that a correlation exists is not enough. Furthermore, using the end statistics does not help us, because if we know the end statistics, we also know if a team wins or not. Therefore, we are interested in whether we can predict the outcome of a match using game statistics that arise during a match. Since a team can only win or lose a match, we are interested in creating a classifier.

In the dataset, the result of a match is recorded to `result.` In order to answer our question of focus, we are building a binary classifier with decision trees, and evaluating the model solely on whether the model gets the `result` correct. A decision tree closely reflects how someone like a commentator or a casual viewer will predict a match outcome. Because the effects of Type I and Type II errors are inconsequential, there is little reason to use a metric which accounts for false positives and negatives.

From before, we need to select variables that can be determined during a match, not after it ends. For this reason, I've selected a few "categories." These are variables that are easy to track, but can show how a game is going that doesn't require the game to end first.
- `Kills, Deaths, Assists at 10` are stats about teamwide kills, deaths, and assists when the match reaches 10 minutes.
- `Gold, XP, CS diff at 10` are stats about which team is leading in gold, xp and CS when the match reaches 10 minutes.
- `Firsttotals` are a stat that I explored in my EDA, which I felt had good correlation to the performance of a team. It encodes how many "first checkpoints" a team accomplishes, such as first tower or first to three towers.
- `League` tells us what competitive league that the match took place in. Even with similar performance metrics, two matches may potentially take different routes depending on the average skill level.

### Baseline Model

For our baseline model, we will make a classifier using a Decision Tree to predict the win or loss of a match. Here, we'll start with two basic features: the `league` that the team is in, and the `killsat10`. 
- `league` is a qualitative feature which is nominal because teams can only belong to one league at a time. One may argue that `league` is ordinal because there are certainly leagues which are more highly regarded in terms of average skill. However, the details of that are messy, and so we regard it as not ordinal. Either way, we can OneHotEncode this qualitative feature.
- `killsat10` is a quantitative feature that is ordinal. Because this is our baseline model, we do not do anything to this feature and use it as is.
We also select a random 75-25 training-validation split. However, since entries are paired via matches, we'll use a custom split where we take random pairs in order to keep our data tidy.

Since we are making this model as our baseline, our hyperparameter for tree depth must be random initially. I picked 5 because that seems somewhat reasonable. 

Once we created the model and evaluated its performance on both our training and testing set, we see about **60% accurate predictions** from our Decision Tree classifier. This isn't particularly good, because we would expect a 50% accurate prediction rate if we were to guess completely at random. However, for such little data provided and random hyperparameter selection, this is good news. We can most likely improve this by a lot!

### Final Model

As outlined in [Framing the Problem](#frame), all of the variables that we selected contain information about how the game is progressing, especially focused around the 10 minute mark. Often in games, teams which can gain an early advantage often use that momentum to snowball into a win. Therefore, we will add all of the variables we selected to our model to see if we can improve on our baseline.

We did a few more transformations of our data before passing it into the model.
- `Kills, Deaths, Assists at 10` were transformed according to the proportion of K, D, and A's that the respective team shared at the 10 minute mark. For example, if Team A had 3 kills and Team B had 4, their corresponding transformed stat would be 3/7 and 4/7. In some way, we could think about this as normalizing the data - you would be ahead if you had 3 kills and the other had 0, whereas you'd be severely behind if the other team had 10.
- `Gold, XP, CS diff at 10` were Z-scored in order to give the data a sense of magnitude - by what *scale* are they leading? Are they performing substantially better than the average, or only marginally better?
- Because I took `firsttotal` from my EDA, I know that this takes integers from 0 to 7. It seems to have distinct enough groups that a transformation is not necessary.
- From before, we OneHotEncode `league`.

In order to select the hyperparameters for our model, I used an iterative algorithm. In total, I trained 58 models. The hyperparameters I varied were the `criterion` of the decision tree, which takes `entropy` or `gini`, and the max-depth of the tree, which I tested 2-30. I fit the model, then calculated the training and testing prediction rates, put them all in a list and found the best combination. 
<iframe src="assets/training.html" width=800 height=600 frameBorder=0></iframe>

This combination (surprisingly) turned out to be `max_depth` = 5, and `criterion` = entropy. I knew from previous experience with decision trees that increasing the depth will increase the accuracy of training predictions, while plateauing or even decreasing the accuracy of testing predictions. Therefore, I predicted that the best `max_depth` would be somewhat low.

With these hyperparameters selected, I checked the model on our training and testing set from before. Now, both of our prediction accuracy rates were approximately **78%**. A big jump from 60%!

##### A Quick Note

While I believe that engineering the features in this way makes sense to more accurately represent the predictors, there was a flaw that I ran into. Specifically, the `K,D,A` engineering required data to be paired in order to calculate the proportion correctly, and so taking a naive random subset would break this feature. It was at this point that I revised my training split. Instead of asking sklearn to do the split for me, I defined a custom function which took a test% of odd indicies, added 1 to get a test% of even indicies, then used the combination of the set and the negation of that set to get my testing and training data. This got my training and testing sets to work properly, but also meant that I could not use sklearn's `GridSearchCV`, because it took both a random subset and potentially an odd number of data. Therefore, I elected not to use k-fold CV, as I figured it would not greatly affect the performance of a hyperparameter if I was going to evaluate the accuracy prediction rate at the end anyway.

### Fairness Assessment

When creating models, one thing that should be considered is whether the model is fair across groups. For this dataset, I'm interested if the model predicts differently depending on the `playoffs`. Being in the playoffs creates a much higher-stakes situation for the teams involved, and that may make matches more unpredictable. Does our model hold up? In the dataset, the `playoffs` are encoded in 1 or 0, a binary classification for if the match takes place in a playoff or not.

To test this, we'll be using the absolute difference in prediction accuracy.<br>
*Null Hypothesis:* The absolute difference is 0. The model is fair to both scrims and playoffs.<br>
*Alternative Hypothesis:* The absolute difference is not 0. The model is potentially biased.

For my significance level, I've decided to use `alpha` = 0.01. While `alpha` = 0.05 is a rule of thumb, I often feel that a 1/20 is too often to reject. Hence, 1/100 feels like a much more reasonable level to reject away. We perform a modified version of permutation testing, and here we show a histogram of our results.
<iframe src="assets/permutation.html" width=800 height=600 frameBorder=0></iframe>

For this permutation test, our p-value is 0.025. Therefore, we'd fail to reject the null. It seems that our model performs similarly for both games in and out of playoffs.

