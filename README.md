by: Jalen Li

## Analysis

### Framing the Problem

In my [exploratory data analysis](https://breezyinterwebs.github.io/Analyzing-League/), I looked at potential correlations between a team's end statistics and the outcome of their match. While we found that there is a strong correlation, knowing that a correlation exists is not enough. Furthermore, using the end statistics does not help us, because if we know the end statistics, we also know if a team wins or not. Therefore, we are interested in whether we can predict the outcome of a match using game statistics that arise during a match. Since a team can only win or lose a match, we are interested in creating a classifier.
In the dataset, the result of a match is recorded to `result.` In order to answer our question of focus, we are building a binary classifier with decision trees, and evaluating the model solely on whether the model gets the `result` correct. A decision tree closely reflects how someone like a commentator or a casual viewer will predict a match outcome. Because the effects of Type I and Type II errors are inconsequential, there is little reason to use a metric which accounts for false positives and negatives.
From before, we need to select variables that can be determined during a match, not after it ends. For this reason, I've selected a few "categories." These are variables that are easy to track, but can show how a game is going that doesn't require the game to end first.
- `Kills, Deaths, Assists at 10` are stats about teamwide kills, deaths, and assists when the match reaches 10 minutes.
- `Gold, XP, CS diff at 10` are stats about which team is leading in gold, xp and CS when the match reaches 10 minutes.
- `Firsttotals` are a stat that I explored in my EDA, which I felt had good correlation to the performance of a team. It encodes how many "first checkpoints" a team accomplishes, such as first tower or first to three towers.
- `League` tells us what competitive league that the match took place in. Even with similar performance metrics, two matches may potentially take different routes depending on the average skill level.

### <a name="baseline">Baseline Model</a>

For our baseline model, we will make a classifier using a Decision Tree to predict the win or loss of a match. Here, we'll start with two basic features: the `league` that the team is in, and the `killsat10`. 
- `league` is a qualitative feature which is nominal because teams can only belong to one league at a time. One may argue that `league` is ordinal because there are certainly leagues which are more highly regarded in terms of average skill. However, the details of that are messy, and so we regard it as not ordinal. Either way, we can OneHotEncode this qualitative feature.
- `killsat10` is a quantitative feature that is ordinal. Because this is our baseline model, we do not do anything to this feature and use it as is.
We also select a random 75-25 training-validation split. However, since entries are paired via matches, we'll use a custom split where we take random pairs in order to keep our data tidy.
Since we are making this model as our baseline, our hyperparameter for tree depth must be random initially. I picked 5 because that seems somewhat reasonable. 
Once we created the model and evaluated its performance on both our training and testing set, we see about 60% accurate predictions from our Decision Tree classifier. This isn't particularly good, because we would expect a 50% accurate prediction rate if we were to guess completely at random. However, for such little data provided and random hyperparameter selection, this is good news. We can most likely improve this by a lot!

### Final Model

As outlined in [Framing the Problem](#baseline), all of the variables that we selected contain information about how the game is progressing, especially focused around the 10 minute mark. Often in games, teams which can gain an early advantage often use that momentum to snowball into a win. Therefore, we will add all of the variables we selected to our model to see if we can improve on our baseline.
We did a few more transformations of our data before passing it into the model.
- `Kills, Deaths, Assists at 10` were transformed according to the proportion of K, D, and A's that the team had during the 