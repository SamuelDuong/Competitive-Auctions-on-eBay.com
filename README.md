# Competitive-Auctions-on-eBay.com


---
author: "Quan Duong"
---

## The file eBayAuctions.csv contains information on 1972 auctions conducted on eBay.com

Between May and June 2004. The *objective* is to utilize this dataset to
construct a model for distinguishing competitive auctions from
non-competitive ones. A competitive auction is defined as an auction
where at least two bids are placed on the auctioned item. The dataset
encompasses variables describing the auctioned item (auction category),
the seller (their eBay rating), and the seller-selected auction terms
(auction duration, opening price, currency, day-of-week of auction
closure)

### We will import the dataset. Remove Category, Currency, EndDay variables from the imported dataset.We then split the data into training and validation datasets using a 60%-40% ratio.

```{r cars}
ebay.df <- read.csv("C:\\Users\\duong\\OneDrive\\Desktop\\QUAN\\Spring 2024\\Big Data Analytics\\eBayAuctions.csv")
ebay.df <- subset(ebay.df, select=-c(Category, Currency, EndDay)) # D
set.seed(1)  
train.index <- sample(c(1:dim(ebay.df)[1]), dim(ebay.df)[1]*0.6)  
train.df <- ebay.df[train.index, ]
valid.df <- ebay.df[-train.index, ]

competitive.ct <- rpart(Competitive ~ ., data = train.df,
                        control = rpart.control(minsplit = 6),
                    method = "class")
summary(competitive.ct)
```

## 3a. Including Plots

```{r pressure, echo=FALSE}
prp(competitive.ct, type = 3, extra = 101,  clip.right.lab = FALSE, box.palette = "GnYlRd", 
    branch = .3, varlen = -10, cex.main=3, space=0)  
```

# 3b. Summary Decision Rule for Predicting Competitiveness

### "Not Competitive" (0):

-   Generally, items with a lower `ClosePrice` compared to their
    `OpenPrice` are less likely to be competitive. Exceptions are when
    both `OpenPrice` and `ClosePrice` are very high, which then also
    considers `SellerRating`.
-   Specific Conditions:
    -   If `OpenPrice` \< 1.2 AND `ClosePrice` \< 1.0.
    -   If `OpenPrice` is between 1.2 to 2.4 AND `ClosePrice` \< 2.0.
    -   If `OpenPrice` is between 2.4 to 4.9 AND `ClosePrice` \< 4.2.
    -   If `OpenPrice` \>= 4.9 AND `ClosePrice` \< 10.0.
    -   If `OpenPrice` \>= 11.0 AND `ClosePrice` is between 10.0 to
        40.0.
    -   If `OpenPrice` \>= 38.2 AND `ClosePrice` \>= 40.0 AND
        `SellerRating` \>= 767.

### "Competitive" (1):

-   Items with a higher `ClosePrice` relative to their `OpenPrice` tend
    to be more competitive, especially when `ClosePrice` significantly
    exceeds `OpenPrice`, regardless of `SellerRating`.
-   Specific Conditions:
    -   If `OpenPrice` \< 1.2 AND `ClosePrice` is between 1.0 to 2.0.
    -   If `OpenPrice` \< 2.4 AND `ClosePrice` \>= 2.0.
    -   If `OpenPrice` is between 2.4 to 4.9 AND `ClosePrice` is between
        4.2 to 10.0.
    -   If `OpenPrice` is between 2.4 to 11.0 AND `ClosePrice` \>= 10.0.
    -   If `OpenPrice` \>= 11.0 AND `ClosePrice` \>= 40.0 AND
        `SellerRating` \< 767.
    -   If `OpenPrice` is between 11.0 to 38.2 AND `ClosePrice` \>= 40.0
        AND `SellerRating` \>= 767.

### General Observation:

-   Competitiveness is often linked to a higher `ClosePrice` relative to
    `OpenPrice`. Items with very high opening and closing prices may
    also need a good `SellerRating` to be considered competitive. Items
    with low `OpenPrice` and `ClosePrice` typically are not competitive,
    with competitiveness increasing as the gap between opening and
    closing prices widens.

### Our prediction confusion matrix of validation data: 

```{r pressure, echo=FALSE}
competitive.ct.point.pred.train <- predict(competitive.ct,train.df,type = "class")
predicted <- factor(competitive.ct.point.pred.train)
actual <- factor(train.df$Competitive)
# generate confusion matrix for training data
confusionMatrix(as.factor(competitive.ct.point.pred.train), as.factor(train.df$Competitive))
# 4-2. repeat the code for the validation data
competitive.ct.point.pred.valid <- predict(competitive.ct,valid.df,type = "class")
# generate confusion matrix for training data
confusionMatrix(as.factor(competitive.ct.point.pred.valid), as.factor(valid.df$Competitive))
modelVarImp <- varImp(competitive.ct, scale = FALSE)
print(modelVarImp)
```

**3c. Summary The performance of the decision tree model on the
validation data is quite promising, as evidenced by the results from the
confusion matrix. Here's a detailed summary of the model's predictive
capabilities:**

-   The model achieved an **overall accuracy of 86.57%**, indicating
    that it correctly identified the majority of competitive and
    non-competitive auctions in the validation set.
-   **Sensitivity** was particularly high at **90.58%**, which suggests
    the model is very effective at correctly identifying auctions that
    are competitive.
-   **Specificity** was also strong at **82.80%**, demonstrating the
    model's proficiency in recognizing auctions that are not
    competitive.
-   The model's **positive predictive value** (precision) stood at
    **83.17%**, and the **negative predictive value** was **90.35%**.
    These metrics reinforce the model's reliability in correctly
    classifying auctions.
-   **Balanced accuracy** was calculated to be **86.69%**, showing that
    the model performs consistently across both the competitive and
    non-competitive classes.

Overall, the decision tree model exhibits a high level of accuracy in
predicting auction competitiveness on the validation dataset. It holds
significant potential as a tool for auctioneers and sellers to devise
strategies for future auctions. However, it is crucial to note that the
model's utility is somewhat diminished in real-time auction scenarios
due to its reliance on the **ClosePrice** predictor, which is not known
until the auction concludes. Therefore, while the model is robust in
post-auction analysis, its application in predicting outcomes before an
auction ends is limited.

### 3d. Most Important Predictors

-   **OpenPrice** and **ClosePrice** are the most critical predictors
    with importance scores of 216.79 and 213.50, respectively. These
    suggest that the auction's opening and closing prices are highly
    influential in determining competitiveness.

**Significant Predictors**

-   **SellerRating** is also significant, with an importance score of
    94.67, highlighting the seller's rating as an essential factor in
    item competitiveness.
-   Categories such as **Art.Collectibles**, **Books**,
    **EverythingElse**, and **Health.Beauty** show notable importance,
    with scores ranging from 11.48 to 27.57, indicating their influence
    on an item's likelihood to be competitive.
-   **EndDay.Weekend** and **Currency.nonUS** have importance scores of
    13.82 and 5.65, suggesting that the auction's ending time and the
    currency used might impact its competitiveness.

**Less Significant or Insignificant Predictors**

-   Categories like **Home.Garden** and **Jewelry** have no importance
    score (0.00), suggesting they do not affect the competitiveness of
    an item.
-   **Duration**, **Computer.Electronics**, **Clothing.Toys**,
    **Coins.Stamps**, and **Music.Movie.Game** possess lower importance
    scores, indicating a less significant impact on competitiveness
    compared to the more critical predictors.

##4. In a practical scenario, we want to predict whether an auction will
be competitive ahead of time, so that we can make informed decisions to
potentially adjust strategy. This could include changing the opening
price, modifying the auction duration, or changing the time the auction
ends (for online auctions). For this kind of predictive modeling, we
need to rely on information that is available before the auction starts
or while it is in progress.

From the decision tree image, the **OpenPrice** and **SellerRating** are
used as predictors, and these are indeed practical because:

-   **OpenPrice** is set before the auction starts.
-   **SellerRating** is known in advance and is a static attribute of
    the seller.

However, any rule involving the **ClosePrice** is not practical for
real-time prediction because this information becomes available only
after the auction has concluded. If the goal is to predict the
competitiveness of an auction in real-time, using **ClosePrice** would
not be feasible.

To summarize, for a predictive model to be useful in a real-world
setting where the goal is to forecast auction outcomes before they
conclude, the model should only include predictors that are known prior
to the auction's end, such as **OpenPrice**, **SellerRating**, item
characteristics, and potentially the timing of the auction (such as
whether it ends on a weekend or not. **ClosePrice**, however, should be
excluded from such a predictive model.

##5a. This time, we will use only the predictors that can be used for
predicting the outcome of a new auction before the auction ends.

```{r pressure, echo=FALSE}
newt.ct <- rpart(Competitive ~ OpenPrice + SellerRating + Category.Art.Collectibles + Category.Books + Duration + Currency.nonUS + EndDay.Weekend, data = train.df,
                 control = rpart.control(minsplit = 6),
                 method = "class")
summary(newt.ct)
prp(newt.ct, type = 3, extra = 101,  clip.right.lab = FALSE, box.palette = "GnYlRd", 
    branch = .3, varlen = -10, cex.main=3, space=0) 
```

### 5b. Decision Tree Summary

Based on the fitted decision tree model, we can outline decision rules
for predicting whether an auction will be competitive (`class = 1`) or
not (`class = 0`). The rules are derived from predictors known before
the auction ends:

1.  **OpenPrice \>= 2.45**:
    -   If **SellerRating \< 3291.5**, predict the auction as not
        competitive (`class = 0`).
    -   If **SellerRating \>= 3291.5**:
        -   And if **EndDay.Weekend \< 0.5**, further decisions are
            based on other conditions not specified here due to the
            initial focus on `OpenPrice` and `SellerRating`.
2.  **OpenPrice \< 2.45**:
    -   This condition leads to a prediction of competitiveness directly
        based on `OpenPrice`, without additional rules provided in the
        summary.

### Interpretation

The decision rules indicate that both the `OpenPrice` of the item and
the `SellerRating` play crucial roles in determining the competitiveness
of an auction. Specifically:

-   A higher **OpenPrice** tends to indicate a non-competitive auction,
    especially in conjunction with a **SellerRating** that is lower than
    3291.5.
-   The model also considers **EndDay.Weekend** as a factor for auctions
    with higher **SellerRating**, suggesting the day of the week may
    influence the competitiveness for sellers with high ratings.

```{r pressure, echo=FALSE}
newt.ct.point.pred.train <- predict(newt.ct,train.df,type = "class")
predicted <- factor(newt.ct.point.pred.train)
actual <- factor(train.df$Competitive)
# generate confusion matrix for training data
confusionMatrix(as.factor(newt.ct.point.pred.train), as.factor(train.df$Competitive))
newt.ct.point.pred.valid <- predict(newt.ct,valid.df,type = "class")

confusionMatrix(as.factor(newt.ct.point.pred.valid), as.factor(valid.df$Competitive))

```

### 5c. Decision Tree Performance Summary

The second decision tree, excluding `ClosePrice`, shows a moderate
predictive accuracy on the validation data:

-   **Accuracy**: 68.69%, indicating a decent ability to predict auction
    outcomes.
-   **Sensitivity (Recall)**: 67.54%, showing its capability to identify
    competitive auctions.
-   **Specificity**: 69.78%, indicating effectiveness in recognizing
    non-competitive auctions.
-   **Positive Predictive Value (Precision)**: 67.72%, reflecting the
    accuracy of positive (competitive) predictions.
-   **Negative Predictive Value**: 69.61%, reflecting the accuracy of
    negative (non-competitive) predictions.
-   **Balanced Accuracy**: 68.66%, a summary measure showing overall
    effectiveness.
-   **Kappa**: 0.3732, indicating fair agreement beyond chance.

This model provides actionable insights before the auction concludes,
balancing between predictive accuracy and practical applicability for
real-time decisions. \### 5d. Predictors Used by the Second Decision
Tree The second decision tree, focused on pre-auction data, utilized the
following predictors to determine the competitiveness of an auction:

-   **OpenPrice**
-   **SellerRating**
-   **Duration**
-   **Currency.nonUS**
-   **EndDay.Weekend**
-   **Category.Art.Collectibles**

## 6. Comparison of Decision Tree Performance

We have developed two decision tree models to predict auction
competitiveness. The first model (from Q3) includes predictors known
both before and after the auction ends, including `ClosePrice`. The
second model (from Q5) uses only predictors known before the auction
ends, excluding `ClosePrice`.

### First Tree Performance

-   **Accuracy**: 86.57%
-   **Sensitivity**: 90.58%
-   **Specificity**: 82.80%
-   **Positive Predictive Value**: 83.17%
-   **Negative Predictive Value**: 90.35%
-   **Balanced Accuracy**: 86.69%
-   **Kappa**: 0.7318

This model shows a high degree of predictive accuracy, with strengths in
both sensitivity and specificity, indicating a strong ability to
correctly identify both competitive and non-competitive auctions.

### Second Tree Performance

-   **Accuracy**: 68.69%
-   **Sensitivity**: 67.54%
-   **Specificity**: 69.78%
-   **Positive Predictive Value**: 67.72%
-   **Negative Predictive Value**: 69.61%
-   **Balanced Accuracy**: 68.66%
-   **Kappa**: 0.3732

The second model, which does not include `ClosePrice` as a predictor,
shows a decrease in performance across all metrics compared to the first
model. The accuracy, sensitivity, and specificity are notably lower,
which indicates a reduced ability to distinguish between competitive and
non-competitive auctions accurately.

### Conclusion

The first decision tree model demonstrates better predictive performance
across all evaluated metrics. This suggests that including post-auction
information (specifically `ClosePrice`) significantly enhances the
model's ability to predict auction competitiveness accurately. However,
while the first model is more accurate, it is less practical for
real-time predictions since it relies on information not available until
after an auction concludes.

The second model, despite its lower accuracy, offers practical value by
relying solely on predictors available before the auction ends. This
makes it more suitable for making real-time predictions or strategic
decisions in advance of an auction's conclusion.

In summary, the choice between models depends on the specific
application requirements: higher accuracy with post-auction data or
real-time prediction capability with pre-auction data.
