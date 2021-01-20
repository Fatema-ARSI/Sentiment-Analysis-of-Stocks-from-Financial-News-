# Stock-News-Sentiment-Analysis
# Google-App-rating-Prediction
## Project Overview:
- Created a model to predict the app rating,which are provided by the customers, with other information about the app provided.
- Featured engineering methods such as one hot encoding to get better predictive accuracy and integer encoding to understand relationship between each category.
- Optimised Linear and Random Forest Regressors to reach the best model.

## Code and Resources used:
**Python:** 3.7.6.

**Packages Used:** pandas,numpy,seaborn,matplotlib,scipy,sklearn

## Fields in the data :
- App: Application name
- Category: Category to which the app belongs 
- Rating: Overall user rating of the app
- Reviews: Number of user reviews for the app
- Size: Size of the app
- Installs: Number of user downloads/installs for the app
- Type: Paid or Free
- Price: Price of the app
- Content Rating: Age group the app is targeted at - Children / Mature 21+ / Adult
- Genres: An app can belong to multiple genres (apart from its main category). For example, a musical family game will belong to Music, Game, Family genres.
- Last Updated: Date when the app was last updated on Play Store
- Current Ver: Current version of the app available on Play Store
- Android Ver: Minimum required Android version

## Data Cleaning:
After loading the data I needed to clean it up.
Following are the changes made:
- Treated null values.
- Fixed incorrect type and inconsistent formatting of Size ,Installs,Reviews and Price.

Did Sanity checks: 
+ For average rating i.e between 1 to 5.
+ Drop the values where reviews were more than installs as only those installed can review the app.
+ For free apps ,the price should not be >0
+ Outliers treatment

## EDA:

Correlation between variables

<img src='download (1).png' width='300' height='300'>

Reviews affecting the Rating of Apps

<img src='download.png' width='300' height='300'>



## Model Building:

- Applied integer encoding on Categories ,Genres and Content Rating column.
- Converted classification Type into binary
- Created another dataframe for dummy variables for each category.

I chose 2 most common models i.e Linear  and Randome Forest Regressor and ran 4 regressions for each model used as we consider one-hot vs integer encoded results for the category section,as well as including/excluding the genres section.

## Model Performance:

Overall performance of both the model:

<img src='googleproject.jpg' width='500' height='500'>




