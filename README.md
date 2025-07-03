# Statistical-regression-Gender-Inequality-worldwide.

# Gender Inequality and Its Determinants – A Regression Analysis

This project investigates the structural factors contributing to gender inequality across different countries using multivariate regression analysis. The target variable is the Gender Inequality Index (GII), a UN-derived measure ranging from 0 to 1, where higher values indicate greater inequality. The study draws from publicly available datasets covering a range of social, economic, and political indicators. 

An original dataset was prepared while referring to different international government and UN datasets. 

These include literacy rates, poverty levels, democracy scores, healthcare coverage, the presence of a state religion, conflict levels via the Global Peace Index (GPI), and the prevalence of child marriage. All data was compiled and standardized to represent conditions around the year 2021 for 138 countries.

The analysis began with visual exploration through scatter plots, revealing that poverty, political instability, religion, and child marriage tend to correlate positively with GII, while higher literacy, stronger democracy, and better health coverage associate with lower inequality. A full multivariate regression model including all variables achieved a high adjusted R² of 0.8636, indicating that about 86% of the variation in gender inequality can be explained by the selected features. To improve model interpretability without losing explanatory power, model selection techniques were applied. Both step-up and step-down procedures yielded a simpler model with three core predictors: healthcare access (uhssci), political instability (gpi), and child marriage rate (chmarriage). All three variables were statistically significant and aligned with the theoretical expectations of how structural and cultural factors affect gender disparities.

An alternate model was generated using Akaike Information Criterion (AIC), which added poverty and religion to the final model. While AIC values slightly favored the larger model, the added predictors did not show statistical significance at conventional thresholds. Root Mean Square Error (RMSE) was computed on a test split of the data (80/20), and results confirmed that the simpler three-variable model slightly outperformed the larger one (RMSE = 0.063 vs. 0.067), justifying its selection based on parsimony and generalizability. All regression assumptions were verified through residual analysis, normality tests (Shapiro-Wilk, QQ plot), and multicollinearity checks using Variance Inflation Factors (VIF).

To further understand how gender inequality varies by economic status, the countries were grouped into high, middle, and low income based on poverty rates. Using Wilcoxon rank-sum tests with Bonferroni correction, the analysis confirmed that GII differs significantly across these groups, reinforcing the observed positive correlation between poverty and inequality.

In conclusion, the study highlights healthcare availability, conflict levels, and child marriage as key predictors of gender inequality worldwide. While the analysis is statistically robust, it acknowledges limitations such as data gaps and the challenges of quantifying cultural variables like religion. Future extensions could involve more nuanced cultural modeling or nonlinear approaches, but the findings already offer valuable insight for policy and global development discussions.

