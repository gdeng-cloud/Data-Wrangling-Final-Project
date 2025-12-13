# Data Wrangling Final Project


## Project Objectives & Motivation

This project is to focus on how soccer teams can minimize goals conceded
through midfield defense. In modern soccer, midfielders has become the
engine of the team. They initiate attacks and break up opponent’s play
before the opponent reach the defensive area. With fast transitions and
high pressing becoming more common, controlling the midfield is
essential to limiting dangerous chances.

``` python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import pandas as pd
from statsmodels.iolib.summary2 import summary_col
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
```

This project used 7 seasons of the English Premire League team-level
data.

``` python
EPL = pd.read_csv("/Users/aleixsd11/Desktop/Data Wrangling Final Project/EPL.csv")
EPL
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Season | Rk | Squad | MP | W | D | L | GF | GA | GD | ... | Def | GCA | GCA90 | PassLive.1 | PassDead.1 | TO.1 | Sh_y.1 | Fld.1 | Def.1 | avg_poss |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 2023-2024 | 1 | Manchester City | 38 | 28 | 7 | 3 | 96 | 34 | 62 | ... | 11 | 56 | 1.47 | 41 | 3 | 5 | 4 | 3 | 0 | 66.628571 |
| 1 | 2023-2024 | 2 | Arsenal | 38 | 28 | 5 | 5 | 91 | 29 | 62 | ... | 23 | 44 | 1.16 | 24 | 4 | 4 | 5 | 2 | 5 | 56.728571 |
| 2 | 2023-2024 | 3 | Liverpool | 38 | 24 | 10 | 4 | 86 | 41 | 45 | ... | 20 | 61 | 1.61 | 37 | 7 | 6 | 7 | 3 | 1 | 61.700000 |
| 3 | 2023-2024 | 4 | Aston Villa | 38 | 20 | 8 | 10 | 76 | 61 | 15 | ... | 17 | 102 | 2.68 | 70 | 9 | 6 | 8 | 4 | 5 | 48.300000 |
| 4 | 2023-2024 | 5 | Tottenham | 38 | 20 | 6 | 12 | 74 | 61 | 13 | ... | 13 | 103 | 2.71 | 62 | 10 | 8 | 11 | 10 | 2 | 55.400000 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 135 | 2017-2018 | 16 | Huddersfield | 38 | 9 | 10 | 19 | 28 | 58 | -30 | ... | 12 | 99 | 2.61 | 70 | 6 | 6 | 8 | 6 | 3 | 46.450000 |
| 136 | 2017-2018 | 17 | Southampton | 38 | 7 | 15 | 16 | 37 | 56 | -19 | ... | 10 | 98 | 2.58 | 72 | 9 | 1 | 12 | 4 | 0 | 47.966667 |
| 137 | 2017-2018 | 18 | Swansea City | 38 | 8 | 9 | 21 | 28 | 56 | -28 | ... | 16 | 87 | 2.29 | 54 | 6 | 6 | 11 | 9 | 1 | 45.300000 |
| 138 | 2017-2018 | 19 | Stoke City | 38 | 7 | 12 | 19 | 35 | 68 | -33 | ... | 9 | 111 | 2.92 | 82 | 5 | 8 | 8 | 8 | 0 | 41.400000 |
| 139 | 2017-2018 | 20 | West Brom | 38 | 6 | 13 | 19 | 31 | 56 | -25 | ... | 11 | 100 | 2.63 | 67 | 7 | 14 | 7 | 5 | 0 | 39.300000 |

<p>140 rows × 57 columns</p>
</div>

There’s a clear nagative relationship between goals conceded and
ranking. The more goals a team allows, the lower the ranking.

``` python
plt.figure(figsize=(10, 6))
sns.regplot(
    x='GA',
    y='Rk',
    data=EPL,
    scatter_kws={'alpha': 0.6},
    line_kws={'color': 'red'},
)
plt.title("Relationship Between League Standing and Goals Against")
plt.xlabel("Goals Against")
plt.ylabel("League Standing")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 25)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

![](readme_files/figure-commonmark/cell-4-output-1.png)

I created a new feature called Weighted Tackles Efficiency in the Mid
Third, which reflects both the success rate of tackles and how often a
team is actually involved in midfield duels—adjusted by their average
possession. I made this adjustment because stronger team will have
higher possession rate and weaker teams will have lower posession rate
and it could make the data balanced.

``` python
EPL['Tkl_won_per'] = (EPL['TklW'] / EPL['Tkl']) * 100
EPL['Exp_Mid_3rd_won'] = (EPL['Tkl_won_per']/100) * EPL['Mid 3rd']
EPL['Mid_3rd_per'] = EPL['Exp_Mid_3rd_won'] / EPL['Mid 3rd']
EPL['weighted_Mid_3rd_per'] = EPL['Mid_3rd_per'] * (EPL['avg_poss'] / 100)
```

The results show that this new metric has a clear relationship with
defensive success. Teams with higher efficiency in the midfield tend to
concede fewer goals. This means winning the ball in midfield, especially
efficiently, is a strong defensive asset

``` python
plt.figure(figsize=(10, 6))
sns.regplot(
    x='weighted_Mid_3rd_per',
    y='GA',
    data=EPL,
    scatter_kws={'alpha': 0.6},
    line_kws={'color': 'red'},
)
plt.title("Relationship Between Goals Against and Weighted Tackles Efficiency in the Mid-Third")
plt.xlabel("Weighted Tackles Efficiency in the Mid-Third")
plt.ylabel("Goals Against")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
```

![](readme_files/figure-commonmark/cell-6-output-1.png)

These teams also allow fewer goal-creating actions. This shows that
strong midfield defending doesn’t just stop goals directly—it also
prevents dangerous plays from developing.

``` python
plt.figure(figsize=(10, 6))
sns.regplot(
    x='weighted_Mid_3rd_per',
    y='GCA90',
    data=EPL,
    scatter_kws={'alpha': 0.6},
    line_kws={'color': 'red'},
)
plt.title("Relationship Between Goals Created Actions Allowed per 90 Min and Weighted Tackles Efficiency in the Mid-Third")
plt.xlabel("Weighted Tackles Efficiency in the Mid-Third")
plt.ylabel("Goals Created Actions Allowed per 90 Min")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
```

![](readme_files/figure-commonmark/cell-7-output-1.png)

In the regression model, shot blocking and goals created actions allowed
per 90 minutes were the most significant predictors of goals conceded.
Surprisingly, midfield tackling alone wasn’t significant—but this may
reflect multicollinearity or overlapping effects in team defense.

``` python
EPLGA_lm = smf.ols(formula='GA ~ weighted_Mid_3rd_per + Sh_x + Pass + Int + GCA90', data = EPL).fit()
print(EPLGA_lm.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     GA   R-squared:                       0.969
    Model:                            OLS   Adj. R-squared:                  0.968
    Method:                 Least Squares   F-statistic:                     835.8
    Date:                Fri, 12 Dec 2025   Prob (F-statistic):           3.92e-99
    Time:                        18:36:57   Log-Likelihood:                -326.34
    No. Observations:                 140   AIC:                             664.7
    Df Residuals:                     134   BIC:                             682.3
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    ========================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------
    Intercept                1.4962      3.684      0.406      0.685      -5.789       8.782
    weighted_Mid_3rd_per     3.1676      6.429      0.493      0.623      -9.548      15.883
    Sh_x                     0.0180      0.009      2.086      0.039       0.001       0.035
    Pass                     0.0049      0.006      0.793      0.429      -0.007       0.017
    Int                     -0.0063      0.004     -1.788      0.076      -0.013       0.001
    GCA90                   21.0987      0.469     45.023      0.000      20.172      22.026
    ==============================================================================
    Omnibus:                        0.897   Durbin-Watson:                   2.135
    Prob(Omnibus):                  0.639   Jarque-Bera (JB):                0.919
    Skew:                           0.042   Prob(JB):                        0.631
    Kurtosis:                       2.612   Cond. No.                     1.69e+04
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.69e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.
