#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPHA 30536
Data and Programming - Final Project
@authors: ojfarrell & ashutayal

"""
# Import libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from statsmodels.formula.api import ols
pd.set_option('mode.chained_assignment', None) #JL: gagging the warning is not the correct approach, when you can just fix the code causing it
# source: https://www.dataquest.io/blog/settingwithcopywarning/

"""
General grading comments:
- All matplotlib operations should be done on an axis object, and not on plt, except for the start and end, e.g. plt.subplots or plt.show
- Your code started out making good use of functions, then stopped half way through.
- 
"""
# Defining functions


def clean_merge(dispositions, initiation, merge_cols):
    disposistions.loc[disposistions.GENDER.isin(unknown), ['GENDER']] = 'Unknown' #JL: unknown is a global variable you don't define for 200 lines
    initiation.loc[initiation.GENDER.isin(unknown), ['GENDER']] = 'Unknown'
    merged = initiation.merge(disposistions, how='inner', on=merge_cols)
    merged['DISPOSITION_DATE_2'] = pd.to_datetime(merged['DISPOSITION_DATE'],
                                                  infer_datetime_format=True,
                                                  errors='coerce')
    # source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html
    merged['YEAR'] = pd.DatetimeIndex(merged['DISPOSITION_DATE_2']).year
    merged = merged[merged['YEAR'].notna()]
    merged['YEAR'] = merged['YEAR'].astype(int)
    df = merged[merged['YEAR'].isin(years)]

    # redefining race labels
    df.loc[(df['RACE'] == "HISPANIC") |
           (df['RACE'] == "White/Black [Hispanic or Latino]") |
           (df['RACE'] == "White [Hispanic or Latino]"),
           "RACE"] = "Latinx"
    df.loc[(df['RACE'] == "ASIAN"), "RACE"] = "Asian"
    df['RACE'] = df['RACE'].fillna('Unknown')
    df['VERDICT'] = df['CHARGE_DISPOSITION'].map(xwalk)

    # Create Variable for Change in Charge
    df['CHANGE'] = (df['DISPOSITION_CHARGED_OFFENSE_TITLE']
                    == df['CHARGE_OFFENSE_TITLE']).astype(int)
    return df

# Create Summary Table


def summary_stats(df, year):

    df = df[df['YEAR'] == year]
    a = df.groupby(['VERDICT'])['CASE_ID'].count() #JL: use better names than a and b
    b = df.groupby(['RACE'])['CASE_ID'].count()

    #JL: this code is copy-pasted twice; instead use a loop, containers, or functions
    a.to_frame()
    a = a.reset_index()
    a = a.rename(columns={'VERDICT': 'Verdict',
                          'CASE_ID': 'Count'})
    a['Percentage'] = a['Count']/sum(a['Count'])
    a.to_csv(os.path.join(path, f'Verdict-{year}.csv'))

    b.to_frame()
    b = b.reset_index()
    b = b.rename(columns={'RACE': 'Race',
                          'CASE_ID': 'Count'})
    b['Percentage'] = b['Count']/sum(b['Count'])
    b.to_csv(os.path.join(path, f'Race-{year}.csv'))


def summarize(df):
    output = pd.DataFrame(columns=["Year",
                                   "Race",
                                   "Verdict",
                                   "Count",
                                   "Changed"])

    for year in years:
        for race in races:
            for verdict in verdicts:
                count = len(df.loc[(df.YEAR == year) & (df.RACE == race) & (df.VERDICT == verdict)])
                changed = len(df.loc[(df.YEAR == year) & (df.RACE == race)
                                     & (df.VERDICT == verdict) & (df.CHANGE == 1)])
                output = output.append({"Year": year,
                                        "Race": race,
                                        "Verdict": verdict,
                                        "Count": count,
                                        "Changed": changed}, ignore_index=True)

    # Type conversions
    output['Year'] = output['Year'].astype(int)
    output['Count'] = output['Count'].astype(int)
    output['Race'] = output['Race'].astype(str)
    output['Verdict'] = output['Verdict'].astype(str)
    output['Changed'] = output['Changed'].astype(int)

    # Create Variable for Percent with Changed Charges
    output['Percent'] = 0
    output['Percent'] = output['Changed'].div(output['Count']).replace(np.nan, 0)

    return(output) #JL: it will work, but don't use parentheses here

# Graphing Verdicts Over Time by Race


def race_lineplt(dataframe, verdict):
    data = dataframe.loc[dataframe['Verdict'] == verdict]
    sns.lineplot(data=data, x='Year', y='Count', hue='Race')
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    plt.xticks([2015, 2016, 2017, 2018, 2019], fontsize=16)
    plt.yticks(fontsize=16)

    # moving legend to outside plot
    # source: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    plt.legend(bbox_to_anchor=(1, 0.6),
               fontsize=12,
               bbox_transform=plt.gcf().transFigure)
    return plt #JL: this is returning a pointer to the library itself for some reason


# Graph Bar Plots for Verdict by Race by Time
def plot_for_year(df, year):
    fig, ax = plt.subplots()
    df = df[df['Year'] == year]
    sns.set()
    sns.set_style(style='ticks')

    ax = sns.barplot(x="Verdict", y='Count', hue="Race",
                     edgecolor=(0, 0, 0), linewidth=0.5, data=df) #JL: this just overwrites the first axis you created above

    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('Verdict', fontsize=12)
    ax.set_title(f'Verdict Counts by Race - {year}', fontsize=14)

    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Adding hatches
    # source: https://matplotlib.org/3.1.1/gallery/shapes_and_collections/hatch_demo.html
    bars = ax.patches
    patterns = ['/', '|', 'o']
    hatches = []
    for h in patterns:
        for i in range(int(len(bars) / len(patterns))):
            hatches.append(h)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.legend()

    plt.subplots_adjust(left=0.15)
    sns.despine(top=True, right=True)

    # set line colors to black
    # source: https://stackoverflow.com/questions/41709257/how-to-change-the-plot-line-color-from-blue-to-black
    plt.setp(ax.artists, edgecolor='k')
    plt.setp(ax.lines, color='k')

    # move legend outside
    # source: https://matplotlib.org/3.1.1/gallery/shapes_and_collections/hatch_demo.html
    plt.legend(bbox_to_anchor=(1, 0.6),
               bbox_transform=plt.gcf().transFigure)

    plt.savefig(os.path.join(path, f'barplot-{year}.jpg'), dpi=300)
    plt.show()
    plt.close()


# Graph Change in Charge by Race
def charges_lineplt(dataframe, verdict):
    data = dataframe.loc[dataframe['Verdict'] == verdict]
    sns.lineplot(data=data, x='Year', y='Percent', hue='Race')
    plt.xlabel('Year', fontsize=18)
    plt.yticks(fontsize=16)
    plt.xticks([2015, 2016, 2017, 2018, 2019], fontsize=16)
    plt.ylabel('Percent of Charges Changed', fontsize=18)
    plt.legend(bbox_to_anchor=(1, 0.6),
               fontsize=12,
               bbox_transform=plt.gcf().transFigure)
    return plt


# Summarizes the df to calculate counts of felonies per city per year
def export_csv(table):
    table.to_frame()
    # source: https://stackoverflow.com/questions/10373660/converting-a-pandas-groupby-output-from-series-to-dataframe
    table = table.reset_index()
    table = table.rename(columns={'INCIDENT_CITY': 'MUNICIPALI'})
    table.to_csv(os.path.join(path, 'class_summary.csv'))

# returns dfs for regression


def regression_dfs(df):
    reg_verdicts = ['Guilty', 'Not Guilty']
    reg_data = df[(df['VERDICT'].isin(reg_verdicts)) & (df['RACE'].isin(races))]
    dummy_verdict = {'Guilty': 1, 'Not Guilty': 0}
    reg_data['VERDICT_DUMMY'] = reg_data['VERDICT'].map(dummy_verdict)

    # Dummy Variables for RACE
    reg_data['BLACK'] = 0
    reg_data['LATINX'] = 0
    reg_data.loc[(reg_data['RACE'] == 'Black'), 'BLACK'] = 1
    reg_data.loc[(reg_data['RACE'] == 'Latinx'), 'LATINX'] = 1

    reg_2_data = reg_data[(reg_data['VERDICT'] == 'Guilty')]
    return reg_data, reg_2_data


# Importing datasets and initializing
path = os.path.join(os.getcwd(), 'Documents', 'GitHub', 'final-project-oiivia-ashu')
initiation = pd.read_csv(os.path.join(path, 'Initiation.csv'), low_memory=False)
disposistions = pd.read_csv(os.path.join(path, 'Dispositions.csv'), low_memory=False)
merge_cols = ['CASE_ID', 'CASE_PARTICIPANT_ID', 'ARRAIGNMENT_DATE',
              'AGE_AT_INCIDENT', 'RACE', 'GENDER', 'INCIDENT_CITY',
              'INCIDENT_BEGIN_DATE', 'INCIDENT_END_DATE', 'ARREST_DATE']
years = [i for i in range(2015, 2020)] #JL: unpacking the generator using a list comprehension helps nothing, e.g. years = range(2015, 2020) would have done the same thing
verdicts = ['Guilty', 'Not Guilty', 'Dropped']
races = ['Black', 'White', 'Latinx']
unknown = ['Male name, no gender given', 'Unknown Gender', np.nan]
change_verdicts = ['Guilty']

# Grouping Verdicts (Guilty, Not Guilty, Dropped, Other/Undetermined)
# Sources: Data dictionary and legal websites
xwalk = {'Nolle Prosecution': 'Dropped',
         'Plea Of Guilty': 'Guilty',
         'FNG': 'Not Guilty',
         'FNPC': 'Dropped',
         'Finding Guilty': 'Guilty',
         'Verdict Guilty': 'Guilty',
         'Verdict-Not Guilty': 'Not Guilty',
         'Death Suggested-Cause Abated': 'Dropped',
         'BFW': 'Dropped',
         'Superseded by Indictment': 'Dropped',
         'Finding Not Not Guilty': 'Not Guilty',
         'Case Dismissed': 'Dropped',
         'Finding Guilty - Lesser Included': 'Guilty',
         'FNG Reason Insanity': 'Guilty',
         'Plea of Guilty - Amended Charge': 'Guilty',
         'SOL': 'Dropped',
         'Mistrial Declared': 'Dropped',
         'Transferred - Misd Crt': 'Dropped',
         'Nolle On Remand': 'Dropped',
         'Charge Vacated': 'Dropped',
         'Plea of Guilty - Lesser Included': 'Guilty',
         'WOWI': 'Undetermined',
         'Verdict Guilty - Lesser Included': 'Dropped',
         'Withdrawn': 'Dropped',
         'Finding Guilty But Mentally Ill': 'Guilty',
         'Plea of Guilty But Mentally Ill': 'Guilty',
         'Finding Guilty - Amended Charge': 'Guilty',
         'Sexually Dangerous Person': 'Guilty',
         'Charge Rejected': 'Dropped',
         'Hold Pending Interlocutory': 'Undetermined',
         'SOLW': 'Dropped',
         'Verdict Guilty - Amended Charge': 'Guilty',
         'Charge Reversed': 'Not Guilty',
         'TRAN-FAI': 'Undetermined',
         'Verdict Guilty BUt Mentally Ill': 'Guilty',
         'Nolle Pros - Aonic': 'Dropped'}

# create df
df = clean_merge(disposistions, initiation, merge_cols)
summary = summarize(df)

# Output summary statistics
for year in years:
    summary_stats(df, year)

# Line plots: Verdicts Over Time by Race
for verdict in verdicts:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set()
    sns.set_style(style='ticks')
    # source: https://seaborn.pydata.org/generated/seaborn.despine.html
    sns.despine(top=True, right=True)
    plot = race_lineplt(summary, verdict)
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.title(verdict, fontsize=18)
    plt.savefig(os.path.join(path, f'{verdict}.jpg'), dpi=300)
    plt.show()
    plt.close()

# bar plot: Verdict by Race by Time
for year in years:
    plot_for_year(summary, year)


# Line plots: Change in Charge by Race
for verdict in change_verdicts:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set()
    sns.set_style(style='ticks')
    sns.despine(top=True, right=True)
    plot = charges_lineplt(summary, verdict)
    plt.title(verdict, fontsize=20)
    plt.savefig(os.path.join(path, f'{verdict}_charge_changed.jpg'), dpi=300)
    plt.show()
    plt.close()

# Export the summary to CSV to import in Jupyter Notebooks for Interactive plot
table = df.groupby(['YEAR', 'DISPOSITION_CHARGED_CLASS', 'INCIDENT_CITY'])['CASE_ID'].count()
export_csv(table)


# Regressions
# souce: https://www.statsmodels.org/stable/index.html
# source: https://stackoverflow.com/questions/55738056/using-categorical-variables-in-statsmodels-ols-class

reg_1_data, reg_2_data = regression_dfs(df)

# First regression : Impact of Race on Charge

fit = ols('VERDICT_DUMMY ~ BLACK + LATINX + C(GENDER) + AGE_AT_INCIDENT + \
        C(COURT_NAME) + YEAR', data=reg_1_data).fit()

print(fit.summary())


# Second Regression : Impact of Race on Change in Charge from Intiation to Disposition

fit2 = ols('CHANGE ~ BLACK + LATINX + C(GENDER) + AGE_AT_INCIDENT + \
        C(COURT_NAME) + YEAR', data=reg_2_data).fit()

print(fit2.summary())
