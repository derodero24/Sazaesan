
import re

import pandas as pd

file = 'サザエさんじゃんけん.txt'
first_year = 1991

df = pd.DataFrame(
    columns=['year', 'month', 'day', 'rock', 'scissors', 'paper'])

for i, line in enumerate(open(file, 'r', encoding='utf-8')):
    year = first_year + i
    month_day = re.findall('[0-9]*/[0-9]*', line)
    month = [m_d.split('/')[0] for m_d in month_day]
    day = [m_d.split('/')[1] for m_d in month_day]
    hand = re.findall('(グー|チョキ|パー)', string=line)

    for m, d, h in zip(month, day, hand):
        # 手変換
        if h == 'グー':
            h = [1,0,0]
        elif h == 'チョキ' :
            h = [0,1,0]
        elif h == 'パー':
            h = [0,0,1]
        # 行追加
        row = pd.Series([year, m, d] + h, index=df.columns)
        df = df.append(row, ignore_index=True)


print(df)
df.to_csv('サザエさんじゃんけん.tsv', sep='\t', index=None)
