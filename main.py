import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup

array_a = np.array([[1,2,3],[4,5,6],[7,8,9]])

array_b = np.array([3,2,1])

print(array_a + array_b)
"""
excel_file_path = 'office_info.xlsx'
df = pd.read_excel(excel_file_path)

print(df.head(2))

for column in df.columns:
    df[column] = df[column].str.replace(r'\W',"", regex=True)
print(df)
df.to_excel("removed_characters.xlsx")
"""


"""
excel_file_path = "Financial Sample.xlsx"
df = pd.read_excel(excel_file_path)
#print(df)

# Time series
df_vtt_canada = df.loc[(df['Country'] == 'Canada') & (df['Product'] == 'VTT') & (df['Segment'] == 'Government')]
df_vtt_canada = df_vtt_canada.sort_values(by=['Date'])
df_vtt_canada.plot(x='Date', y='Profit')
plt.show()

df_products = df.groupby(['Product']).sum()
df_products['Units Sold'].plot.pie()
plt.show()

df_products['Units Sold'].plot.bar()
plt.show()
"""
# --------------------------------------------------------
#response = requests.get('https://trn.nipex-ng.com/sap/bc/zcontacts?sap-client=300' )
#print(response.text)
#response = requests.get('https://www.derricksherrill.com')
#print(response.status_code)

#soup = BeautifulSoup(response.text, 'html_parser')
#print(soup.find_all('a')[1:5:1])
# --------------------------------------------------------
# excel_book_1_relative_path = 'Purchases - Home B.xlsx'
# excel_book_prices = 'PriceBook.xlsx'

# df_prices = pd.read_excel(excel_book_prices)
# df_home_1 = pd.read_excel(excel_book_1_relative_path)

# print(df_prices, df_home_1)

# df_total = df_home_1.merge(df_prices, on='ID')

# df_total['Total Price'] = df_total['PURCHASED AMOUNT'] * df_total['Price']

# print(df_total)

# fig = px.pie(df_total[['MATERIAL', 'Total Price']], values='Total Price', names='MATERIAL')
# fig.show()


# ---------------------------------------------------------------------------------------------------
# stock_df = pd.read_html(
#    'https://github.com/Derrick-Sherrill/DerrickSherrill.com/blob/master/Sample%20Data/companylist.csv')
# df = pd.read_excel('stocks.xlsx')
# print(stock_df)
# print(df['Symbol'])
# increased_symbols = []

# for stock in df['Symbol']:
#    stock = stock.upper()
#    if '^' in stock:
#        pass
#    else:
#        try:
#            stock_info = yf.Ticker(stock)
#            hist = stock_info.history(period="5d")
#            previous_averaged_volume = hist['Volume'].iloc[1:4:1].mean()
#            todays_volume = hist['Volume'][-1]
#            if todays_volume > previous_averaged_volume * 4:
#                increased_symbols.append(stock)
#        except:
#            pass

# print(increased_symbols)

# ---------------------------------------------------------------------------------------------------
# scores_df = pd.read_excel('sample_scores.xlsx')
# print(scores_df)

# scores_df['average'] = scores_df.mean(axis=1, numeric_only=True)

# scores_df['Pass/Fail'] = np.where(scores_df['average'] > 60, 'Pass', 'Fail')
# print(scores_df)

# conditions = [
#    (scores_df['average'] >= 90),
#    (scores_df['average'] < 90) & (scores_df['average'] >= 80),
#    (scores_df['average'] < 80) & (scores_df['average'] >= 70),
#    (scores_df['average'] < 70) & (scores_df['average'] >= 60),
#    (scores_df['average'] < 60)
# ]
# results = ['A', 'B', 'C', 'D', 'F']

# scores_df['Letter Grade'] = np.select(conditions, results)
# print(scores_df)


# ---------------------------------------------------------------------------------------------------
# df = pd.read_excel('OrderHistory.xlsx')
# df = df.sort_values(by='Order ID', ascending=True)
# print(df.head(10))

# agg_functions = {
# 'Sales Amount'  : 'sum',
# 'Order Type'    : ', '.join,
# }

# simplified_df = df.groupby('Order ID').agg(agg_functions)
# simplified_df.to_excel('simplified_order_history.xlsx')

# multiple_item_orders = simplified_df[simplified_df['Order Type'].str.contains(',')]
# print(multiple_item_orders)


# ---------------------------------------------------------------------------------------------------
# read_excel for excel file and read_csv for csv file
# df_gold_prices = pd.read_csv('monthly_csv.csv')

# Viewing data: head(20) for first 20 data and tail(20) for last 20 data
# print(df_gold_prices.tail(20))

# dates = df_gold_prices['Date']
# prices = df_gold_prices['Price']

# simple operations
# df_gold_prices['buy_price'] = prices * .9
# print(df_gold_prices['Price'].max())
# df_gold_prices['Date'] = df_gold_prices['Date'].str.replace('-', '')

# print(df_gold_prices)

# fig = px.line(df_gold_prices, x = dates, y = prices, title = 'Gold Prices over Time')
# fig.show()
# ------------
# plt.plot(dates, prices)
# Previous customizations
# plt.xlabel('Dates')
# plt.ylabel('Prices')
# plt.title('Price of Gold over the year')
# plt.show()


# -------------------------------------------------------------------------------------------------------
# excel_file_path = 'electric_motor_data.csv'

# df = pd.read_csv(excel_file_path)
# print(df.columns)

# df_info = df.info()
# print(df_info)

# print(df.describe())
# print(df.describe()['coolant'])

# grouped_df = df.groupby(['profile_id']).max()
# print(grouped_df['torque'])

# profile_id_4_df = df[df['profile_id'] == 4]
# profile_id_4_df.to_excel('output.xlsx')
