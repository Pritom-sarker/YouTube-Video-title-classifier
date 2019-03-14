import sqlite3
import pandas as pd
# Create your connection.
cnx = sqlite3.connect('youtube 1.db')

df = pd.read_sql_query("SELECT * FROM youtube_review", cnx)
df1=pd.DataFrame()
df1['type']=df['type']
df1['video_name']=df['video_name']

df1.to_csv('youtube1.csv')
print(df.head())

