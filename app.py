import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

data = pd.read_csv('tripadvisor_cleaned_final.csv')

# 预处理
data['rating'] = data['rating'].fillna(0)
data['priceRange_clean'] = data['priceRange_clean'].fillna(0)
data['reviewTags_combined'] = data['reviewTags_combined'].fillna('')
data = data.dropna(subset=['latitude', 'longitude'])

# 聚类（考虑位置、评分、价格）
features = data[['latitude', 'longitude', 'priceRange_clean', 'rating']]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=6, random_state=0)
data['cluster'] = kmeans.fit_predict(data_scaled)

# 初始化 app
app = Dash(__name__)

# 主地图图层
map_fig = px.scatter_mapbox(
    data,
    lat='latitude', lon='longitude', color='cluster',
    hover_name='name',
    hover_data={'rating': True, 'priceRange_clean': True, 'cluster': True},
    zoom=10,
    height=700,
    mapbox_style='carto-positron',
)

# Layout
app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id='cluster-map',
            figure=map_fig,
            style={'width': '100%', 'height': '700px'}
        )
    ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Div([
        html.H4("该区域关键词 Top10"),
        dcc.Graph(id='keyword-bar'),
        html.H4("评分分布"),
        dcc.Graph(id='rating-hist'),
        html.H4("价格分布"),
        dcc.Graph(id='price-hist')
    ], style={'width': '38%', 'display': 'inline-block', 'paddingLeft': '1%'})
])

# 提取关键词函数
def extract_keywords(text_series, topn=10):
    all_text = ' '.join(text_series)
    words = re.findall(r'\w+', all_text.lower())
    filtered = [w for w in words if len(w) > 1]
    common = Counter(filtered).most_common(topn)
    return pd.DataFrame(common, columns=['word', 'count'])

# 回调函数
def gen_hist(df, col, nbins=10, title=''):
    return px.histogram(df, x=col, nbins=nbins, title=title)

@app.callback(
    [Output('keyword-bar', 'figure'),
     Output('rating-hist', 'figure'),
     Output('price-hist', 'figure')],
    Input('cluster-map', 'hoverData')
)
def update_dashboard(hoverData):
    if hoverData is None:
        return go.Figure(), go.Figure(), go.Figure()
    
    point_info = hoverData['points'][0]
    cluster_id = point_info['customdata'][2]
    cluster_df = data[data['cluster'] == cluster_id]

    keywords = extract_keywords(cluster_df['reviewTags_combined'])
    keyword_fig = px.bar(keywords, x='word', y='count', title='评论关键词 Top10')
    rating_fig = gen_hist(cluster_df, 'rating', 10, '评分分布')
    price_fig = gen_hist(cluster_df, 'priceRange_clean', 10, '价格分布')
    return keyword_fig, rating_fig, price_fig

if __name__ == '__main__':
    app.run(debug=True)