import pickle
import plotly.express as px

with open('/mnt/data/dpietrzak/panda/umap_results.pkl', 'rb') as f:
    df = pickle.load(f)
    
    
configs = ['config_1','config_2','config_3','config_4','config_5']
for config in configs:
  px.scatter(df[config], x=df[config]['dim1'], y=df[config]['dim2'], color=df[config]['label'])
