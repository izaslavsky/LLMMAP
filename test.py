import geopandas as gpd

gdf = gpd.read_file('dataset/Jordan Purchasing Power/governorate.geojson')
gdf = gdf.iloc[:, 43:45]
print(gdf)
