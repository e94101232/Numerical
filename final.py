import requests
import pandas as pd
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import webbrowser

def get_lat_long(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data:
        location = data[0]
        return location["lat"], location["lon"]
    else:
        return None, None

# 讀取地址文件
addresses = pd.read_csv('addresses.csv', encoding='big5')

# 創建經緯度儲存結果
latitudes = []
longitudes = []

# 依序取得地址
for address in addresses['address']:
    lat, lng = get_lat_long(address)
    latitudes.append(lat)
    longitudes.append(lng)
    time.sleep(0.1)  # 延遲 0.1 秒
    
addresses['latitude'] = latitudes
addresses['longitude'] = longitudes

# 將結果保存到新的 CSV 文件中，指定編碼方式
addresses.to_csv('addresses_with_latlong.csv', index=False, encoding='utf-8')

# 讀取地址資料
addresses = pd.read_csv('addresses_with_latlong.csv')
locations = addresses[['latitude', 'longitude']].values
addr = addresses['address'].tolist()

# 計算距離矩陣
def distance_matrix(locations):
    num_locations = len(locations)
    dist_matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            dist_matrix[i, j] = np.linalg.norm(locations[i] - locations[j])
    return dist_matrix
    
dist_matrix = distance_matrix(locations)
# 最近鄰算法
def nearest_neighbor(dist_matrix):
    num_locations = len(dist_matrix)
    visited = [False] * num_locations
    route = [0]  # 從成功大學開始
    visited[0] = True
    total_distance = 0

    for _ in range(num_locations - 1):
        current_location = route[-1]
        nearest_dist = float('inf')
        nearest_location = None
        for j in range(num_locations):
            if not visited[j] and dist_matrix[current_location, j] < nearest_dist:
                nearest_dist = dist_matrix[current_location, j]
                nearest_location = j
        if nearest_location is not None:
            route.append(nearest_location)
            visited[nearest_location] = True
            total_distance += nearest_dist
        else:
            break

    return route, total_distance

route, total_distance = nearest_neighbor(dist_matrix)

# 將最佳路徑轉換為指定格式的列表
path_order = [(addr[i], locations[i, 0], locations[i, 1]) for i in route]

# 印出最佳路徑列表
'''
print("最佳路徑順序:")
for location in path_order:
    print(location)
# 結果可視化
plt.figure(figsize=(8, 8))
plt.scatter(locations[:, 1], locations[:, 0], color='blue')  
for i, (lat, lon) in enumerate(locations):
    if i==0:
        plt.text(lon, lat, f'initial', fontsize=12, color='red', weight='bold')
    else:
        plt.text(lon, lat, str(i), fontsize=12)
'''    
route_locations = locations[route]
plt.plot(route_locations[:, 1], route_locations[:, 0], color='red', marker='o') 
plt.title(f'Nearest Neighbor Route\nTotal Distance: {total_distance:.2f}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()
google_maps_url = "https://www.google.com/maps/dir/"
for address, lat, lon in path_order:
    google_maps_url += "/" + address
webbrowser.open(google_maps_url)